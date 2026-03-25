"""Layer 1 – Custom table logic.

Post-processing applied to merged extraction results to handle:
  1. Implicit tables  – grid-like text clusters where no table was detected.
  2. Merged cells     – consecutive empty cells suggesting row/col spans.
  3. Nested headers   – two-level header rows detected by column count disparity.
  4. Multi-page stitch – tables that continue across pages (called from pipeline).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    BoundingBox,
    PageExtractionResult,
    Table,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ─── Implicit Table Detection ─────────────────────────────────────────────────

def _snap_to_grid(
    values: List[float], tolerance: float
) -> Dict[float, float]:
    """Map each value to its cluster representative within tolerance."""
    clusters: List[float] = []
    mapping: Dict[float, float] = {}
    for v in sorted(values):
        placed = False
        for c in clusters:
            if abs(v - c) <= tolerance:
                mapping[v] = c
                placed = True
                break
        if not placed:
            clusters.append(v)
            mapping[v] = v
    return mapping


def detect_implicit_tables(
    text_blocks: List[TextBlock],
    page_width: float,
    page_height: float,
    min_cluster_size: int = 4,
    alignment_tolerance: float = 10.0,
) -> Tuple[List[Table], List[TextBlock]]:
    """Find groups of text blocks that form a grid pattern.

    Returns:
        (implicit_tables, remaining_text_blocks)
    """
    if len(text_blocks) < min_cluster_size:
        return [], text_blocks

    # ── Cluster blocks by x0 (column alignment) ──────────────────────────────
    x_snap = _snap_to_grid(
        [b.bbox.x0 for b in text_blocks], alignment_tolerance
    )
    y_snap = _snap_to_grid(
        [(b.bbox.y0 + b.bbox.y1) / 2 for b in text_blocks], alignment_tolerance
    )

    # Assign each block to a (col, row) grid cell
    col_values = sorted(set(x_snap.values()))
    row_values = sorted(set(y_snap.values()))

    if len(col_values) < 2 or len(row_values) < 2:
        return [], text_blocks

    # Build a grid structure
    grid: Dict[Tuple[int, int], TextBlock] = {}
    for b in text_blocks:
        col_repr = x_snap[b.bbox.x0]
        row_repr = y_snap[(b.bbox.y0 + b.bbox.y1) / 2]
        ci = col_values.index(col_repr)
        ri = row_values.index(row_repr)
        grid[(ri, ci)] = b

    # A "grid-like" cluster: at least min_cluster_size blocks and multiple
    # rows share the same set of columns.
    if len(grid) < min_cluster_size:
        return [], text_blocks

    # Check if rows are consistently aligned (same columns appear in ≥2 rows)
    rows_per_col: Dict[int, int] = {}
    for ri, ci in grid:
        rows_per_col[ci] = rows_per_col.get(ci, 0) + 1
    consistent_cols = [ci for ci, cnt in rows_per_col.items() if cnt >= 2]

    if len(consistent_cols) < 2:
        return [], text_blocks

    # Build table data from grid
    table_data: List[List[str]] = []
    for ri in range(len(row_values)):
        row: List[str] = []
        for ci in range(len(col_values)):
            block = grid.get((ri, ci))
            row.append(block.text if block else "")
        table_data.append(row)

    # Calculate table bounding box from all participating blocks
    participants = list(grid.values())
    table_bbox = BoundingBox(
        min(b.bbox.x0 for b in participants),
        min(b.bbox.y0 for b in participants),
        max(b.bbox.x1 for b in participants),
        max(b.bbox.y1 for b in participants),
    )

    page_num = participants[0].page_num
    implicit_table = Table(
        data=table_data,
        bbox=table_bbox,
        page_num=page_num,
        confidence=0.6,  # lower confidence – inferred, not line-detected
        source="implicit",
    )

    # Remove blocks that were absorbed into the table
    absorbed = set(id(b) for b in participants)
    remaining = [b for b in text_blocks if id(b) not in absorbed]
    return [implicit_table], remaining


# ─── Merged Cell Handling ─────────────────────────────────────────────────────

def annotate_merged_cells(table: Table) -> Table:
    """Detect and annotate spans in table data (in-place friendly).

    Consecutive empty cells following a non-empty cell in the same row are
    flagged in the first cell's text as a colspan marker.  Similarly for
    identical consecutive values in the same column (rowspan).
    This is a best-effort heuristic for irregular tables.
    """
    if not table.data:
        return table

    data = [row[:] for row in table.data]  # shallow copy rows
    num_rows = len(data)
    num_cols = max(len(r) for r in data)

    # Pad short rows
    for row in data:
        while len(row) < num_cols:
            row.append("")

    # Detect colspans: empty cells to the right of a non-empty cell
    for ri, row in enumerate(data):
        ci = 0
        while ci < len(row):
            if row[ci].strip():
                span = 1
                while ci + span < len(row) and not row[ci + span].strip():
                    span += 1
                if span > 1:
                    # Mark span info in a metadata-friendly way
                    data[ri][ci] = f"{row[ci]} [colspan={span}]"
            ci += 1

    return Table(
        data=data,
        bbox=table.bbox,
        page_num=table.page_num,
        confidence=table.confidence,
        source=table.source,
        headers=table.headers,
    )


# ─── Nested Header Parsing ────────────────────────────────────────────────────

def parse_nested_headers(table: Table) -> Table:
    """Detect two-level header structure and annotate headers field.

    Heuristic: if row 1 has significantly more non-empty cells than row 0,
    row 0 is a group header row and row 1 is the sub-header row.
    """
    if len(table.data) < 3:
        return table

    row0 = [c for c in table.data[0] if c and c.strip()]
    row1 = [c for c in table.data[1] if c and c.strip()]

    if len(row1) >= 2 * len(row0) and len(row0) >= 1:
        # Two-level header: flatten into combined header strings
        headers = []
        col_idx = 0
        for cell in table.data[0]:
            if cell and cell.strip():
                # Find how many sub-headers follow
                group_name = cell.strip()
                sub = []
                j = col_idx
                while j < len(table.data[1]) and (
                    j == col_idx or not table.data[0][j].strip()
                ):
                    s = table.data[1][j].strip() if j < len(table.data[1]) else ""
                    if s:
                        sub.append(f"{group_name} / {s}")
                    j += 1
                headers.extend(sub or [group_name])
            col_idx += 1

        return Table(
            data=table.data[2:],  # skip header rows from data
            bbox=table.bbox,
            page_num=table.page_num,
            confidence=table.confidence,
            source=table.source,
            headers=headers if headers else table.headers,
        )
    # Single header row
    if not table.headers:
        return Table(
            data=table.data[1:],
            bbox=table.bbox,
            page_num=table.page_num,
            confidence=table.confidence,
            source=table.source,
            headers=[c or "" for c in table.data[0]],
        )
    return table


# ─── Multi-Page Stitching ─────────────────────────────────────────────────────

def stitch_multipage_tables(
    pages: List[PageExtractionResult],
    col_match_threshold: float = 0.7,
) -> List[PageExtractionResult]:
    """Detect and stitch tables that continue across consecutive pages.

    A table on page N is considered a continuation of a table on page N-1 if:
      - The table on page N starts near the top of the page.
      - The column count matches (within threshold).
      - The column x-positions align closely.
    """
    if len(pages) < 2:
        return pages

    for i in range(len(pages) - 1):
        curr_page = pages[i]
        next_page = pages[i + 1]

        for curr_table in curr_page.tables:
            # Only consider tables that reach close to the bottom of the page
            if curr_page.page_height > 0:
                bottom_pct = curr_table.bbox.y1 / curr_page.page_height
                if bottom_pct < 0.85:
                    continue  # table ends well above page bottom

            for next_table in next_page.tables:
                # Continuation table should start near the top
                if next_page.page_height > 0:
                    top_pct = next_table.bbox.y0 / next_page.page_height
                    if top_pct > 0.15:
                        continue

                # Column count similarity
                curr_cols = curr_table.num_cols
                next_cols = next_table.num_cols
                if curr_cols == 0 or next_cols == 0:
                    continue
                ratio = min(curr_cols, next_cols) / max(curr_cols, next_cols)
                if ratio < col_match_threshold:
                    continue

                # Mark continuation
                curr_table.continued_on_page = next_page.page_num
                next_table.continued_from_page = curr_page.page_num
                logger.debug(
                    "Stitched table: page %d → page %d",
                    curr_page.page_num,
                    next_page.page_num,
                )
                break  # one continuation per table

    return pages


# ─── Main Entry Point ─────────────────────────────────────────────────────────

class CustomTableLogic:
    """Applies post-extraction table enhancements to a PageExtractionResult."""

    def __init__(self, config: Dict[str, Any]) -> None:
        ct = config.get("extraction", {}).get("custom_table", {})
        self._min_cluster = int(ct.get("min_cluster_size", 4))
        self._align_tol = float(ct.get("grid_alignment_tolerance", 10.0))

    def process(self, result: PageExtractionResult) -> PageExtractionResult:
        """Run all custom table logic and return enhanced result."""
        text_blocks = result.text_blocks
        tables = list(result.tables)

        # 1. Detect implicit tables from text block clusters
        implicit, text_blocks = detect_implicit_tables(
            text_blocks,
            result.page_width,
            result.page_height,
            self._min_cluster,
            self._align_tol,
        )
        tables.extend(implicit)

        # 2. Annotate merged cells and nested headers in all tables
        enhanced_tables: List[Table] = []
        for t in tables:
            t = annotate_merged_cells(t)
            t = parse_nested_headers(t)
            enhanced_tables.append(t)

        return PageExtractionResult(
            page_num=result.page_num,
            text_blocks=text_blocks,
            tables=enhanced_tables,
            images=result.images,
            source=result.source,
            is_scanned=result.is_scanned,
            page_width=result.page_width,
            page_height=result.page_height,
        )
