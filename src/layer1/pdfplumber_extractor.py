"""Layer 1 – pdfplumber-based text and table extractor.

Extracts text blocks, tables, and image metadata from PDF pages using
pdfplumber. Supports both lattice (line-based) and stream (whitespace-based)
table detection modes.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from ..models import (
    BoundingBox,
    ExtractionParameters,
    ImageElement,
    PageExtractionResult,
    Table,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _is_scanned(page, scanned_threshold: int = 20) -> bool:
    """Return True if the page has too little selectable text (likely scanned)."""
    return len(page.chars) < scanned_threshold


def _words_to_blocks(
    words: List[Dict],
    line_gap: float = 5.0,
    block_gap: float = 15.0,
) -> List[Dict[str, Any]]:
    """Cluster pdfplumber word dicts into rectangular text blocks.

    Args:
        words: Output of page.extract_words().
        line_gap: Max vertical separation (pts) between words on the same line.
        block_gap: Max vertical separation (pts) between consecutive lines in
                   the same block.

    Returns:
        List of dicts with keys: text, x0, y0 (=top), x1, y1 (=bottom).
    """
    if not words:
        return []

    # pdfplumber word dict keys: text, x0, top, x1, bottom, ...
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))

    # ── Group words into lines ──────────────────────────────────────────────
    lines: List[List[Dict]] = []
    current_line: List[Dict] = [sorted_words[0]]
    for word in sorted_words[1:]:
        if abs(word["top"] - current_line[-1]["top"]) <= line_gap:
            current_line.append(word)
        else:
            lines.append(current_line)
            current_line = [word]
    lines.append(current_line)

    # ── Group lines into blocks ─────────────────────────────────────────────
    blocks_of_lines: List[List[List[Dict]]] = [[lines[0]]]
    for line in lines[1:]:
        prev_bottom = max(w["bottom"] for w in blocks_of_lines[-1][-1])
        curr_top = min(w["top"] for w in line)
        if curr_top - prev_bottom <= block_gap:
            blocks_of_lines[-1].append(line)
        else:
            blocks_of_lines.append([line])

    result: List[Dict[str, Any]] = []
    for block_lines in blocks_of_lines:
        all_words = [w for line in block_lines for w in line]
        text = " ".join(w["text"] for w in all_words).strip()
        if not text:
            continue
        result.append(
            {
                "text": text,
                "x0": min(w["x0"] for w in all_words),
                "y0": min(w["top"] for w in all_words),
                "x1": max(w["x1"] for w in all_words),
                "y1": max(w["bottom"] for w in all_words),
            }
        )
    return result


def _build_table_settings(params: ExtractionParameters) -> Dict[str, Any]:
    """Convert ExtractionParameters into a pdfplumber table_settings dict."""
    if params.pdfplumber_use_text_alignment:
        return {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": params.pdfplumber_snap_tolerance,
            "join_tolerance": params.pdfplumber_join_tolerance,
        }
    if params.pdfplumber_mode == "lattice":
        return {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": params.pdfplumber_snap_tolerance,
            "join_tolerance": params.pdfplumber_join_tolerance,
            "edge_min_length": params.pdfplumber_edge_min_length,
        }
    # stream mode
    return {
        "vertical_strategy": "text",
        "horizontal_strategy": "lines_strict",
        "snap_tolerance": params.pdfplumber_snap_tolerance,
        "join_tolerance": params.pdfplumber_join_tolerance,
        "edge_min_length": params.pdfplumber_edge_min_length,
    }


# ─── Extractor ────────────────────────────────────────────────────────────────

class PDFPlumberExtractor:
    """Extracts text and tables from a PDF page using pdfplumber."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        pp_cfg = config.get("extraction", {}).get("pdfplumber", {})
        self._line_gap = float(pp_cfg.get("text_block_line_gap", 5.0))
        self._block_gap = float(pp_cfg.get("text_block_block_gap", 15.0))

    def extract_page(
        self,
        pdf_path: str,
        page_num: int,
        params: ExtractionParameters,
    ) -> PageExtractionResult:
        """Extract content from a single page and return structured result."""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            return self._process_page(page, page_num, params)

    # ── Private ───────────────────────────────────────────────────────────────

    def _process_page(
        self,
        page,
        page_num: int,
        params: ExtractionParameters,
    ) -> PageExtractionResult:
        page_width: float = page.width
        page_height: float = page.height

        # Optional: crop to target region for retry passes
        working_page = page
        crop_offset: Tuple[float, float] = (0.0, 0.0)
        if params.target_bbox is not None:
            tb = params.target_bbox
            # pdfplumber crop expects (x0, top, x1, bottom)
            cropped = page.crop((tb.x0, tb.y0, tb.x1, tb.y1), relative=False)
            working_page = cropped
            crop_offset = (tb.x0, tb.y0)

        is_scanned = _is_scanned(page)
        tables = self._extract_tables(working_page, page_num, params, crop_offset)
        table_bboxes = [t.bbox for t in tables]
        text_blocks = self._extract_text_blocks(
            working_page, page_num, params, crop_offset, table_bboxes
        )
        images = self._extract_images(working_page, page_num, crop_offset)

        return PageExtractionResult(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=tables,
            images=images,
            source="pdfplumber",
            is_scanned=is_scanned,
            page_width=page_width,
            page_height=page_height,
        )

    def _extract_tables(
        self,
        page,
        page_num: int,
        params: ExtractionParameters,
        offset: Tuple[float, float],
    ) -> List[Table]:
        settings = _build_table_settings(params)
        try:
            found = page.find_tables(settings)
        except Exception as exc:
            logger.warning("pdfplumber table extraction failed p%d: %s", page_num, exc)
            return []

        results: List[Table] = []
        ox, oy = offset
        for table_obj in found:
            try:
                data = table_obj.extract()
            except Exception:
                continue
            if not data:
                continue
            # Clean None cells
            cleaned = [
                [cell if cell is not None else "" for cell in row]
                for row in data
            ]
            total = sum(len(r) for r in cleaned)
            filled = sum(1 for r in cleaned for c in r if c.strip())
            conf = filled / total if total > 0 else 0.0

            x0, y0, x1, y1 = table_obj.bbox
            results.append(
                Table(
                    data=cleaned,
                    bbox=BoundingBox(x0 + ox, y0 + oy, x1 + ox, y1 + oy),
                    page_num=page_num,
                    confidence=conf,
                    source="pdfplumber",
                )
            )
        return results

    def _extract_text_blocks(
        self,
        page,
        page_num: int,
        params: ExtractionParameters,
        offset: Tuple[float, float],
        table_bboxes: List[BoundingBox],
    ) -> List[TextBlock]:
        try:
            words = page.extract_words(
                x_tolerance=params.pdfplumber_word_x_tolerance,
                y_tolerance=params.pdfplumber_word_y_tolerance,
            )
        except Exception as exc:
            logger.warning("pdfplumber word extraction failed p%d: %s", page_num, exc)
            return []

        if not words:
            return []

        # Filter out words that fall inside detected table regions
        ox, oy = offset
        filtered_words = []
        for w in words:
            cx = (w["x0"] + w["x1"]) / 2 + ox
            cy = (w["top"] + w["bottom"]) / 2 + oy
            in_table = any(bb.contains_point(cx, cy) for bb in table_bboxes)
            if not in_table:
                filtered_words.append(w)

        raw_blocks = _words_to_blocks(filtered_words, self._line_gap, self._block_gap)
        result: List[TextBlock] = []
        for b in raw_blocks:
            result.append(
                TextBlock(
                    text=b["text"],
                    bbox=BoundingBox(
                        b["x0"] + ox, b["y0"] + oy, b["x1"] + ox, b["y1"] + oy
                    ),
                    page_num=page_num,
                    confidence=1.0,
                    source="pdfplumber",
                )
            )
        return result

    def _extract_images(
        self,
        page,
        page_num: int,
        offset: Tuple[float, float],
    ) -> List[ImageElement]:
        ox, oy = offset
        result: List[ImageElement] = []
        for img in page.images:
            result.append(
                ImageElement(
                    bbox=BoundingBox(
                        img["x0"] + ox,
                        img["top"] + oy,
                        img["x1"] + ox,
                        img["bottom"] + oy,
                    ),
                    page_num=page_num,
                )
            )
        return result
