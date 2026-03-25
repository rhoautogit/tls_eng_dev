"""Layer 1 – Result merger.

Combines outputs from the pdfplumber and OpenCV extractors into a single
best-effort PageExtractionResult.  Weights each source according to whether
the page appears to be digital (text-selectable) or scanned.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

from ..models import (
    BoundingBox,
    ImageElement,
    PageExtractionResult,
    Table,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _char_count(result: PageExtractionResult) -> int:
    """Count selectable characters in an extraction result."""
    return sum(len(b.text) for b in result.text_blocks) + sum(
        len(c) for t in result.tables for r in t.data for c in r if c
    )


def _is_digital(pdfplumber_result: PageExtractionResult, threshold: int = 100) -> bool:
    """Return True when the pdfplumber result has enough text to indicate a digital page."""
    return _char_count(pdfplumber_result) >= threshold


def _has_data(table: Table) -> bool:
    """Return True if the table has at least one non-empty cell."""
    return any(cell.strip() for row in table.data for cell in row if cell)


def _dedupe_tables(
    tables: List[Table], iou_threshold: float = 0.4
) -> List[Table]:
    """Remove duplicate tables by IoU overlap.

    When two tables overlap, prefer the one with actual cell content over an
    empty one, regardless of confidence weighting.  Among tables that both have
    content (or both are empty), keep the higher-confidence one.
    """
    # Sort: tables with data first, then by descending confidence
    sorted_tables = sorted(tables, key=lambda x: (not _has_data(x), -x.confidence))
    kept: List[Table] = []
    for t in sorted_tables:
        overlap_idx = next(
            (i for i, k in enumerate(kept) if t.bbox.iou(k.bbox) >= iou_threshold),
            None,
        )
        if overlap_idx is None:
            kept.append(t)
        else:
            existing = kept[overlap_idx]
            # If incoming table has data but existing doesn't, replace it
            if _has_data(t) and not _has_data(existing):
                kept[overlap_idx] = t
            # Otherwise keep existing (it was already the better candidate)
    return kept


def _dedupe_text_blocks(
    blocks: List[TextBlock], iou_threshold: float = 0.5
) -> List[TextBlock]:
    """Remove duplicate text blocks by IoU overlap, keeping higher confidence."""
    kept: List[TextBlock] = []
    for b in sorted(blocks, key=lambda x: -x.confidence):
        overlap = any(b.bbox.iou(k.bbox) >= iou_threshold for k in kept)
        if not overlap:
            kept.append(b)
    return kept


def _remove_text_inside_tables(
    text_blocks: List[TextBlock], tables: List[Table]
) -> List[TextBlock]:
    """Filter out text blocks whose centre lies inside a detected table."""
    result: List[TextBlock] = []
    for b in text_blocks:
        cx = (b.bbox.x0 + b.bbox.x1) / 2
        cy = (b.bbox.y0 + b.bbox.y1) / 2
        if not any(t.bbox.contains_point(cx, cy) for t in tables):
            result.append(b)
    return result


# ─── Merger ───────────────────────────────────────────────────────────────────

class ResultMerger:
    """Merges pdfplumber and OpenCV extraction results into a unified output."""

    def __init__(self, config: Dict[str, Any]) -> None:
        merger_cfg = config.get("extraction", {}).get("merger", {})
        self._digital_thresh: int = int(merger_cfg.get("digital_threshold", 100))
        self._scanned_thresh: int = int(merger_cfg.get("scanned_threshold", 20))
        self._pp_w_digital: float = float(merger_cfg.get("digital_pdfplumber_weight", 0.7))
        self._cv_w_digital: float = float(merger_cfg.get("digital_opencv_weight", 0.3))
        self._pp_w_scanned: float = float(merger_cfg.get("scanned_pdfplumber_weight", 0.3))
        self._cv_w_scanned: float = float(merger_cfg.get("scanned_opencv_weight", 0.7))
        self._table_iou: float = float(merger_cfg.get("table_iou_threshold", 0.4))
        self._text_iou: float = float(merger_cfg.get("text_iou_threshold", 0.5))

    def merge(
        self,
        pdfplumber_result: PageExtractionResult,
        opencv_result: PageExtractionResult,
    ) -> PageExtractionResult:
        """Return the best combined extraction from both sources."""
        pp_chars = _char_count(pdfplumber_result)
        is_digital = pp_chars >= self._digital_thresh

        if is_digital:
            pp_w, cv_w = self._pp_w_digital, self._cv_w_digital
        else:
            pp_w, cv_w = self._pp_w_scanned, self._cv_w_scanned

        logger.debug(
            "Page %d: digital=%s pp_chars=%d pp_w=%.1f cv_w=%.1f",
            pdfplumber_result.page_num,
            is_digital,
            pp_chars,
            pp_w,
            cv_w,
        )

        # ── Tables ────────────────────────────────────────────────────────────
        # Scale confidences by source weight before deduplication
        pp_tables = [
            Table(
                data=t.data,
                bbox=t.bbox,
                page_num=t.page_num,
                confidence=t.confidence * pp_w,
                source=t.source,
                headers=t.headers,
            )
            for t in pdfplumber_result.tables
        ]
        cv_tables = [
            Table(
                data=t.data,
                bbox=t.bbox,
                page_num=t.page_num,
                confidence=t.confidence * cv_w,
                source=t.source,
            )
            for t in opencv_result.tables
        ]
        merged_tables = _dedupe_tables(pp_tables + cv_tables, self._table_iou)

        # ── Text blocks ───────────────────────────────────────────────────────
        # Use pdfplumber text for digital pages; supplement with OpenCV for scanned
        if is_digital:
            primary_text = pdfplumber_result.text_blocks
            secondary_text = [
                TextBlock(
                    text=b.text,
                    bbox=b.bbox,
                    page_num=b.page_num,
                    confidence=b.confidence * cv_w,
                    source=b.source,
                )
                for b in opencv_result.text_blocks
            ]
        else:
            primary_text = [
                TextBlock(
                    text=b.text,
                    bbox=b.bbox,
                    page_num=b.page_num,
                    confidence=b.confidence * cv_w,
                    source=b.source,
                )
                for b in opencv_result.text_blocks
            ]
            secondary_text = pdfplumber_result.text_blocks

        all_text = primary_text + secondary_text
        all_text = _remove_text_inside_tables(all_text, merged_tables)
        merged_text = _dedupe_text_blocks(all_text, self._text_iou)

        # ── Images ────────────────────────────────────────────────────────────
        # pdfplumber is more reliable for embedded image detection
        merged_images = pdfplumber_result.images

        is_scanned = not is_digital and pp_chars < self._scanned_thresh

        return PageExtractionResult(
            page_num=pdfplumber_result.page_num,
            text_blocks=merged_text,
            tables=merged_tables,
            images=merged_images,
            source="merged",
            is_scanned=is_scanned,
            page_width=pdfplumber_result.page_width,
            page_height=pdfplumber_result.page_height,
        )

    def merge_with_previous(
        self,
        previous: PageExtractionResult,
        retry: PageExtractionResult,
        target_bbox: "BoundingBox | None",
    ) -> PageExtractionResult:
        """Merge a retry extraction with the previous best result.

        If target_bbox is set, keep previous elements outside the targeted
        region and replace/add elements from the retry within it.
        """
        if target_bbox is None:
            # Full page re-run: prefer whatever source has more content
            prev_chars = _char_count(previous)
            retry_chars = _char_count(retry)
            return retry if retry_chars >= prev_chars else previous

        def outside(bbox: BoundingBox) -> bool:
            cx, cy = (bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2
            return not target_bbox.contains_point(cx, cy)

        kept_text = [b for b in previous.text_blocks if outside(b.bbox)]
        kept_tables = [t for t in previous.tables if outside(t.bbox)]
        kept_images = [i for i in previous.images if outside(i.bbox)]

        merged_tables = _dedupe_tables(
            kept_tables + retry.tables, self._table_iou
        )
        merged_text = _dedupe_text_blocks(
            kept_text + retry.text_blocks, self._text_iou
        )
        merged_text = _remove_text_inside_tables(merged_text, merged_tables)

        return PageExtractionResult(
            page_num=previous.page_num,
            text_blocks=merged_text,
            tables=merged_tables,
            images=kept_images + retry.images,
            source="merged_retry",
            is_scanned=previous.is_scanned,
            page_width=previous.page_width,
            page_height=previous.page_height,
        )
