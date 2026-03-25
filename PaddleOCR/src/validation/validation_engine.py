"""5-Layer Validation Engine.

All extraction paths (digital, scanned, hybrid) converge here.
Produces quantitative validation metrics for the report/dashboard.

Layers:
  V1 - Coverage:          spatial area covered vs total content area
  V2 - Accuracy:          text correctness (char match or OCR confidence)
  V3 - Completeness:      missing content inventory with locations
  V4 - Structural:        document structure integrity
  V5 - Cross-Validation:  composite weighted score (headline number)
"""
from __future__ import annotations

import collections
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import fitz

from ..layer2.coverage_scorer import CoverageScorer, extract_calibrated_baseline
from ..models import (
    BoundingBox,
    ConfidenceLevel,
    Gap,
    PageExtractionResult,
    PageResult,
    PageType,
    ValidationResult,
)
from ..page_classifier import is_garbled_text

logger = logging.getLogger(__name__)


class ValidationEngine:
    """Runs the 5-layer validation on a page extraction result."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self._scorer = CoverageScorer(config)

        # Validation weights (configurable)
        val_cfg = config.get("validation", {})
        weights = val_cfg.get("weights", {})
        self._w_coverage = float(weights.get("coverage", 0.30))
        self._w_accuracy = float(weights.get("accuracy", 0.25))
        self._w_completeness = float(weights.get("completeness", 0.15))
        self._w_structural = float(weights.get("structural", 0.15))
        self._w_cross_val = float(weights.get("cross_validation", 0.15))
        self._pass_threshold = float(val_cfg.get("pass_threshold", 0.95))

    def validate(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
        rich_text: str = "",
    ) -> ValidationResult:
        """Run all 5 validation layers and produce a composite score."""

        v1 = self._v1_coverage(pdf_path, page_num, extraction, rich_text)
        v2 = self._v2_accuracy(pdf_path, page_num, extraction)
        v3 = self._v3_completeness(pdf_path, page_num, extraction)
        v4 = self._v4_structural(extraction)
        v5_composite = (
            self._w_coverage * v1
            + self._w_accuracy * v2
            + self._w_completeness * v3
            + self._w_structural * v4
            + self._w_cross_val * v1  # cross-val uses coverage as proxy for now
        )

        # Normalize in case weights don't sum to 1
        total_weight = (
            self._w_coverage + self._w_accuracy + self._w_completeness
            + self._w_structural + self._w_cross_val
        )
        if total_weight > 0:
            v5_composite = v5_composite / total_weight

        passed = v5_composite >= self._pass_threshold

        result = ValidationResult(
            coverage_score=v1,
            accuracy_score=v2,
            completeness_score=v3,
            structural_score=v4,
            cross_validation_score=v5_composite,
            composite_score=v5_composite,
            passed=passed,
        )

        logger.info(
            "Page %d validation: coverage=%.1f%% accuracy=%.1f%% "
            "completeness=%.1f%% structural=%.1f%% composite=%.1f%% [%s]",
            page_num + 1,
            v1 * 100, v2 * 100, v3 * 100, v4 * 100, v5_composite * 100,
            "PASS" if passed else "FAIL",
        )

        return result

    # ── V1: Coverage ─────────────────────────────────────────────────────────

    def _v1_coverage(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
        rich_text: str,
    ) -> float:
        """Spatial coverage -- what fraction of content area was extracted.

        Reuses the existing CoverageScorer for consistency.
        For digital pages, uses the detailed scorer (pdfplumber + rich).
        For scanned pages, uses the basic scorer (OCR baseline).
        """
        if extraction.page_type == PageType.SCANNED:
            return self._scorer.score_page(pdf_path, page_num, extraction)
        else:
            detailed = self._scorer.score_page_detailed(
                pdf_path, page_num, extraction, rich_text,
            )
            return detailed["total_score"]

    # ── V2: Accuracy ─────────────────────────────────────────────────────────

    def _v2_accuracy(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
    ) -> float:
        """Text accuracy -- how correct is the extracted content.

        Digital pages: character-level fuzzy match against PyMuPDF baseline.
        Scanned pages: uses OCR confidence from the confidence gate.
        """
        if extraction.page_type == PageType.SCANNED:
            return self._accuracy_from_ocr_confidence(extraction)
        else:
            return self._accuracy_from_text_match(pdf_path, page_num, extraction)

    def _accuracy_from_ocr_confidence(
        self, extraction: PageExtractionResult
    ) -> float:
        """Use OCR confidence gate as accuracy proxy for scanned pages."""
        if extraction.confidence_gate:
            return extraction.confidence_gate.ocr_confidence
        # Fallback: average confidence across text blocks
        if extraction.text_blocks:
            return sum(b.confidence for b in extraction.text_blocks) / len(
                extraction.text_blocks
            )
        return 0.0

    def _accuracy_from_text_match(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
    ) -> float:
        """Word-frequency overlap for digital pages.

        Compares extracted text against PyMuPDF baseline using
        order-independent word multiset overlap. This handles the fact
        that pdfplumber and PyMuPDF produce text blocks in different
        orders and groupings.

        If the baseline text is garbled (broken font encodings), both
        the baseline and extracted text are unreliable, so accuracy
        cannot be measured. Returns None-equivalent score that defers
        to OCR-based validation after reclassification.
        """
        baseline = extract_calibrated_baseline(
            pdf_path, page_num, self.config
        )
        if not baseline.strip():
            return 1.0  # empty page, nothing to validate

        # If baseline is garbled, we cannot measure accuracy against it.
        # The page should have been reclassified as scanned, but if it
        # reaches here anyway, don't penalize with a false 0.0.
        if is_garbled_text(baseline):
            logger.warning(
                "Page %d: baseline text is garbled (broken font encoding), "
                "skipping text-match accuracy",
                page_num + 1,
            )
            # Check if extracted text is also garbled
            extracted = extraction.all_text()
            if is_garbled_text(extracted):
                # Both garbled -- accuracy is unmeasurable, return a
                # neutral score so it doesn't tank the composite
                return 0.5
            # Extracted text is clean (e.g. from OCR), baseline is not --
            # trust the extraction
            return 0.85

        extracted = extraction.all_text()
        if not extracted.strip():
            return 0.0

        # If extracted text is garbled but baseline is clean, that's a
        # real extraction failure
        if is_garbled_text(extracted):
            return 0.0

        # Tokenize into lowercase words (strip punctuation edges)
        def tokenize(text: str) -> List[str]:
            return [w.lower().strip(".,;:!?()[]{}\"'") for w in text.split() if w.strip()]

        baseline_words = tokenize(baseline)
        extracted_words = tokenize(extracted)

        if not baseline_words:
            return 1.0

        # Multiset (Counter) overlap: how many baseline words are present
        baseline_counts = collections.Counter(baseline_words)
        extracted_counts = collections.Counter(extracted_words)

        # Intersection: min count for each word
        matched = sum((baseline_counts & extracted_counts).values())
        total = sum(baseline_counts.values())

        return min(1.0, matched / total) if total > 0 else 1.0

    # ── V3: Completeness ─────────────────────────────────────────────────────

    def _v3_completeness(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
    ) -> float:
        """Missing content check -- are there content blocks not extracted.

        For each PyMuPDF baseline text block, checks whether its words
        appear in the extracted text. This avoids bbox granularity mismatches
        between PyMuPDF (paragraph-level) and pdfplumber (line-level).

        If baseline blocks contain garbled text (broken font encodings),
        completeness cannot be measured against them.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        page_height = page.rect.height

        # Get all content blocks from the original
        blocks = page.get_text("blocks")
        doc.close()

        # Filter to meaningful text blocks (exclude header/footer)
        cov_cfg = self.config.get("coverage", {})
        header_pct = float(cov_cfg.get("header_margin_pct", 0.08))
        footer_pct = float(cov_cfg.get("footer_margin_pct", 0.08))
        header_cutoff = page_height * header_pct
        footer_cutoff = page_height * (1.0 - footer_pct)

        original_blocks: list[str] = []
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type != 0:
                continue
            if y0 < header_cutoff or y1 > footer_cutoff:
                continue
            text = text.strip()
            if len(text) >= 3:
                original_blocks.append(text)

        if not original_blocks:
            return 1.0  # nothing to miss

        # Check if the baseline blocks are garbled -- if so, we cannot
        # meaningfully measure completeness against them.
        all_block_text = " ".join(original_blocks)
        if is_garbled_text(all_block_text):
            logger.warning(
                "Page %d: baseline blocks are garbled (broken font encoding), "
                "skipping completeness check",
                page_num + 1,
            )
            return 0.5  # neutral score, don't penalize or reward

        # Build a set of extracted words for fast lookup
        extracted_text = extraction.all_text().lower()
        extracted_words = set(extracted_text.split())

        # For each baseline block, check what fraction of its words
        # appear in the extracted text
        covered = 0
        for block_text in original_blocks:
            block_words = [w.lower() for w in block_text.split() if len(w) >= 2]
            if not block_words:
                covered += 1
                continue
            found = sum(1 for w in block_words if w in extracted_words)
            # Block is "covered" if >= 60% of its words are in extracted text
            if found / len(block_words) >= 0.6:
                covered += 1

        return covered / len(original_blocks)

    # ── V4: Structural ───────────────────────────────────────────────────────

    def _v4_structural(
        self,
        extraction: PageExtractionResult,
    ) -> float:
        """Document structure integrity checks.

        Validates:
        - Tables have consistent row/column counts
        - Text blocks have valid bounding boxes
        - No overlapping elements that suggest extraction errors
        """
        if not extraction.text_blocks and not extraction.tables:
            return 1.0  # nothing to validate

        scores: List[float] = []

        # Check 1: Table structure consistency
        for table in extraction.tables:
            if table.num_rows == 0:
                scores.append(0.5)
                continue

            # All rows should have similar column count
            col_counts = [len(row) for row in table.data]
            if col_counts:
                max_cols = max(col_counts)
                if max_cols > 0:
                    consistency = sum(
                        c / max_cols for c in col_counts
                    ) / len(col_counts)
                    scores.append(consistency)
                else:
                    scores.append(0.5)

        # Check 2: Valid bounding boxes (non-zero area, within page)
        pw = extraction.page_width or 1000
        ph = extraction.page_height or 1000
        valid_bboxes = 0
        total_elements = len(extraction.text_blocks) + len(extraction.tables)

        for tb in extraction.text_blocks:
            if (tb.bbox.area > 0
                and tb.bbox.x0 >= 0 and tb.bbox.y0 >= 0
                and tb.bbox.x1 <= pw * 1.1 and tb.bbox.y1 <= ph * 1.1):
                valid_bboxes += 1

        for table in extraction.tables:
            if (table.bbox.area > 0
                and table.bbox.x0 >= 0 and table.bbox.y0 >= 0
                and table.bbox.x1 <= pw * 1.1 and table.bbox.y1 <= ph * 1.1):
                valid_bboxes += 1

        if total_elements > 0:
            scores.append(valid_bboxes / total_elements)

        # Check 3: No excessive overlaps between text blocks
        overlap_count = 0
        total_pairs = 0
        for i, tb1 in enumerate(extraction.text_blocks):
            for tb2 in extraction.text_blocks[i + 1:]:
                total_pairs += 1
                if tb1.bbox.iou(tb2.bbox) > 0.5:
                    overlap_count += 1

        if total_pairs > 0:
            no_overlap_ratio = 1.0 - (overlap_count / total_pairs)
            scores.append(no_overlap_ratio)
        else:
            scores.append(1.0)

        return sum(scores) / len(scores) if scores else 1.0

    # ── Flagged Items ────────────────────────────────────────────────────────

    def get_flagged_items(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
        validation: ValidationResult,
    ) -> List[Dict[str, Any]]:
        """Generate a list of flagged items for the report.

        Returns specific issues with page number, coordinates, type,
        and what action is needed.
        """
        flagged: List[Dict[str, Any]] = []

        # Flag low accuracy
        if validation.accuracy_score < 0.90:
            flagged.append({
                "page": page_num + 1,
                "type": "low_accuracy",
                "severity": "high" if validation.accuracy_score < 0.85 else "medium",
                "detail": f"Accuracy score {validation.accuracy_score:.1%}",
                "action": "Review extracted text for errors",
            })

        # Flag low completeness
        if validation.completeness_score < 0.90:
            flagged.append({
                "page": page_num + 1,
                "type": "missing_content",
                "severity": "high" if validation.completeness_score < 0.80 else "medium",
                "detail": f"Completeness {validation.completeness_score:.1%} -- content regions missed",
                "action": "Check for missing text blocks or tables",
            })

        # Flag OCR confidence issues
        if extraction.confidence_gate:
            gate = extraction.confidence_gate
            if gate.needs_qwen_vl:
                flagged.append({
                    "page": page_num + 1,
                    "type": "low_ocr_confidence",
                    "severity": "high" if gate.needs_human_review else "medium",
                    "detail": (
                        f"OCR confidence {gate.ocr_confidence:.1%} "
                        f"({gate.flagged_words} words flagged)"
                    ),
                    "action": (
                        "Qwen-VL review" + (" + human review" if gate.needs_human_review else "")
                    ),
                })

        # Flag structural issues
        if validation.structural_score < 0.90:
            flagged.append({
                "page": page_num + 1,
                "type": "structural_issue",
                "severity": "medium",
                "detail": f"Structural score {validation.structural_score:.1%}",
                "action": "Check table structure and element positioning",
            })

        return flagged
