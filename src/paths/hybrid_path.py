"""Hybrid path executor.

Handles pages that contain both digital text regions and scanned/image
regions on the same page. Splits the page into regions, routes each
to the appropriate extraction tools, then merges into a unified result.

Flow:
  Region Splitter -> identifies digital vs scanned zones
  -> Digital regions: pdfplumber + PyMuPDF
  -> Scanned regions: Tesseract OCR + OpenCV
  -> If scanned OCR confidence < threshold: retry loop with escalating
     preprocessing (same strategies as scanned path)
  -> Unified Page Merger
  -> Dual scoring (coverage on digital, confidence on scanned)
  -> Qwen-VL if OCR regions flagged
  -> Output
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..layer1.custom_table_logic import CustomTableLogic
from ..layer1.opencv_extractor import OpenCVExtractor
from ..layer1.pdfplumber_extractor import PDFPlumberExtractor
from ..layer2.coverage_scorer import CoverageScorer
from ..models import (
    BoundingBox,
    ConfidenceGate,
    ConfidenceLevel,
    ExtractionParameters,
    PageClassification,
    PageExtractionResult,
    PageResult,
    PageType,
    RunRecord,
    TextBlock,
    VerificationStatus,
)
from ..rich_extractor import extract_rich_page, save_rich_page
from ..ocr_rich_extractor import extract_rich_page_ocr
from ..qwen_vl_verifier import QwenVLVerifier, apply_corrections
from .region_splitter import RegionSplitter

logger = logging.getLogger(__name__)


def _extract_rich_text(rich_data: Dict[str, Any]) -> str:
    """Extract plain text from rich visual JSON data."""
    parts: List[str] = []
    for block in rich_data.get("text_blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    parts.append(text)
    return " ".join(parts)


class HybridPathExecutor:
    """Executes the hybrid extraction path for a single page."""

    def __init__(
        self,
        config: Dict[str, Any],
        pdfplumber_extractor: PDFPlumberExtractor,
        opencv_extractor: OpenCVExtractor,
        custom_table_logic: CustomTableLogic,
        scorer: CoverageScorer,
        qwen_verifier: Optional[QwenVLVerifier] = None,
    ) -> None:
        self.config = config
        self._pp = pdfplumber_extractor
        self._cv = opencv_extractor
        self._custom = custom_table_logic
        self._scorer = scorer
        self._qwen = qwen_verifier
        self._splitter = RegionSplitter(config)

        self._threshold: float = float(config.get("accuracy_threshold", 0.95))

        retry_cfg = config.get("scanned_path", {}).get("retry", {})
        self._max_retries: int = int(retry_cfg.get("max_retries", 3))
        self._early_stop_delta: float = float(
            retry_cfg.get("early_termination_threshold", 0.01)
        )

    def execute(
        self,
        pdf_path: str,
        page_num: int,
        classification: PageClassification,
        params: ExtractionParameters,
        output_base: Path,
    ) -> PageResult:
        """Run the hybrid extraction path.

        1. Split page into digital and scanned regions
        2. Extract digital regions with pdfplumber + PyMuPDF
        3. Extract scanned regions with OCR + OpenCV
        4. If scanned OCR confidence < threshold: retry loop
        5. Merge into unified page result
        6. Dual scoring
        """
        # -- Step 1: Region splitting --
        regions = self._splitter.split_page(pdf_path, page_num)
        has_digital = any(r.region_type == PageType.DIGITAL for r in regions)
        has_scanned = any(r.region_type == PageType.SCANNED for r in regions)

        logger.info(
            "Page %d [HYBRID] regions: %d digital, %d scanned",
            page_num + 1,
            sum(1 for r in regions if r.region_type == PageType.DIGITAL),
            sum(1 for r in regions if r.region_type == PageType.SCANNED),
        )

        # -- Step 2: Digital extraction --
        digital_extraction = None
        rich_text = ""
        if has_digital:
            try:
                digital_extraction = self._pp.extract_page(
                    pdf_path, page_num, params
                )
                digital_extraction = self._custom.process(digital_extraction)
            except Exception as e:
                logger.warning("pdfplumber failed on hybrid p%d: %s", page_num + 1, e)

            try:
                rich_data = extract_rich_page(pdf_path, page_num)
                save_rich_page(rich_data, output_base / "rich", page_num)
                rich_text = _extract_rich_text(rich_data)
            except Exception as e:
                logger.warning("Rich extraction failed on hybrid p%d: %s", page_num + 1, e)

        # -- Step 3: Scanned extraction --
        scanned_extraction = None
        ocr_confidence_gate = None
        run_records: List[RunRecord] = []

        if has_scanned:
            try:
                ocr_rich_data = extract_rich_page_ocr(pdf_path, page_num)
                scanned_extraction = self._build_ocr_extraction(
                    ocr_rich_data, page_num,
                    classification.page_width, classification.page_height,
                )
            except Exception as e:
                logger.warning("OCR extraction failed on hybrid p%d: %s", page_num + 1, e)

            try:
                cv_result = self._cv.extract_page(pdf_path, page_num, params)
                if scanned_extraction:
                    scanned_extraction = self._merge_opencv_tables(
                        scanned_extraction, cv_result
                    )
            except Exception as e:
                logger.warning("OpenCV failed on hybrid p%d: %s", page_num + 1, e)

            # Confidence scoring on OCR regions
            if scanned_extraction:
                from .scanned_path import (
                    _compute_page_confidence,
                    _preprocess_for_ocr,
                    _preprocess_for_retry,
                    _render_page_image,
                )
                import cv2

                img_bgr, _, _ = _render_page_image(pdf_path, page_num)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                preprocessed = _preprocess_for_ocr(gray, self.config)
                ocr_confidence_gate = _compute_page_confidence(
                    pdf_path, page_num, preprocessed, self.config,
                )

                logger.info(
                    "Page %d [HYBRID] initial OCR confidence: %.1f%% (%s)",
                    page_num + 1,
                    ocr_confidence_gate.ocr_confidence * 100,
                    ocr_confidence_gate.level.value,
                )

                # -- Step 4: Retry loop for scanned regions --
                if ocr_confidence_gate.ocr_confidence < self._threshold:
                    scanned_extraction, ocr_confidence_gate, run_records = \
                        self._retry_scanned_regions(
                            pdf_path, page_num,
                            classification.page_width,
                            classification.page_height,
                            scanned_extraction,
                            ocr_confidence_gate,
                            params,
                        )

        # -- Step 5: Inline Qwen-VL verification on scanned regions (GPU) --
        if (ocr_confidence_gate and ocr_confidence_gate.needs_qwen_vl
                and self._qwen and self._qwen.is_available()
                and scanned_extraction):
            logger.info(
                "Page %d [HYBRID] running inline Qwen-VL on scanned regions "
                "(confidence=%.1f%%)",
                page_num + 1, ocr_confidence_gate.ocr_confidence * 100,
            )
            ocr_text = scanned_extraction.all_text()
            table_text = ""
            if scanned_extraction.tables:
                table_parts = []
                for ti, table in enumerate(scanned_extraction.tables, 1):
                    table_parts.append(f"Table {ti}:")
                    if table.headers:
                        table_parts.append(" | ".join(table.headers))
                        table_parts.append("-" * 40)
                    for row in table.data:
                        table_parts.append(" | ".join(str(c) for c in row))
                    table_parts.append("")
                table_text = "\n".join(table_parts)

            qwen_response = self._qwen.verify_page(
                pdf_path, page_num, ocr_text, table_text
            )

            if qwen_response.has_corrections:
                scanned_extraction, was_modified = apply_corrections(
                    scanned_extraction, qwen_response
                )
                if was_modified:
                    qwen_status = VerificationStatus.QWEN_CORRECTED
                    logger.info(
                        "Page %d [HYBRID] Qwen-VL applied %d corrections, "
                        "%d missing text items",
                        page_num + 1,
                        len(qwen_response.corrections),
                        len(qwen_response.missing_text),
                    )
                else:
                    qwen_status = VerificationStatus.QWEN_VERIFIED
            elif qwen_response.is_accurate:
                qwen_status = VerificationStatus.QWEN_VERIFIED
                logger.info(
                    "Page %d [HYBRID] Qwen-VL confirmed OCR accurate",
                    page_num + 1,
                )
            else:
                qwen_status = VerificationStatus.PENDING_HUMAN
                logger.info(
                    "Page %d [HYBRID] Qwen-VL inconclusive, marking for human review",
                    page_num + 1,
                )

        # -- Step 6: Merge results --
        merged = self._merge_results(
            digital_extraction, scanned_extraction, page_num,
            classification.page_width, classification.page_height,
        )
        merged.page_type = PageType.HYBRID
        merged.is_scanned = False  # hybrid, not fully scanned

        if ocr_confidence_gate:
            merged.confidence_gate = ocr_confidence_gate

        # Set verification status
        if ocr_confidence_gate and ocr_confidence_gate.needs_qwen_vl:
            if self._qwen and self._qwen.is_available():
                merged.verification_status = qwen_status
            else:
                merged.verification_status = VerificationStatus.NOT_VERIFIED
        else:
            merged.verification_status = VerificationStatus.AUTO_VERIFIED

        # -- Step 7: Scoring --
        score = self._scorer.score_page(pdf_path, page_num, merged)
        initial_score = score

        contributions = {
            "pdfplumber": 0.0,
            "tesseract_ocr": 0.0,
            "opencv": 0.0,
            "total": round(score, 4),
        }

        if has_digital:
            detailed = self._scorer.score_page_detailed(
                pdf_path, page_num, merged, rich_text
            )
            contributions["pdfplumber"] = detailed.get("pdfplumber_pct", 0.0)
            contributions["rich_extractor"] = detailed.get("rich_pct", 0.0)
            score = detailed["total_score"]
            contributions["total"] = round(score, 4)

        if ocr_confidence_gate:
            contributions["tesseract_ocr"] = ocr_confidence_gate.ocr_confidence

        logger.info(
            "Page %d [HYBRID] score: %.1f%%%s",
            page_num + 1, score * 100,
            f" (after {len(run_records)} OCR retries)" if run_records else "",
        )

        threshold = float(self.config.get("accuracy_threshold", 0.95))
        passed = score >= threshold

        if run_records:
            status = "resolved" if passed else "unresolved"
        else:
            status = "passed_initial" if passed else "unresolved"

        return PageResult(
            page_num=page_num,
            final_score=score,
            initial_score=initial_score,
            passed=passed,
            extraction=merged,
            run_records=run_records,
            gap_map_paths=[],
            status=status,
            source_contributions=contributions,
            classification=classification,
        )

    # ── Retry loop for scanned regions ─────────────────────────────────────────

    def _retry_scanned_regions(
        self,
        pdf_path: str,
        page_num: int,
        pw: float,
        ph: float,
        initial_extraction: PageExtractionResult,
        initial_gate: ConfidenceGate,
        params: ExtractionParameters,
    ) -> Tuple[PageExtractionResult, ConfidenceGate, List[RunRecord]]:
        """Retry OCR on scanned regions with escalating preprocessing.

        Uses the same retry strategies as the scanned path (higher DPI,
        CLAHE, adaptive binarization, morphological cleanup).
        """
        from .scanned_path import (
            _compute_page_confidence,
            _preprocess_for_retry,
            _render_page_image,
        )
        import cv2

        best_extraction = initial_extraction
        best_gate = initial_gate
        best_confidence = initial_gate.ocr_confidence
        run_records: List[RunRecord] = []
        recent_deltas: List[float] = []

        strategies = self.config.get("scanned_path", {}).get(
            "retry", {}
        ).get("strategies", {})

        for retry_num in range(1, self._max_retries + 1):
            strategy_key = f"retry_{retry_num}"
            strategy = strategies.get(strategy_key, {})
            if not strategy:
                continue

            retry_dpi = int(strategy.get("dpi", 300 + retry_num * 100))

            logger.info(
                "Page %d [HYBRID] OCR retry %d/%d (confidence=%.1f%%, dpi=%d)",
                page_num + 1, retry_num, self._max_retries,
                best_confidence * 100, retry_dpi,
            )

            # Re-render at higher DPI
            try:
                img_bgr, _, _ = _render_page_image(
                    pdf_path, page_num, dpi=retry_dpi
                )
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(
                    "Hybrid retry %d render failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Escalated preprocessing
            try:
                preprocessed = _preprocess_for_retry(gray, strategy)
            except Exception as e:
                logger.warning(
                    "Hybrid retry %d preprocessing failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Re-run confidence scoring
            try:
                retry_gate = _compute_page_confidence(
                    pdf_path, page_num, preprocessed, self.config
                )
            except Exception as e:
                logger.warning(
                    "Hybrid retry %d confidence scoring failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Re-run OCR extraction if confidence improved
            retry_extraction = best_extraction
            if retry_gate.ocr_confidence > best_confidence:
                try:
                    rich_data = extract_rich_page_ocr(
                        pdf_path, page_num, dpi=retry_dpi
                    )
                    retry_extraction = self._build_ocr_extraction(
                        rich_data, page_num, pw, ph
                    )
                    # Re-run OpenCV tables
                    try:
                        cv_result = self._cv.extract_page(
                            pdf_path, page_num, params
                        )
                        retry_extraction = self._merge_opencv_tables(
                            retry_extraction, cv_result
                        )
                    except Exception:
                        pass

                    retry_extraction.page_type = PageType.SCANNED
                    retry_extraction.is_scanned = True
                    retry_extraction.confidence_gate = retry_gate
                except Exception as e:
                    logger.warning(
                        "Hybrid retry %d re-extraction failed p%d: %s",
                        retry_num, page_num + 1, e,
                    )

            delta = retry_gate.ocr_confidence - best_confidence

            run_records.append(RunRecord(
                run_number=retry_num,
                parameters={
                    "dpi": retry_dpi,
                    "strategy": strategy_key,
                    "description": strategy.get("description", ""),
                    "path": "hybrid_scanned_retry",
                },
                score_before=best_confidence,
                score_after=retry_gate.ocr_confidence,
                delta=delta,
                gap_map_path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                gaps=[],
            ))

            logger.info(
                "Page %d [HYBRID] OCR retry %d: confidence %.1f%% -> %.1f%% "
                "(delta=%.2f%%)",
                page_num + 1, retry_num,
                best_confidence * 100,
                retry_gate.ocr_confidence * 100,
                delta * 100,
            )

            if retry_gate.ocr_confidence > best_confidence:
                best_confidence = retry_gate.ocr_confidence
                best_gate = retry_gate
                best_extraction = retry_extraction

            # Check threshold
            if best_confidence >= self._threshold:
                logger.info(
                    "Page %d [HYBRID] OCR passed at retry %d "
                    "(confidence=%.1f%%)",
                    page_num + 1, retry_num, best_confidence * 100,
                )
                break

            # Early termination on plateau
            recent_deltas.append(abs(delta))
            if len(recent_deltas) >= 2:
                if all(d < self._early_stop_delta for d in recent_deltas[-2:]):
                    logger.info(
                        "Page %d [HYBRID] OCR early stop at retry %d",
                        page_num + 1, retry_num,
                    )
                    break

        return best_extraction, best_gate, run_records

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_ocr_extraction(
        self,
        rich_data: Dict[str, Any],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> PageExtractionResult:
        """Build extraction result from OCR rich data."""
        text_blocks: List[TextBlock] = []

        for block in rich_data.get("text_blocks", []):
            for line in block.get("lines", []):
                line_text_parts = []
                line_bbox = None
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    if text.strip():
                        line_text_parts.append(text)
                        bbox = span.get("bbox")
                        if bbox and not line_bbox:
                            line_bbox = BoundingBox(
                                bbox[0], bbox[1], bbox[2], bbox[3]
                            )
                if line_text_parts and line_bbox:
                    text_blocks.append(TextBlock(
                        text=" ".join(line_text_parts),
                        bbox=line_bbox,
                        page_num=page_num,
                        confidence=0.9,
                        source="tesseract_ocr",
                    ))

        return PageExtractionResult(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=[],
            images=[],
            source="tesseract_ocr",
            is_scanned=True,
            page_width=page_width,
            page_height=page_height,
            page_type=PageType.SCANNED,
        )

    def _merge_opencv_tables(
        self,
        extraction: PageExtractionResult,
        cv_result: PageExtractionResult,
    ) -> PageExtractionResult:
        """Merge OpenCV tables into extraction without duplicates."""
        for cv_table in cv_result.tables:
            overlap = any(
                t.bbox.iou(cv_table.bbox) > 0.3 for t in extraction.tables
            )
            if not overlap:
                cv_table.source = "opencv"
                extraction.tables.append(cv_table)
        return extraction

    def _merge_results(
        self,
        digital: Optional[PageExtractionResult],
        scanned: Optional[PageExtractionResult],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> PageExtractionResult:
        """Merge digital and scanned extraction results into one."""
        all_text_blocks: List[TextBlock] = []
        all_tables = []
        all_images = []

        if digital:
            all_text_blocks.extend(digital.text_blocks)
            all_tables.extend(digital.tables)
            all_images.extend(digital.images)

        if scanned:
            # Add scanned text blocks that don't overlap with digital ones
            for s_block in scanned.text_blocks:
                overlap = any(
                    s_block.bbox.iou(d_block.bbox) > 0.5
                    for d_block in all_text_blocks
                )
                if not overlap:
                    all_text_blocks.append(s_block)

            # Add scanned tables that don't overlap
            for s_table in scanned.tables:
                overlap = any(
                    s_table.bbox.iou(t.bbox) > 0.3 for t in all_tables
                )
                if not overlap:
                    all_tables.append(s_table)

            all_images.extend(scanned.images)

        return PageExtractionResult(
            page_num=page_num,
            text_blocks=all_text_blocks,
            tables=all_tables,
            images=all_images,
            source="hybrid",
            page_width=page_width,
            page_height=page_height,
            page_type=PageType.HYBRID,
        )
