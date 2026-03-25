"""Scanned path executor.

Handles pages with no embedded text layer (fully scanned PDFs).
Uses Tesseract OCR for text extraction and OpenCV for table/form detection.
Applies confidence gating to flag pages for Qwen-VL verification.

Flow:
  Preprocessing (deskew, binarize, noise/line removal)
  -> Tesseract OCR (word-level text + confidence)
  -> OpenCV (table structure, checkboxes)
  -> Page Confidence Scoring
  -> If below threshold: retry loop with escalating preprocessing
     Retry 1: Higher DPI + CLAHE contrast enhancement
     Retry 2: Max DPI + aggressive CLAHE + adaptive binarization
     Retry 3: Full combined -- max DPI, max CLAHE, morphological cleanup
  -> Confidence Gate:
     High (>95%):  auto-verified
     Medium (85-95%): flagged for Qwen-VL
     Low (<85%):  flagged for Qwen-VL + human review
  -> Output
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np

from ..layer1.opencv_extractor import OpenCVExtractor
from ..layer2.coverage_scorer import CoverageScorer
from ..models import (
    ConfidenceGate,
    ConfidenceLevel,
    ExtractionParameters,
    PageClassification,
    PageExtractionResult,
    PageResult,
    PageType,
    RunRecord,
    TextBlock,
    BoundingBox,
    VerificationStatus,
)
from ..ocr_rich_extractor import (
    extract_rich_page_ocr,
    is_scanned_page,
)
from ..qwen_vl_verifier import QwenVLVerifier, apply_corrections

logger = logging.getLogger(__name__)


# ── Tesseract setup ──────────────────────────────────────────────────────────

def _get_tesseract():
    """Import and configure pytesseract."""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    return pytesseract


# ── Preprocessing (reuses logic from ocr_rich_extractor) ─────────────────────

def _render_page_image(
    pdf_path: str, page_num: int, dpi: int = 300
) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to BGR numpy array."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, pw, ph


def _preprocess_for_ocr(gray: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Run the preprocessing pipeline on a grayscale scan image.

    Imports and reuses the preprocessing functions from ocr_rich_extractor.
    """
    from ..ocr_rich_extractor import _deskew, _remove_lines, _remove_non_text_elements

    scan_cfg = config.get("scanned_path", {}).get("preprocessing", {})

    if scan_cfg.get("deskew", True):
        gray = _deskew(gray)
    if scan_cfg.get("line_removal", True):
        gray = _remove_lines(gray)
    if scan_cfg.get("noise_removal", True):
        gray = _remove_non_text_elements(gray)

    return gray


def _preprocess_for_retry(
    gray: np.ndarray, strategy: Dict[str, Any]
) -> np.ndarray:
    """Run an escalated preprocessing pipeline for a retry attempt.

    Each retry strategy can enable/disable preprocessing steps and
    adjust their parameters (CLAHE clip limit, adaptive threshold, etc.).
    """
    from ..ocr_rich_extractor import _deskew, _remove_lines, _remove_non_text_elements

    if strategy.get("deskew", True):
        gray = _deskew(gray)

    # CLAHE contrast enhancement
    clahe_cfg = strategy.get("clahe", {})
    if clahe_cfg:
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        gray = clahe.apply(gray)

    if strategy.get("line_removal", True):
        gray = _remove_lines(gray)
    if strategy.get("denoise", True):
        gray = _remove_non_text_elements(gray)

    # Adaptive binarization (more aggressive than default Otsu)
    if strategy.get("adaptive_binarize", False):
        block_size = int(strategy.get("adaptive_block_size", 15))
        constant = int(strategy.get("adaptive_constant", 5))
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, constant,
        )

    # Morphological opening to clean up noise
    if strategy.get("morphological_open", False):
        kernel_size = tuple(strategy.get("morphological_kernel", [2, 2]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return gray


# ── Confidence scoring ───────────────────────────────────────────────────────

def _compute_page_confidence(
    pdf_path: str,
    page_num: int,
    preprocessed_gray: np.ndarray,
    config: Dict[str, Any],
) -> ConfidenceGate:
    """Compute page-level OCR confidence from Tesseract word-level scores.

    Runs Tesseract image_to_data on the preprocessed image to get per-word
    confidence scores, then aggregates into a page-level metric.
    """
    pytesseract = _get_tesseract()
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(preprocessed_gray)
    data = pytesseract.image_to_data(
        pil_img, config="--psm 6", output_type=pytesseract.Output.DICT
    )

    # Collect word confidences (skip empty/invalid entries)
    word_confidences: List[float] = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 0:
            continue
        word_confidences.append(conf / 100.0)

    if not word_confidences:
        return ConfidenceGate(
            ocr_confidence=0.0,
            level=ConfidenceLevel.LOW,
            needs_qwen_vl=True,
            needs_human_review=True,
            word_count=0,
            high_confidence_words=0,
            flagged_words=0,
        )

    avg_confidence = sum(word_confidences) / len(word_confidences)
    high_conf_words = sum(1 for c in word_confidences if c >= 0.95)
    flagged_words = sum(1 for c in word_confidences if c < 0.85)

    # Determine confidence level from config thresholds
    gate_cfg = config.get("scanned_path", {}).get("confidence_gate", {})
    high_threshold = float(gate_cfg.get("high_threshold", 0.95))
    medium_threshold = float(gate_cfg.get("medium_threshold", 0.85))

    if avg_confidence >= high_threshold:
        level = ConfidenceLevel.HIGH
        needs_qwen = False
        needs_human = False
    elif avg_confidence >= medium_threshold:
        level = ConfidenceLevel.MEDIUM
        needs_qwen = True
        needs_human = False
    else:
        level = ConfidenceLevel.LOW
        needs_qwen = True
        needs_human = True

    return ConfidenceGate(
        ocr_confidence=avg_confidence,
        level=level,
        needs_qwen_vl=needs_qwen,
        needs_human_review=needs_human,
        word_count=len(word_confidences),
        high_confidence_words=high_conf_words,
        flagged_words=flagged_words,
    )


class ScannedPathExecutor:
    """Executes the scanned extraction path for a single page."""

    def __init__(
        self,
        config: Dict[str, Any],
        opencv_extractor: OpenCVExtractor,
        scorer: CoverageScorer,
        qwen_verifier: Optional[QwenVLVerifier] = None,
    ) -> None:
        self.config = config
        self._cv = opencv_extractor
        self._scorer = scorer
        self._qwen = qwen_verifier

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
        """Run the scanned extraction path.

        1. Render page to image
        2. Preprocess (deskew, line removal, noise removal)
        3. OCR rich extraction (Tesseract + layout analysis)
        4. OpenCV table/form detection
        5. Confidence scoring and gating
        6. If below threshold: retry loop with escalating preprocessing
        """
        # -- Step 1-2: Render and preprocess --
        default_dpi = int(
            self.config.get("scanned_path", {}).get("ocr", {}).get("dpi", 300)
        )
        img_bgr, pw, ph = _render_page_image(pdf_path, page_num, dpi=default_dpi)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        preprocessed = _preprocess_for_ocr(gray, self.config)

        # -- Step 3: OCR rich extraction --
        try:
            rich_data = extract_rich_page_ocr(pdf_path, page_num)
            from ..rich_extractor import save_rich_page
            save_rich_page(rich_data, output_base / "rich", page_num)
        except Exception as e:
            logger.warning("OCR rich extraction failed p%d: %s", page_num + 1, e)
            rich_data = {}

        # Build extraction result from OCR data
        extraction = self._build_extraction_from_ocr(
            rich_data, page_num, pw, ph
        )

        # -- Step 4: OpenCV table/form detection --
        try:
            cv_result = self._cv.extract_page(pdf_path, page_num, params)
            extraction = self._merge_opencv_tables(extraction, cv_result)
        except Exception as e:
            logger.warning("OpenCV failed on scanned p%d: %s", page_num + 1, e)

        # Tag as scanned
        extraction.page_type = PageType.SCANNED
        extraction.is_scanned = True

        # -- Step 5: Confidence scoring --
        confidence_gate = _compute_page_confidence(
            pdf_path, page_num, preprocessed, self.config
        )
        extraction.confidence_gate = confidence_gate

        # Coverage scoring
        score = self._scorer.score_page(pdf_path, page_num, extraction)
        initial_score = score

        logger.info(
            "Page %d [SCANNED] initial: OCR confidence=%.1f%% (%s) | "
            "words=%d high=%d flagged=%d | coverage=%.1f%%",
            page_num + 1,
            confidence_gate.ocr_confidence * 100,
            confidence_gate.level.value,
            confidence_gate.word_count,
            confidence_gate.high_confidence_words,
            confidence_gate.flagged_words,
            score * 100,
        )

        # -- Step 6: Retry loop if below threshold --
        run_records: List[RunRecord] = []

        if confidence_gate.ocr_confidence < self._threshold:
            best_extraction, best_gate, best_score, run_records = \
                self._retry_loop(
                    pdf_path, page_num, pw, ph,
                    extraction, confidence_gate, score,
                    params, output_base,
                )
            extraction = best_extraction
            confidence_gate = best_gate
            score = best_score
            extraction.confidence_gate = confidence_gate

        # -- Step 7: Inline Qwen-VL verification (GPU) if still below threshold --
        if confidence_gate.needs_qwen_vl and self._qwen and self._qwen.is_available():
            logger.info(
                "Page %d [SCANNED] running inline Qwen-VL verification (confidence=%.1f%%)",
                page_num + 1, confidence_gate.ocr_confidence * 100,
            )
            ocr_text = extraction.all_text()
            table_text = ""
            if extraction.tables:
                table_parts = []
                for ti, table in enumerate(extraction.tables, 1):
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
                extraction, was_modified = apply_corrections(extraction, qwen_response)
                if was_modified:
                    extraction.verification_status = VerificationStatus.QWEN_CORRECTED
                    score = self._scorer.score_page(pdf_path, page_num, extraction)
                    logger.info(
                        "Page %d [SCANNED] Qwen-VL applied %d corrections, "
                        "%d missing text items, new coverage=%.1f%%",
                        page_num + 1,
                        len(qwen_response.corrections),
                        len(qwen_response.missing_text),
                        score * 100,
                    )
                else:
                    extraction.verification_status = VerificationStatus.QWEN_VERIFIED
            elif qwen_response.is_accurate:
                extraction.verification_status = VerificationStatus.QWEN_VERIFIED
                logger.info(
                    "Page %d [SCANNED] Qwen-VL confirmed OCR accurate (confidence=%.1f%%)",
                    page_num + 1, qwen_response.confidence * 100,
                )
            else:
                extraction.verification_status = VerificationStatus.PENDING_HUMAN
                logger.info(
                    "Page %d [SCANNED] Qwen-VL inconclusive, marking for human review",
                    page_num + 1,
                )
        elif confidence_gate.level == ConfidenceLevel.HIGH:
            extraction.verification_status = VerificationStatus.AUTO_VERIFIED
        else:
            extraction.verification_status = VerificationStatus.NOT_VERIFIED

        contributions = {
            "tesseract_ocr": confidence_gate.ocr_confidence,
            "opencv": 0.0,
            "total": round(score, 4),
        }

        passed = confidence_gate.ocr_confidence >= self._threshold
        if passed:
            status = "passed_initial" if not run_records else "resolved"
        else:
            status = "unresolved"

        logger.info(
            "Page %d [SCANNED] final: OCR confidence=%.1f%% | coverage=%.1f%% [%s]%s",
            page_num + 1,
            confidence_gate.ocr_confidence * 100,
            score * 100,
            status.upper(),
            f" (after {len(run_records)} retries)" if run_records else "",
        )

        return PageResult(
            page_num=page_num,
            final_score=score,
            initial_score=initial_score,
            passed=passed,
            extraction=extraction,
            run_records=run_records,
            gap_map_paths=[],
            status=status,
            source_contributions=contributions,
            classification=classification,
        )

    # ── Retry loop ────────────────────────────────────────────────────────────

    def _retry_loop(
        self,
        pdf_path: str,
        page_num: int,
        pw: float,
        ph: float,
        initial_extraction: PageExtractionResult,
        initial_gate: ConfidenceGate,
        initial_score: float,
        params: ExtractionParameters,
        output_base: Path,
    ) -> Tuple[PageExtractionResult, ConfidenceGate, float, List[RunRecord]]:
        """Retry OCR with progressively more aggressive preprocessing.

        Each retry re-renders the page at a higher DPI and applies stronger
        preprocessing (CLAHE, adaptive binarization, morphological cleanup),
        then re-runs Tesseract to see if confidence improves.
        """
        best_extraction = initial_extraction
        best_gate = initial_gate
        best_score = initial_score
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
                "Page %d [SCANNED] retry %d/%d (confidence=%.1f%%, dpi=%d) -- %s",
                page_num + 1, retry_num, self._max_retries,
                best_confidence * 100, retry_dpi,
                strategy.get("description", ""),
            )

            # Re-render at higher DPI
            try:
                img_bgr, _, _ = _render_page_image(
                    pdf_path, page_num, dpi=retry_dpi
                )
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(
                    "Retry %d render failed p%d: %s", retry_num, page_num + 1, e
                )
                continue

            # Escalated preprocessing
            try:
                preprocessed = _preprocess_for_retry(gray, strategy)
            except Exception as e:
                logger.warning(
                    "Retry %d preprocessing failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Re-run confidence scoring on the new preprocessed image
            try:
                retry_gate = _compute_page_confidence(
                    pdf_path, page_num, preprocessed, self.config
                )
            except Exception as e:
                logger.warning(
                    "Retry %d confidence scoring failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Re-run OCR extraction if confidence improved
            retry_extraction = best_extraction
            retry_score = best_score
            if retry_gate.ocr_confidence > best_confidence:
                try:
                    rich_data = extract_rich_page_ocr(
                        pdf_path, page_num, dpi=retry_dpi
                    )
                    retry_extraction = self._build_extraction_from_ocr(
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

                    retry_score = self._scorer.score_page(
                        pdf_path, page_num, retry_extraction
                    )
                except Exception as e:
                    logger.warning(
                        "Retry %d re-extraction failed p%d: %s",
                        retry_num, page_num + 1, e,
                    )

            delta = retry_gate.ocr_confidence - best_confidence

            # Record the retry attempt
            run_records.append(RunRecord(
                run_number=retry_num,
                parameters={
                    "dpi": retry_dpi,
                    "strategy": strategy_key,
                    "description": strategy.get("description", ""),
                },
                score_before=best_confidence,
                score_after=retry_gate.ocr_confidence,
                delta=delta,
                gap_map_path="",
                timestamp=datetime.now(timezone.utc).isoformat(),
                gaps=[],
            ))

            logger.info(
                "Page %d [SCANNED] retry %d: confidence %.1f%% -> %.1f%% "
                "(delta=%.2f%%)",
                page_num + 1, retry_num,
                best_confidence * 100,
                retry_gate.ocr_confidence * 100,
                delta * 100,
            )

            # Update best if improved
            if retry_gate.ocr_confidence > best_confidence:
                best_confidence = retry_gate.ocr_confidence
                best_gate = retry_gate
                best_extraction = retry_extraction
                best_score = retry_score

            # Check if we've reached the threshold
            if best_confidence >= self._threshold:
                logger.info(
                    "Page %d [SCANNED] passed at retry %d (confidence=%.1f%%)",
                    page_num + 1, retry_num, best_confidence * 100,
                )
                break

            # Early termination if plateau
            recent_deltas.append(abs(delta))
            if len(recent_deltas) >= 2:
                if all(d < self._early_stop_delta for d in recent_deltas[-2:]):
                    logger.info(
                        "Page %d [SCANNED] early stop at retry %d "
                        "(no improvement)",
                        page_num + 1, retry_num,
                    )
                    break

        return best_extraction, best_gate, best_score, run_records

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_extraction_from_ocr(
        self,
        rich_data: Dict[str, Any],
        page_num: int,
        page_width: float,
        page_height: float,
    ) -> PageExtractionResult:
        """Build a PageExtractionResult from OCR rich data."""
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
                        confidence=0.9,  # default, refined by confidence gate
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
        """Merge OpenCV-detected tables into the OCR extraction result."""
        if not cv_result.tables:
            return extraction

        # Add OpenCV tables that don't overlap with existing ones
        for cv_table in cv_result.tables:
            overlap = False
            for existing in extraction.tables:
                if existing.bbox.iou(cv_table.bbox) > 0.3:
                    overlap = True
                    break
            if not overlap:
                cv_table.source = "opencv"
                extraction.tables.append(cv_table)

        return extraction
