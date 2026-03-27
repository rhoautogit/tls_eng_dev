"""Scanned path executor (PaddleOCR).

Handles pages with no usable embedded text layer (fully scanned PDFs or
pages with garbled font encodings). Uses PaddleOCR for text detection and
recognition, OpenCV for preprocessing, and Qwen-VL for verification of
low-confidence pages.

PaddleOCR runs as a subprocess so GPU memory is released before Qwen-VL
loads through Ollama.

Flow:
  Render page image (PyMuPDF)
  -> OpenCV preprocessing (deskew, denoise, binarize, CLAHE)
  -> PaddleOCR (text detection + recognition + table structure)
  -> Confidence gate evaluates results
  -> If below threshold: retry loop with escalating preprocessing
     Retry 1: Higher DPI + CLAHE contrast enhancement
     Retry 2: Max DPI + aggressive CLAHE + adaptive binarization
     Retry 3: Full combined -- max DPI, max CLAHE, morphological cleanup
  -> If confidence still low: Qwen-VL corrects flagged pages
  -> Output
"""
from __future__ import annotations

import gc
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np
import paddle

from ..layer1.opencv_extractor import OpenCVExtractor
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
    Table,
    TextBlock,
    VerificationStatus,
)
from ..paddle_ocr_engine import (
    OCRPageResult,
    PaddleOCREngine,
    run_paddleocr_subprocess,
)

logger = logging.getLogger(__name__)


# -- Page rendering -----------------------------------------------------------

def render_page_image(
    pdf_path: str, page_num: int, dpi: int = 300
) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to BGR numpy array.

    Returns (image_bgr, page_width_pts, page_height_pts).
    """
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


# -- OpenCV preprocessing ----------------------------------------------------

def preprocess_for_ocr(gray: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    """Standard preprocessing pipeline for scanned page images."""
    scan_cfg = config.get("scanned_path", {}).get("preprocessing", {})

    if scan_cfg.get("deskew", True):
        gray = _deskew(gray)
    if scan_cfg.get("denoise", True):
        gray = cv2.fastNlMeansDenoising(gray, h=10)
    if scan_cfg.get("binarize", True):
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )[1]

    clahe_cfg = scan_cfg.get("clahe", {})
    if clahe_cfg:
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        gray = clahe.apply(gray)

    return gray


def preprocess_for_retry(
    gray: np.ndarray, strategy: Dict[str, Any]
) -> np.ndarray:
    """Escalated preprocessing for a retry attempt."""
    if strategy.get("deskew", True):
        gray = _deskew(gray)

    # CLAHE contrast enhancement
    clahe_cfg = strategy.get("clahe", {})
    if clahe_cfg:
        clip = float(clahe_cfg.get("clip_limit", 2.0))
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        gray = clahe.apply(gray)

    if strategy.get("denoise", True):
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if strategy.get("line_removal", False):
        gray = _remove_lines(gray)

    # Adaptive binarization
    if strategy.get("adaptive_binarize", False):
        block_size = int(strategy.get("adaptive_block_size", 15))
        constant = int(strategy.get("adaptive_constant", 5))
        if block_size % 2 == 0:
            block_size += 1
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, constant,
        )

    # Morphological opening
    if strategy.get("morphological_open", False):
        kernel_size = tuple(strategy.get("morphological_kernel", [2, 2]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

    return gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct skew using Hough lines."""
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100,
        minLineLength=100, maxLineGap=10,
    )
    if lines is None or len(lines) == 0:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.1:
        return gray

    h, w = gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _remove_lines(gray: np.ndarray) -> np.ndarray:
    """Remove horizontal and vertical lines (table grids, form lines)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 20)))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    lines_mask = cv2.bitwise_or(h_lines, v_lines)

    # Inpaint the lines out
    result = cv2.inpaint(gray, lines_mask, 3, cv2.INPAINT_TELEA)
    return result


# -- Coordinate conversion ----------------------------------------------------

def _pixel_to_points(
    bbox_px: List[float], img_width: int, img_height: int,
    page_width_pt: float, page_height_pt: float,
) -> BoundingBox:
    """Convert pixel coordinates to PDF point coordinates."""
    scale_x = page_width_pt / img_width
    scale_y = page_height_pt / img_height
    return BoundingBox(
        x0=bbox_px[0] * scale_x,
        y0=bbox_px[1] * scale_y,
        x1=bbox_px[2] * scale_x,
        y1=bbox_px[3] * scale_y,
    )


# -- Confidence gate ----------------------------------------------------------

def compute_confidence_gate(
    ocr_result: OCRPageResult, config: Dict[str, Any]
) -> ConfidenceGate:
    """Evaluate OCR confidence and determine gating level."""
    gate_cfg = config.get("scanned_path", {}).get("confidence_gate", {})
    high_threshold = float(gate_cfg.get("high_threshold", 0.95))
    medium_threshold = float(gate_cfg.get("medium_threshold", 0.85))

    conf = ocr_result.page_confidence

    if conf >= high_threshold:
        level = ConfidenceLevel.HIGH
        needs_qwen = False
        needs_human = False
    elif conf >= medium_threshold:
        level = ConfidenceLevel.MEDIUM
        needs_qwen = True
        needs_human = False
    else:
        level = ConfidenceLevel.LOW
        needs_qwen = True
        needs_human = True

    return ConfidenceGate(
        ocr_confidence=conf,
        level=level,
        needs_qwen_vl=needs_qwen,
        needs_human_review=needs_human,
        word_count=ocr_result.word_count,
        high_confidence_words=ocr_result.high_confidence_words,
        flagged_words=ocr_result.flagged_words,
    )


# -- Build extraction result from PaddleOCR output ---------------------------

def build_extraction_from_paddle(
    ocr_result: OCRPageResult,
    page_num: int,
    page_width: float,
    page_height: float,
    img_width: int,
    img_height: int,
) -> PageExtractionResult:
    """Convert PaddleOCR result into a PageExtractionResult."""
    text_blocks: List[TextBlock] = []

    for line in ocr_result.lines:
        bbox = _pixel_to_points(
            line.bbox, img_width, img_height, page_width, page_height
        )
        text_blocks.append(TextBlock(
            text=line.text,
            bbox=bbox,
            page_num=page_num,
            confidence=line.confidence,
            source="paddleocr",
        ))

    # Convert PaddleOCR tables
    tables: List[Table] = []
    for ocr_table in ocr_result.tables:
        grid = ocr_table.to_grid()
        if not grid:
            continue

        table_bbox = _pixel_to_points(
            ocr_table.bbox, img_width, img_height, page_width, page_height
        )

        tables.append(Table(
            data=grid,
            bbox=table_bbox,
            num_rows=ocr_table.num_rows,
            num_cols=ocr_table.num_cols,
            headers=grid[0] if grid else [],
            page_num=page_num,
            confidence=0.9,
            source="paddleocr",
        ))

    return PageExtractionResult(
        page_num=page_num,
        text_blocks=text_blocks,
        tables=tables,
        images=[],
        source="paddleocr",
        is_scanned=True,
        page_width=page_width,
        page_height=page_height,
        page_type=PageType.SCANNED,
    )


# -- Scanned path executor ----------------------------------------------------

class ScannedPathExecutor:
    """Executes the scanned extraction path using PaddleOCR."""

    def __init__(
        self,
        config: Dict[str, Any],
        opencv_extractor: OpenCVExtractor,
        scorer: CoverageScorer,
        gap_analyzer=None,
    ) -> None:
        self.config = config
        self._cv = opencv_extractor
        self._scorer = scorer
        self._gap_analyzer = gap_analyzer

        self._threshold: float = float(config.get("accuracy_threshold", 0.95))

        retry_cfg = config.get("scanned_path", {}).get("retry", {})
        self._max_retries: int = int(retry_cfg.get("max_retries", 3))
        self._early_stop_delta: float = float(
            retry_cfg.get("early_termination_threshold", 0.01)
        )

        # PaddleOCR engine (shared singleton to avoid re-initialization)
        self._engine = PaddleOCREngine.get_shared(config)

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
        2. OpenCV preprocessing
        3. PaddleOCR text detection + recognition
        4. PaddleOCR table structure recognition
        5. OpenCV table detection (merge with PaddleOCR tables)
        6. Confidence gate
        7. Retry loop if below threshold
        """
        default_dpi = int(
            self.config.get("scanned_path", {}).get("ocr", {}).get("dpi", 300)
        )

        # -- Step 1: Render --
        img_bgr, pw, ph = render_page_image(pdf_path, page_num, dpi=default_dpi)
        img_h, img_w = img_bgr.shape[:2]

        # -- Step 2: Preprocess --
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        preprocessed = preprocess_for_ocr(gray, self.config)

        # -- Step 3: PaddleOCR (subprocess for GPU memory isolation) --
        ocr_result = self._engine.ocr_image(preprocessed)

        # -- Step 5: Build extraction result --
        extraction = build_extraction_from_paddle(
            ocr_result, page_num, pw, ph, img_w, img_h
        )

        # OpenCV table detection (merge non-overlapping)
        try:
            cv_result = self._cv.extract_page(pdf_path, page_num, params)
            extraction = self._merge_opencv_tables(extraction, cv_result)
        except Exception as e:
            logger.warning("OpenCV failed on scanned p%d: %s", page_num + 1, e)

        extraction.page_type = PageType.SCANNED
        extraction.is_scanned = True

        # -- Step 5b: OCR rich extraction for PDF reconstruction --
        try:
            from ..ocr_rich_extractor import extract_rich_page_ocr
            from ..rich_extractor import save_rich_page
            rich_data = extract_rich_page_ocr(pdf_path, page_num)
            save_rich_page(rich_data, output_base / "rich", page_num)
        except Exception as e:
            logger.warning("OCR rich extraction failed p%d: %s", page_num + 1, e)

        # -- Step 6: Confidence gate --
        confidence_gate = compute_confidence_gate(ocr_result, self.config)
        extraction.confidence_gate = confidence_gate

        score = self._scorer.score_page(pdf_path, page_num, extraction)
        initial_score = score

        logger.info(
            "Page %d [SCANNED] initial: PaddleOCR confidence=%.1f%% (%s) | "
            "words=%d high=%d flagged=%d | coverage=%.1f%%",
            page_num + 1,
            confidence_gate.ocr_confidence * 100,
            confidence_gate.level.value,
            confidence_gate.word_count,
            confidence_gate.high_confidence_words,
            confidence_gate.flagged_words,
            score * 100,
        )

        # -- Step 7: Retry loop --
        run_records: List[RunRecord] = []

        if confidence_gate.ocr_confidence < self._threshold:
            best_extraction, best_gate, best_score, run_records = (
                self._retry_loop(
                    pdf_path, page_num, pw, ph,
                    extraction, confidence_gate, score,
                    params, output_base,
                )
            )
            extraction = best_extraction
            confidence_gate = best_gate
            score = best_score
            extraction.confidence_gate = confidence_gate

        # Set verification status (Qwen-VL runs at Stage 3b in the pipeline)
        if confidence_gate.level == ConfidenceLevel.HIGH:
            extraction.verification_status = VerificationStatus.AUTO_VERIFIED
        else:
            extraction.verification_status = VerificationStatus.NOT_VERIFIED

        contributions = {
            "paddleocr": confidence_gate.ocr_confidence,
            "opencv": 0.0,
            "total": round(score, 4),
        }

        passed = confidence_gate.ocr_confidence >= self._threshold
        if passed:
            status = "passed_initial" if not run_records else "resolved"
        else:
            status = "unresolved"

        logger.info(
            "Page %d [SCANNED] final: PaddleOCR confidence=%.1f%% | "
            "coverage=%.1f%% [%s]%s",
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

    # -- Retry loop -----------------------------------------------------------

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
        """Retry OCR with progressively more aggressive preprocessing."""
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

            # Free previous image buffers before re-rendering
            img_bgr = gray = preprocessed = None
            gc.collect()
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()

            # Re-render at higher DPI
            try:
                img_bgr, _, _ = render_page_image(
                    pdf_path, page_num, dpi=retry_dpi
                )
                img_h, img_w = img_bgr.shape[:2]
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            except Exception as e:
                logger.warning(
                    "Retry %d render failed p%d: %s", retry_num, page_num + 1, e
                )
                continue

            # Escalated preprocessing
            try:
                preprocessed = preprocess_for_retry(gray, strategy)
            except Exception as e:
                logger.warning(
                    "Retry %d preprocessing failed p%d: %s",
                    retry_num, page_num + 1, e,
                )
                continue

            # Re-run PaddleOCR
            try:
                retry_ocr = self._engine.ocr_image(preprocessed)
                retry_gate = compute_confidence_gate(retry_ocr, self.config)
            except Exception as e:
                logger.warning(
                    "Retry %d OCR failed p%d: %s", retry_num, page_num + 1, e
                )
                continue

            # Build extraction if confidence improved
            retry_extraction = best_extraction
            retry_score = best_score
            if retry_gate.ocr_confidence > best_confidence:
                try:
                    # Also run table detection on retry
                    retry_tables = self._engine.detect_table_structure(img_bgr)
                    retry_ocr.tables = retry_tables

                    retry_extraction = build_extraction_from_paddle(
                        retry_ocr, page_num, pw, ph, img_w, img_h
                    )

                    # Merge OpenCV tables
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
                        "Retry %d extraction build failed p%d: %s",
                        retry_num, page_num + 1, e,
                    )

            delta = retry_gate.ocr_confidence - best_confidence

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

            if retry_gate.ocr_confidence > best_confidence:
                best_confidence = retry_gate.ocr_confidence
                best_gate = retry_gate
                best_extraction = retry_extraction
                best_score = retry_score

            if best_confidence >= self._threshold:
                logger.info(
                    "Page %d [SCANNED] passed at retry %d (confidence=%.1f%%)",
                    page_num + 1, retry_num, best_confidence * 100,
                )
                break

            recent_deltas.append(abs(delta))
            if len(recent_deltas) >= 2:
                if all(d < self._early_stop_delta for d in recent_deltas[-2:]):
                    logger.info(
                        "Page %d [SCANNED] early stop at retry %d (no improvement)",
                        page_num + 1, retry_num,
                    )
                    break

        return best_extraction, best_gate, best_score, run_records

    # -- Helpers --------------------------------------------------------------

    def _merge_opencv_tables(
        self,
        extraction: PageExtractionResult,
        cv_result: PageExtractionResult,
    ) -> PageExtractionResult:
        """Merge OpenCV-detected tables into the extraction without duplicates."""
        if not cv_result.tables:
            return extraction

        for cv_table in cv_result.tables:
            overlap = any(
                t.bbox.iou(cv_table.bbox) > 0.3 for t in extraction.tables
            )
            if not overlap:
                cv_table.source = "opencv"
                extraction.tables.append(cv_table)

        return extraction

    def _run_qwen_vl_verification(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
    ) -> PageExtractionResult:
        """Run Qwen-VL verification on a flagged page.

        If Qwen-VL is available and returns corrections, applies them
        to the extraction and updates verification status.
        """
        try:
            from ..qwen_vl_verifier import (
                QwenVLVerifier,
                apply_corrections,
            )

            verifier = QwenVLVerifier(self.config)

            if not verifier.is_available():
                logger.warning(
                    "Page %d: Qwen-VL not available, skipping verification",
                    page_num + 1,
                )
                return extraction

            # Gather OCR text
            ocr_text = extraction.all_text()

            # Format table data if present
            table_text = ""
            if extraction.tables:
                table_parts = []
                for ti, table in enumerate(extraction.tables, 1):
                    table_parts.append(f"Table {ti}:")
                    if table.headers:
                        table_parts.append(" | ".join(table.headers))
                        table_parts.append("-" * 40)
                    for row in table.data:
                        table_parts.append(
                            " | ".join(str(c) for c in row)
                        )
                    table_parts.append("")
                table_text = "\n".join(table_parts)

            response = verifier.verify_page(
                pdf_path, page_num, ocr_text, table_text
            )

            if response.has_corrections:
                extraction, was_modified = apply_corrections(
                    extraction, response
                )
                if was_modified:
                    extraction.verification_status = (
                        VerificationStatus.QWEN_CORRECTED
                    )
                    logger.info(
                        "Page %d: Qwen-VL applied %d corrections, "
                        "%d missing text items",
                        page_num + 1,
                        len(response.corrections),
                        len(response.missing_text),
                    )
            else:
                if response.is_accurate:
                    extraction.verification_status = (
                        VerificationStatus.QWEN_VERIFIED
                    )
                    logger.info(
                        "Page %d: Qwen-VL confirmed OCR is accurate "
                        "(confidence=%.1f%%)",
                        page_num + 1,
                        response.confidence * 100,
                    )

        except Exception as e:
            logger.warning(
                "Page %d: Qwen-VL verification failed: %s",
                page_num + 1, e,
            )

        return extraction
