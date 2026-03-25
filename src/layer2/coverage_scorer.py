"""Layer 2 – Content Coverage Scorer.

Uses PyMuPDF to extract a calibrated text baseline per page (excluding
headers, footers, page numbers, and watermarks), then compares it against
the combined extraction output (pdfplumber + OpenCV + rich extractor) to
produce a per-page coverage score with per-tool contribution breakdown.

For scanned pages (no selectable text), uses Tesseract OCR on the rendered
page image as the baseline instead.

Score = extracted_chars / baseline_chars, capped at 1.0.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import fitz  # PyMuPDF

from ..models import PageExtractionResult
from ..page_classifier import is_garbled_text

logger = logging.getLogger(__name__)


# ─── Baseline Extraction ──────────────────────────────────────────────────────

def _compile_page_number_patterns(patterns: List[str]) -> List[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


def extract_calibrated_baseline(
    pdf_path: str,
    page_num: int,
    config: Dict[str, Any],
) -> str:
    """Return the calibrated baseline text for coverage comparison.

    Excludes:
      - Text in the top ``header_margin_pct`` of the page.
      - Text in the bottom ``footer_margin_pct`` of the page.
      - Blocks shorter than ``min_text_length`` characters (page numbers, etc.).
      - Blocks matching any ``page_number_patterns`` regex.
    """
    cov_cfg = config.get("coverage", {})
    header_pct: float = float(cov_cfg.get("header_margin_pct", 0.08))
    footer_pct: float = float(cov_cfg.get("footer_margin_pct", 0.08))
    min_len: int = int(cov_cfg.get("min_text_length", 3))
    pn_patterns = _compile_page_number_patterns(
        cov_cfg.get("page_number_patterns", [r"^\d+$"])
    )

    doc = fitz.open(pdf_path)
    page = doc[page_num]
    page_height: float = page.rect.height

    header_cutoff = page_height * header_pct
    footer_cutoff = page_height * (1.0 - footer_pct)

    # get_text("blocks") -> (x0, y0, x1, y1, text, block_no, block_type)
    # block_type 0 = text, 1 = image
    blocks = page.get_text("blocks")
    doc.close()

    kept: List[str] = []
    for block in blocks:
        x0, y0, x1, y1, text, block_no, block_type = block
        if block_type != 0:
            continue  # skip image blocks
        if y0 < header_cutoff or y1 > footer_cutoff:
            continue  # skip header / footer region
        text = text.strip()
        if len(text) < min_len:
            continue
        if any(pat.match(text) for pat in pn_patterns):
            continue
        kept.append(text)

    baseline = " ".join(kept)

    # Fallback: if no selectable text found OR text is garbled (broken font
    # encodings), use Tesseract OCR on the rendered page image as baseline.
    if _count_meaningful_chars(baseline) == 0:
        baseline = _ocr_baseline(pdf_path, page_num, header_pct, footer_pct, min_len, pn_patterns)
    elif is_garbled_text(baseline):
        logger.warning(
            "Page %d: baseline text is garbled (broken font encoding), "
            "falling back to OCR baseline",
            page_num + 1,
        )
        baseline = _ocr_baseline(pdf_path, page_num, header_pct, footer_pct, min_len, pn_patterns)

    return baseline


def _ocr_baseline(
    pdf_path: str,
    page_num: int,
    header_pct: float,
    footer_pct: float,
    min_len: int,
    pn_patterns: List[re.Pattern],
) -> str:
    """Generate baseline text from Tesseract OCR for scanned pages."""
    try:
        import pytesseract
        from PIL import Image as PILImage
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

        # Render page to image
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        dpi = 200
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        doc.close()

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        img_h = img.shape[0]
        header_cutoff_px = int(img_h * header_pct)
        footer_cutoff_px = int(img_h * (1.0 - footer_pct))

        pil_img = PILImage.fromarray(img)
        data = pytesseract.image_to_data(pil_img, config="--psm 6", output_type=pytesseract.Output.DICT)

        # Filter individual words by position (not whole blocks, since
        # Tesseract may group an entire page into one block)
        kept: List[str] = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text or int(data["conf"][i]) < 30:
                continue
            top = data["top"][i]
            bottom = top + data["height"][i]
            # Exclude header/footer regions
            if top < header_cutoff_px or bottom > footer_cutoff_px:
                continue
            if len(text) >= min_len or len(text) >= 1:
                kept.append(text)

        result = " ".join(kept)
        if _count_meaningful_chars(result) > 0:
            logger.debug("Page %d: OCR baseline = %d chars", page_num, _count_meaningful_chars(result))
        return result
    except Exception as e:
        logger.debug("OCR baseline failed p%d: %s", page_num, e)
        return ""


# ─── Coverage Calculation ─────────────────────────────────────────────────────

def _count_meaningful_chars(text: str) -> int:
    """Count non-whitespace characters."""
    return len(text.replace(" ", "").replace("\n", "").replace("\t", ""))


def calculate_coverage(
    layer1_result: PageExtractionResult,
    baseline_text: str,
) -> float:
    """Compute coverage score in [0, 1].

    If the baseline is empty (blank page / all headers+footers), returns 1.0
    to avoid false failures on whitespace-only pages.
    """
    baseline_chars = _count_meaningful_chars(baseline_text)
    if baseline_chars == 0:
        logger.debug("Page %d: empty baseline -> score=1.0", layer1_result.page_num)
        return 1.0

    extracted_text = layer1_result.all_text()
    extracted_chars = _count_meaningful_chars(extracted_text)

    score = min(1.0, extracted_chars / baseline_chars)
    logger.debug(
        "Page %d: extracted=%d baseline=%d score=%.3f",
        layer1_result.page_num,
        extracted_chars,
        baseline_chars,
        score,
    )
    return score


def calculate_coverage_detailed(
    layer1_result: PageExtractionResult,
    rich_text: str,
    baseline_text: str,
) -> Dict[str, Any]:
    """Compute per-tool coverage breakdown.

    Returns a dict with percentage contributions from pdfplumber, OpenCV,
    and the rich extractor (PyMuPDF), plus the combined total score.
    """
    baseline_chars = _count_meaningful_chars(baseline_text)
    if baseline_chars == 0:
        return {
            "total_score": 1.0, "pdfplumber_pct": 0.0,
            "opencv_pct": 0.0, "rich_pct": 0.0,
            "baseline_chars": 0, "pdfplumber_chars": 0,
            "opencv_chars": 0, "rich_chars": 0,
        }

    # Count Layer 1 chars by source
    pp_chars = 0
    cv_chars = 0
    for block in layer1_result.text_blocks:
        chars = _count_meaningful_chars(block.text)
        if "pdfplumber" in (block.source or "").lower():
            pp_chars += chars
        else:
            cv_chars += chars

    for table in layer1_result.tables:
        t_text = " ".join(cell for row in table.data for cell in row if cell)
        chars = _count_meaningful_chars(t_text)
        if "pdfplumber" in (table.source or "").lower():
            pp_chars += chars
        else:
            cv_chars += chars

    layer1_total = pp_chars + cv_chars

    # Rich extractor contribution = what it covers beyond Layer 1
    rich_total = _count_meaningful_chars(rich_text)
    rich_additional = max(0, min(baseline_chars, rich_total) - layer1_total)

    combined = layer1_total + rich_additional
    total_score = min(1.0, combined / baseline_chars)

    # Normalize per-tool percentages proportionally so they sum to total_score.
    # Raw chars can exceed baseline (extracted text includes headers/footers
    # that the baseline excludes, or overlapping content from both tools),
    # so we distribute the capped total_score by each tool's share.
    if combined > 0:
        pp_pct = round(total_score * (pp_chars / combined), 4)
        cv_pct = round(total_score * (cv_chars / combined), 4)
        ri_pct = round(total_score * (rich_additional / combined), 4)
    else:
        pp_pct = cv_pct = ri_pct = 0.0

    return {
        "total_score": total_score,
        "pdfplumber_pct": pp_pct,
        "opencv_pct": cv_pct,
        "rich_pct": ri_pct,
        "baseline_chars": baseline_chars,
        "pdfplumber_chars": pp_chars,
        "opencv_chars": cv_chars,
        "rich_chars": rich_additional,
    }


# ─── Scorer ───────────────────────────────────────────────────────────────

class CoverageScorer:
    """Orchestrates per-page coverage scoring."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def score_page(
        self,
        pdf_path: str,
        page_num: int,
        layer1_result: PageExtractionResult,
    ) -> float:
        """Return coverage score for a single page (Layer 1 only)."""
        baseline = extract_calibrated_baseline(pdf_path, page_num, self.config)
        return calculate_coverage(layer1_result, baseline)

    def score_page_detailed(
        self,
        pdf_path: str,
        page_num: int,
        layer1_result: PageExtractionResult,
        rich_text: str = "",
    ) -> Dict[str, Any]:
        """Return detailed coverage breakdown by extraction tool."""
        baseline = extract_calibrated_baseline(pdf_path, page_num, self.config)
        return calculate_coverage_detailed(layer1_result, rich_text, baseline)

    def get_baseline(self, pdf_path: str, page_num: int) -> str:
        """Return the raw calibrated baseline text (useful for debugging)."""
        return extract_calibrated_baseline(pdf_path, page_num, self.config)
