"""OCR-based rich extraction for scanned PDF pages (PaddleOCR).

When a PDF page has no usable embedded text layer (fully scanned or garbled
font encodings), this module uses PaddleOCR to extract text with position
data. Output schema matches rich_extractor.extract_rich_page() so that
downstream code (reporting, reconstruct) works identically.

Output schema:
  - text_blocks[].lines[].spans[] with origin, size, font, color, bbox
  - drawings[] for detected table grid lines
  - images[] only for real figures (not the full-page scan)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz
import numpy as np

from .paddle_ocr_engine import PaddleOCREngine, OCRPageResult

logger = logging.getLogger(__name__)


# -- Page rendering -----------------------------------------------------------

def _render_page(
    pdf_path: str, page_num: int, dpi: int = 300
) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to BGR numpy array at the given DPI."""
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


# -- Scanned-page detection ---------------------------------------------------

def is_scanned_page(pdf_path: str, page_num: int, threshold: int = 20) -> bool:
    """Return True if the page has fewer than `threshold` embedded text words."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    words = page.get_text("words")
    doc.close()
    return len(words) < threshold


# -- Rich extraction using PaddleOCR -----------------------------------------

def extract_rich_page_ocr(
    pdf_path: str,
    page_num: int,
    dpi: int = 300,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract rich page data using PaddleOCR.

    Returns a dict matching the schema of rich_extractor.extract_rich_page():
      {
        "page_num": int,
        "width": float,
        "height": float,
        "text_blocks": [...],
        "drawings": [...],
        "images": [...]
      }
    """
    config = config or {}

    # Render page
    img_bgr, pw, ph = _render_page(pdf_path, page_num, dpi=dpi)
    img_h, img_w = img_bgr.shape[:2]

    # Scale factors: pixel -> PDF points
    scale_x = pw / img_w
    scale_y = ph / img_h

    # Run PaddleOCR (reuse shared engine to avoid re-initialization)
    engine = PaddleOCREngine.get_shared(config)
    ocr_result = engine.ocr_image(img_bgr)

    # Build rich text blocks from OCR result
    text_blocks = []
    block_lines: Dict[int, List] = {}

    # Group OCR lines into blocks by vertical proximity
    sorted_lines = sorted(ocr_result.lines, key=lambda l: l.bbox[1])

    current_block_idx = 0
    prev_bottom = -999.0
    block_gap_threshold = 15.0 * (dpi / 72.0)  # ~15pt gap

    for line in sorted_lines:
        line_top = line.bbox[1]
        if line_top - prev_bottom > block_gap_threshold:
            current_block_idx += 1

        if current_block_idx not in block_lines:
            block_lines[current_block_idx] = []
        block_lines[current_block_idx].append(line)
        prev_bottom = line.bbox[3]

    for block_idx, lines in block_lines.items():
        if not lines:
            continue

        # Block bbox = union of all line bboxes
        block_x0 = min(l.bbox[0] for l in lines)
        block_y0 = min(l.bbox[1] for l in lines)
        block_x1 = max(l.bbox[2] for l in lines)
        block_y1 = max(l.bbox[3] for l in lines)

        rich_lines = []
        for line in lines:
            span_bbox = [
                line.bbox[0] * scale_x,
                line.bbox[1] * scale_y,
                line.bbox[2] * scale_x,
                line.bbox[3] * scale_y,
            ]

            span_height = span_bbox[3] - span_bbox[1]

            rich_lines.append({
                "spans": [{
                    "text": line.text,
                    "origin": [span_bbox[0], span_bbox[3]],
                    "size": max(6.0, span_height * 0.8),
                    "font": "PaddleOCR",
                    "color": 0,
                    "bbox": span_bbox,
                    "confidence": line.confidence,
                }],
                "bbox": span_bbox,
            })

        text_blocks.append({
            "lines": rich_lines,
            "bbox": [
                block_x0 * scale_x,
                block_y0 * scale_y,
                block_x1 * scale_x,
                block_y1 * scale_y,
            ],
        })

    # Detect table grid lines using OpenCV (for drawings)
    drawings = _detect_grid_lines(img_bgr, scale_x, scale_y)

    return {
        "page_num": page_num,
        "width": pw,
        "height": ph,
        "text_blocks": text_blocks,
        "drawings": drawings,
        "images": [],
    }


def _detect_grid_lines(
    img_bgr: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """Detect horizontal and vertical grid lines for the drawings field."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    h, w = binary.shape
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, w // 20), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, h // 20)))

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    drawings = []

    # Extract line segments via contours
    for mask, orientation in [(h_lines, "horizontal"), (v_lines, "vertical")]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            drawings.append({
                "type": "line",
                "orientation": orientation,
                "rect": [
                    x * scale_x, y * scale_y,
                    (x + cw) * scale_x, (y + ch) * scale_y,
                ],
            })

    return drawings
