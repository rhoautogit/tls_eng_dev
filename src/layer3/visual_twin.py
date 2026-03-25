"""Layer 3 – Visual Twin renderer.

Renders the Layer 1 extraction result as a colour-coded overlay on a blank
canvas so it can be compared pixel-by-pixel against the original page image.

Colour coding:
  - Text blocks  → semi-transparent green
  - Tables       → semi-transparent blue
  - Images       → semi-transparent yellow
"""
from __future__ import annotations

from typing import Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

from ..models import PageExtractionResult

# BGR colours
_TEXT_COLOUR = (100, 200, 100)    # green
_TABLE_COLOUR = (200, 150, 80)    # blue
_IMAGE_COLOUR = (80, 200, 220)    # yellow


def render_page_image(
    pdf_path: str, page_num: int, dpi: int = 150
) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to a BGR image.

    Returns:
        (image, page_width_pts, page_height_pts)
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pw, ph = page.rect.width, page.rect.height
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    doc.close()
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    elif pix.n == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img, pw, ph


def render_extraction_twin(
    result: PageExtractionResult,
    img_width: int,
    img_height: int,
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    """Paint extracted regions onto a white canvas at image resolution.

    Regions:
        Green  – text blocks
        Blue   – tables
        Yellow – images
    """
    canvas = np.full((img_height, img_width, 3), 255, dtype=np.uint8)

    for b in result.text_blocks:
        x0 = max(0, int(b.bbox.x0 * scale_x))
        y0 = max(0, int(b.bbox.y0 * scale_y))
        x1 = min(img_width - 1, int(b.bbox.x1 * scale_x))
        y1 = min(img_height - 1, int(b.bbox.y1 * scale_y))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _TEXT_COLOUR, -1)

    for t in result.tables:
        x0 = max(0, int(t.bbox.x0 * scale_x))
        y0 = max(0, int(t.bbox.y0 * scale_y))
        x1 = min(img_width - 1, int(t.bbox.x1 * scale_x))
        y1 = min(img_height - 1, int(t.bbox.y1 * scale_y))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _TABLE_COLOUR, -1)

    for img_el in result.images:
        x0 = max(0, int(img_el.bbox.x0 * scale_x))
        y0 = max(0, int(img_el.bbox.y0 * scale_y))
        x1 = min(img_width - 1, int(img_el.bbox.x1 * scale_x))
        y1 = min(img_height - 1, int(img_el.bbox.y1 * scale_y))
        if x1 > x0 and y1 > y0:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), _IMAGE_COLOUR, -1)

    return canvas


def create_covered_mask(twin: np.ndarray) -> np.ndarray:
    """Return a binary mask where the twin has non-white pixels (= covered regions)."""
    white = np.all(twin == 255, axis=2)
    mask = np.where(white, np.uint8(0), np.uint8(255))
    return mask
