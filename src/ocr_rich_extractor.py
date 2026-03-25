"""OCR-based rich extraction for scanned PDF pages.

When a PDF page has no embedded text layer (fully scanned), the standard
rich_extractor.py produces empty text blocks and one giant page-image.
This module produces the SAME JSON schema but derives everything from
Tesseract OCR + OpenCV layout analysis, so that reconstruct_pdf.py can
rebuild the page with real positioned text instead of a pasted scan.

Output schema matches rich_extractor.extract_rich_page() exactly:
  - text_blocks[].lines[].spans[] with origin, size, font, color, bbox
  - drawings[] for detected table grid lines
  - images[] only for real figures (not the full-page scan)
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

logger = logging.getLogger(__name__)

# ── Tesseract setup ──────────────────────────────────────────────────────────

def _get_tesseract():
    """Import and configure pytesseract."""
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    return pytesseract


# ── Page rendering ───────────────────────────────────────────────────────────

def _render_page(pdf_path: str, page_num: int, dpi: int = 300
                 ) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to BGR numpy array at the given DPI.

    Returns (image, page_width_pts, page_height_pts).
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


# ── Scanned-page detection ──────────────────────────────────────────────────

def is_scanned_page(pdf_path: str, page_num: int, threshold: int = 20) -> bool:
    """Return True if the page has fewer than `threshold` embedded text words."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    words = page.get_text("words")
    doc.close()
    return len(words) < threshold


# ── Image preprocessing pipeline for OCR ─────────────────────────────────────

def _deskew(gray: np.ndarray) -> np.ndarray:
    """Detect and correct skew angle of a scanned page.

    Uses Hough line detection to find the dominant angle, then rotates
    to straighten. Only corrects small angles (< 5 degrees) to avoid
    misinterpreting page layout as skew.
    """
    # Threshold for line detection
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Detect lines
    lines = cv2.HoughLinesP(
        thresh, 1, np.pi / 180, threshold=100,
        minLineLength=gray.shape[1] // 8, maxLineGap=10,
    )
    if lines is None or len(lines) < 5:
        return gray

    # Compute angles of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = x2 - x1, y2 - y1
        if abs(dx) < 1:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Only consider near-horizontal lines (within 15 degrees of horizontal)
        if abs(angle) < 15:
            angles.append(angle)

    if not angles:
        return gray

    median_angle = float(np.median(angles))
    # Only correct if skew is noticeable but not extreme
    if abs(median_angle) < 0.1 or abs(median_angle) > 5.0:
        return gray

    logger.debug("  Deskew: correcting %.2f degree skew", median_angle)
    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        gray, M, (w, h), flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _remove_lines(gray: np.ndarray) -> np.ndarray:
    """Remove horizontal and vertical lines from the image.

    Table borders and decorative lines confuse Tesseract into reading
    them as characters (=, |, -, etc.).  This detects long lines via
    morphological operations and inpaints over them, leaving text intact.
    """
    h, w = gray.shape
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )

    # Detect horizontal lines (wide kernel)
    h_kernel_size = max(w // 25, 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_size, 1))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Detect vertical lines (tall kernel)
    v_kernel_size = max(h // 25, 30)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_size))
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # Also detect shorter decorative lines (borders, underlines, logo elements)
    # that are too short for the main detection but still confuse OCR
    h_short_size = max(w // 60, 15)
    h_short_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_short_size, 1))
    h_short = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_short_kern, iterations=2)
    # Only keep short lines that are very thin (height < 4px) — true decorative lines
    contours_short, _ = cv2.findContours(
        h_short, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    h_decor = np.zeros_like(h_short)
    for cnt in contours_short:
        _, _, cw, ch = cv2.boundingRect(cnt)
        # Thin and wide = decorative line, not text
        if ch <= 4 and cw > h_short_size:
            cv2.drawContours(h_decor, [cnt], -1, 255, thickness=cv2.FILLED)

    # Combine into a single line mask
    line_mask = cv2.add(h_lines, v_lines)
    line_mask = cv2.add(line_mask, h_decor)

    # Dilate the mask slightly to catch edges of thick lines
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    line_mask = cv2.dilate(line_mask, dilate_kernel, iterations=1)

    # Inpaint: replace line pixels with surrounding background
    cleaned = cv2.inpaint(gray, line_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    return cleaned


def _remove_non_text_elements(gray: np.ndarray) -> np.ndarray:
    """Remove graphic elements (icons, logos, decorative marks) that aren't text.

    Uses connected component analysis on the binarized image to classify
    each blob.  Text characters have predictable aspect ratios and sizes
    relative to their neighbours.  Non-text elements (lightning bolts,
    logos, decorative symbols) tend to be:
      - Isolated (far from other components of similar size)
      - Unusually shaped (very tall and narrow, or nearly square, while
        surrounding text is rectangular)
      - Vertically offset from the text baseline

    Rather than trying to identify every possible graphic, this focuses on
    components that would confuse Tesseract — those that are near text and
    could be mis-segmented as part of a word.
    """
    # Binarize to find components
    binary_inv = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_inv, connectivity=8
    )

    if num_labels < 3:
        return gray

    h, w = gray.shape

    # Gather component info (skip background label 0)
    comps = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]
        comps.append({
            "label": i, "area": area, "w": cw, "h": ch,
            "cx": cx, "cy": cy,
            "x": stats[i, cv2.CC_STAT_LEFT],
            "y": stats[i, cv2.CC_STAT_TOP],
            "aspect": ch / max(cw, 1),
            "solidity": area / max(cw * ch, 1),
        })

    # Compute typical text character statistics
    # Filter to components that look like characters (reasonable size range)
    char_candidates = [
        c for c in comps
        if 30 < c["area"] < 5000
        and 0.3 < c["aspect"] < 5.0
        and c["solidity"] > 0.15
    ]

    if not char_candidates:
        return gray

    median_h = float(np.median([c["h"] for c in char_candidates]))
    median_area = float(np.median([c["area"] for c in char_candidates]))

    remove_mask = np.zeros((h, w), dtype=np.uint8)

    for c in comps:
        # Very small speckles — always remove
        if c["area"] < 15:
            remove_mask[labels == c["label"]] = 255
            continue

        # Skip components that look like normal text characters
        if 0.3 * median_h < c["h"] < 3.0 * median_h and c["solidity"] > 0.2:
            continue

        # Tall thin components (much taller than text) — likely decorative
        if c["h"] > 3.0 * median_h and c["w"] < 0.5 * median_h:
            remove_mask[labels == c["label"]] = 255
            continue

        # Very low solidity + unusual aspect — likely icon/graphic
        if c["solidity"] < 0.15 and c["area"] > 50:
            remove_mask[labels == c["label"]] = 255
            continue

        # Tiny components that are isolated (not near other chars)
        if c["area"] < 0.3 * median_area:
            # Check if there are other components nearby at similar y
            nearby = [
                o for o in char_candidates
                if abs(o["cy"] - c["cy"]) < median_h
                and abs(o["cx"] - c["cx"]) < median_h * 5
                and o["label"] != c["label"]
            ]
            if not nearby:
                remove_mask[labels == c["label"]] = 255

    if np.count_nonzero(remove_mask) > 0:
        cleaned = cv2.inpaint(gray, remove_mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
        return cleaned
    return gray


def _remove_small_noise_components(gray: np.ndarray) -> np.ndarray:
    """Remove small connected components that are likely icons/logos/artifacts.

    After binarization, small isolated blobs (not connected to text regions)
    are noise — lightning bolts, stray marks, scan artifacts. Remove them
    to prevent Tesseract from reading them as characters.
    """
    # Work on inverted binary (text is white)
    binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_inv, connectivity=8
    )

    h, w = gray.shape
    page_area = h * w

    # Compute median component area (excluding background label 0)
    if num_labels < 2:
        return gray
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    median_area = float(np.median(areas))

    # Build mask of components to remove
    remove_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_w = stats[i, cv2.CC_STAT_WIDTH]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(comp_w, comp_h) / max(min(comp_w, comp_h), 1)

        # Very small speckles (< 15 pixels) — always noise
        if area < 15:
            remove_mask[labels == i] = 255
            continue

        # Small-ish blobs that are non-text shaped:
        # - Too small to be a character at scan DPI (< 50 px)
        # - Nearly square or irregular (aspect ratio < 2) — icons, not letters
        # - But NOT if they're part of a dense text region
        if area < 50 and aspect < 2.0:
            remove_mask[labels == i] = 255

    if np.count_nonzero(remove_mask) > 0:
        # Inpaint over removed components
        cleaned = cv2.inpaint(gray, remove_mask, inpaintRadius=2, flags=cv2.INPAINT_NS)
        return cleaned
    return gray


def preprocess_for_ocr(img: np.ndarray) -> np.ndarray:
    """Full preprocessing pipeline to prepare a scanned page for Tesseract.

    Steps (in order):
      1. Grayscale conversion
      2. Denoising — remove scan speckles and artifacts
      3. Deskew — straighten tilted scans
      4. CLAHE — normalize contrast for uneven lighting / faded scans
      5. Line removal — erase table borders / decorative lines that Tesseract
         misreads as characters (=, |, etc.)
      6. Adaptive binarization — clean black text on white background

    Returns a grayscale uint8 image optimized for OCR.
    """
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Denoise — remove scan noise without blurring text edges
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 3. Deskew
    gray = _deskew(gray)

    # 4. CLAHE — adaptive contrast normalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 5. Line removal — erase table/border lines before OCR
    gray = _remove_lines(gray)

    # 6. Remove non-text graphic elements (logos, icons, decorative marks)
    gray = _remove_non_text_elements(gray)

    # 7. Adaptive binarization — Otsu gives clean black/white
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return gray


# ── OCR text cleaning ────────────────────────────────────────────────────────

def _clean_ocr_word(text: str) -> str:
    """Clean up a single OCR word by removing common Tesseract artifacts.

    Handles:
      - Leading/trailing noise characters from line remnants (=, |, ~, etc.)
      - Stray punctuation that doesn't belong (e.g. "=SELECTRIC=" → "ELECTRIC")
      - Preserves legitimate punctuation in context (e.g. "Ph:", "$1,200")
    """
    # Strip leading/trailing characters that are common line-detection artifacts
    noise_chars = "=|~`_][}{<>"
    cleaned = text.strip(noise_chars)

    if not cleaned:
        return text

    return cleaned


# Common two-letter word beginnings in English — used to avoid stripping
# real leading characters during artifact repair.  Covers consonant+vowel,
# consonant clusters, and vowel-start bigrams.
_COMMON_BIGRAM_STARTS = {
    # Vowel starts
    'AB', 'AC', 'AD', 'AF', 'AG', 'AL', 'AM', 'AN', 'AP', 'AR', 'AS', 'AT', 'AU', 'AV', 'AW',
    'EA', 'EB', 'EC', 'ED', 'EF', 'EL', 'EM', 'EN', 'EQ', 'ER', 'ES', 'EV', 'EX',
    'ID', 'IF', 'IG', 'IL', 'IM', 'IN', 'IR', 'IS',
    'OB', 'OC', 'OF', 'ON', 'OP', 'OR', 'OT', 'OU', 'OV', 'OX',
    'UL', 'UN', 'UP', 'UR', 'US', 'UT',
    # Consonant + vowel
    'BA', 'BE', 'BI', 'BO', 'BU', 'BY',
    'CA', 'CE', 'CI', 'CO', 'CU', 'CY',
    'DA', 'DE', 'DI', 'DO', 'DR', 'DU', 'DW', 'DY',
    'FA', 'FE', 'FI', 'FL', 'FO', 'FR', 'FU',
    'GA', 'GE', 'GI', 'GL', 'GO', 'GR', 'GU',
    'HA', 'HE', 'HI', 'HO', 'HU', 'HY',
    'JA', 'JO', 'JU',
    'KE', 'KI', 'KN',
    'LA', 'LE', 'LI', 'LO', 'LU',
    'MA', 'ME', 'MI', 'MO', 'MU', 'MY',
    'NA', 'NE', 'NI', 'NO', 'NU',
    'PA', 'PE', 'PH', 'PI', 'PL', 'PO', 'PR', 'PU',
    'QU',
    'RA', 'RE', 'RI', 'RO', 'RU',
    'SA', 'SC', 'SE', 'SH', 'SI', 'SK', 'SL', 'SM', 'SN', 'SO', 'SP', 'SQ', 'ST', 'SU', 'SW', 'SY',
    'TA', 'TE', 'TH', 'TI', 'TO', 'TR', 'TU', 'TW', 'TY',
    'VA', 'VE', 'VI', 'VO',
    'WA', 'WE', 'WH', 'WI', 'WO', 'WR',
    'YA', 'YE', 'YI',
    'ZA', 'ZE', 'ZO',
}


def _repair_leading_artifact(text: str, confidence: int = 100) -> str:
    """Fix words where a graphic element was OCR'd as a leading character.

    When icons/logos touch text, Tesseract often prepends a spurious
    character (e.g. lightning bolt read as 'S', checkmark as 'V').

    Uses two signals:
      1. Bigram check — if the first two chars don't form a common English
         word-start, the leading char is likely an artifact.
      2. Confidence — if Tesseract's confidence for the word is low (< 60),
         even common bigrams are suspect and the leading char gets stripped.

    Only applies to alphabetic words >= 5 chars where removing char[0]
    still leaves a coherent word >= 4 chars.
    """
    if len(text) < 5:
        return text

    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 5:
        return text

    first = text[0]
    rest = text[1:]
    rest_alpha = ''.join(c for c in rest if c.isalpha())

    if len(rest_alpha) < 4:
        return text

    # Check if rest forms a coherent word (same case pattern)
    rest_is_word = False
    if rest_alpha.isupper() and first.isupper():
        rest_is_word = True
    elif rest_alpha[0].isupper() and rest_alpha[1:].islower() and first.isupper():
        rest_is_word = True

    if not rest_is_word:
        return text

    first_two = (first + rest[0]).upper()
    bigram_is_common = first_two in _COMMON_BIGRAM_STARTS

    # Low confidence + rest is a valid word → strip even if bigram is common
    # High confidence + uncommon bigram → strip (unlikely real word)
    # High confidence + common bigram → keep (real word)
    if not bigram_is_common:
        return rest
    if confidence < 60:
        return rest

    return text


def _clean_ocr_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Clean OCR artifacts from all words and filter out empty/noise results."""
    cleaned: List[Dict[str, Any]] = []
    for w in words:
        text = _clean_ocr_word(w["text"])

        # Skip words that are purely noise/artifacts
        if not text:
            continue
        # Skip words that are just repeated punctuation (e.g. "===", "|||", "---")
        if re.fullmatch(r'[\W_]+', text) and len(text) > 1:
            continue
        # Skip single-character noise (but keep real single chars like "I", "a", numbers)
        if len(text) == 1 and text in '=|~`_][}{<>':
            continue

        # Repair leading artifact characters from icons/graphics
        text = _repair_leading_artifact(text, confidence=w.get("conf", 100))

        w_copy = w.copy()
        w_copy["text"] = text
        cleaned.append(w_copy)
    return cleaned


# ── OCR with full positional data ────────────────────────────────────────────

def _ocr_page_detailed(img: np.ndarray, dpi: int = 300
                       ) -> List[Dict[str, Any]]:
    """Run Tesseract image_to_data on a preprocessed page image.

    Applies the full preprocessing pipeline before OCR, then cleans
    the results to remove artifacts from line remnants.

    Returns a list of word dicts with pixel-space positions and hierarchy:
      {text, x0, y0, x1, y1, conf, block_num, par_num, line_num, word_num,
       height_px}
    """
    from PIL import Image as PILImage
    pytesseract = _get_tesseract()

    # Preprocess the image for OCR quality
    preprocessed = preprocess_for_ocr(img)

    pil_img = PILImage.fromarray(preprocessed)
    data = pytesseract.image_to_data(
        pil_img,
        config="--psm 3 --oem 3",
        output_type=pytesseract.Output.DICT,
    )

    words: List[Dict[str, Any]] = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not text or conf < 25:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append({
            "text": text,
            "x0": float(x),
            "y0": float(y),
            "x1": float(x + w),
            "y1": float(y + h),
            "height_px": float(h),
            "conf": conf,
            "block_num": int(data["block_num"][i]),
            "par_num": int(data["par_num"][i]),
            "line_num": int(data["line_num"][i]),
            "word_num": int(data["word_num"][i]),
        })

    # Clean OCR artifacts
    words = _clean_ocr_words(words)
    return words


# ── Group OCR words into the rich-extractor schema ───────────────────────────

def _words_to_rich_text_blocks(
    words: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
    dpi: int = 300,
) -> List[Dict[str, Any]]:
    """Convert OCR words (pixel coords) to rich-extractor text_blocks schema.

    Groups words by (block_num, par_num, line_num) to reconstruct lines,
    then by block_num to reconstruct blocks.

    Coordinates are converted from pixel space to PDF-point space.
    Font size is estimated from word height in pixels → points.
    """
    if not words:
        return []

    pts_per_px = 72.0 / dpi

    # Group into lines: (block_num, par_num, line_num) → [words]
    from collections import defaultdict
    lines_map: Dict[Tuple[int, int, int], List[Dict]] = defaultdict(list)
    for w in words:
        key = (w["block_num"], w["par_num"], w["line_num"])
        lines_map[key].append(w)

    # Group lines into blocks by block_num
    blocks_map: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for key in sorted(lines_map.keys()):
        blocks_map[key[0]].append(key)

    text_blocks: List[Dict[str, Any]] = []

    for block_num in sorted(blocks_map.keys()):
        line_keys = blocks_map[block_num]
        lines_out: List[Dict[str, Any]] = []

        block_x0, block_y0 = float("inf"), float("inf")
        block_x1, block_y1 = 0.0, 0.0

        for lk in line_keys:
            line_words = sorted(lines_map[lk], key=lambda w: w["x0"])
            if not line_words:
                continue

            spans_out: List[Dict[str, Any]] = []
            line_x0 = min(w["x0"] for w in line_words)
            line_y0 = min(w["y0"] for w in line_words)
            line_x1 = max(w["x1"] for w in line_words)
            line_y1 = max(w["y1"] for w in line_words)

            for w in line_words:
                # Estimate font size from word height (pixels → points)
                font_size_pt = round(w["height_px"] * pts_per_px * 0.85, 2)
                font_size_pt = max(4.0, min(font_size_pt, 72.0))

                # Origin: baseline left — approximate as bottom-left minus descender
                origin_x = w["x0"] * scale_x
                origin_y = w["y1"] * scale_y - (font_size_pt * 0.15)

                spans_out.append({
                    "text": w["text"],
                    "font": "Helvetica",
                    "size": font_size_pt,
                    "flags": 0,
                    "color": "#000000",
                    "origin": [round(origin_x, 4), round(origin_y, 4)],
                    "bbox": [
                        round(w["x0"] * scale_x, 4),
                        round(w["y0"] * scale_y, 4),
                        round(w["x1"] * scale_x, 4),
                        round(w["y1"] * scale_y, 4),
                    ],
                    "ascender": round(font_size_pt * 0.8, 4),
                    "descender": round(-font_size_pt * 0.2, 4),
                })

            lines_out.append({
                "bbox": [
                    round(line_x0 * scale_x, 4),
                    round(line_y0 * scale_y, 4),
                    round(line_x1 * scale_x, 4),
                    round(line_y1 * scale_y, 4),
                ],
                "wmode": 0,
                "dir": [1.0, 0.0],
                "spans": spans_out,
            })

            block_x0 = min(block_x0, line_x0)
            block_y0 = min(block_y0, line_y0)
            block_x1 = max(block_x1, line_x1)
            block_y1 = max(block_y1, line_y1)

        if lines_out:
            text_blocks.append({
                "bbox": [
                    round(block_x0 * scale_x, 4),
                    round(block_y0 * scale_y, 4),
                    round(block_x1 * scale_x, 4),
                    round(block_y1 * scale_y, 4),
                ],
                "lines": lines_out,
            })

    return text_blocks


# ── Table grid lines → drawings ─────────────────────────────────────────────

def _detect_table_drawings(
    img: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """Detect table grid lines via OpenCV and return them as drawing dicts.

    These match the rich_extractor drawings[] schema so reconstruct_pdf.py
    can draw them as vector lines.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )

    img_h, img_w = gray.shape

    # Horizontal lines
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(img_w // 30, 20), 1)
    )
    h_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # Vertical lines
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(img_h // 30, 20))
    )
    v_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

    drawings: List[Dict[str, Any]] = []
    seqno = 0

    # Extract horizontal line segments
    for mask, is_horiz in [(h_mask, True), (v_mask, False)]:
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # Convert pixel → PDF points
            p1_x = round(x * scale_x, 4)
            p1_y = round(y * scale_y, 4)
            p2_x = round((x + w) * scale_x, 4)
            p2_y = round((y + h) * scale_y, 4)

            if is_horiz:
                mid_y = round((p1_y + p2_y) / 2, 4)
                items = [{"type": "l", "p1": [p1_x, mid_y], "p2": [p2_x, mid_y]}]
            else:
                mid_x = round((p1_x + p2_x) / 2, 4)
                items = [{"type": "l", "p1": [mid_x, p1_y], "p2": [mid_x, p2_y]}]

            drawings.append({
                "seqno": seqno,
                "items": items,
                "color": [0.0, 0.0, 0.0],
                "fill": None,
                "width": 0.5,
                "lineCap": [0, 0, 0],
                "lineJoin": 0,
                "dashes": "",
                "closePath": False,
                "even_odd": False,
                "fill_opacity": 1.0,
                "stroke_opacity": 1.0,
                "rect": [min(p1_x, p2_x), min(p1_y, p2_y),
                         max(p1_x, p2_x), max(p1_y, p2_y)],
            })
            seqno += 1

    return drawings


# ── Figure detection ─────────────────────────────────────────────────────────

def _detect_figures(
    img: np.ndarray,
    ocr_words: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
    min_area_pct: float = 0.01,
    max_area_pct: float = 0.75,
) -> List[Dict[str, Any]]:
    """Detect actual figures/photos (not the full-page scan) and return as images[].

    Uses contour detection to find large non-text regions with high pixel
    variance (photos, diagrams, logos).  Skips regions that are mostly text
    or that cover >75% of the page (the scan itself).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape
    total_area = img_h * img_w

    # Build a text mask from OCR word positions
    text_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for w in ocr_words:
        x0, y0 = max(0, int(w["x0"])), max(0, int(w["y0"]))
        x1, y1 = min(img_w, int(w["x1"])), min(img_h, int(w["y1"]))
        text_mask[y0:y1, x0:x1] = 255

    # Dilate text mask to cover surrounding area
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    text_mask = cv2.dilate(text_mask, dilate_kernel, iterations=2)

    # Edge detection to find figure boundaries
    edges = cv2.Canny(gray, 30, 100)
    # Close gaps in edges
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    images_out: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        area_pct = area / total_area

        if area_pct < min_area_pct or area_pct > max_area_pct:
            continue

        # Check if this region is mostly text (skip it)
        region_text = text_mask[y:y+h, x:x+w]
        text_coverage = np.count_nonzero(region_text) / max(area, 1)
        if text_coverage > 0.5:
            continue

        # Check pixel variance — figures have higher variance than blank areas
        region_gray = gray[y:y+h, x:x+w]
        if region_gray.std() < 15:
            continue

        # Crop and encode as PNG
        crop = img[y:y+h, x:x+w]
        _, png_buf = cv2.imencode(".png", crop)
        b64_data = base64.b64encode(png_buf.tobytes()).decode("ascii")

        images_out.append({
            "bbox": [
                round(x * scale_x, 4),
                round(y * scale_y, 4),
                round((x + w) * scale_x, 4),
                round((y + h) * scale_y, 4),
            ],
            "width": w,
            "height": h,
            "ext": "png",
            "data_b64": b64_data,
            "xref": -1,  # no xref for OCR-detected images
        })

    return images_out


# ── Checkbox detection ───────────────────────────────────────────────────────

def _detect_checkboxes(
    img: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """Detect checkbox form elements via OpenCV contour analysis.

    Finds small square-ish contours that match checkbox characteristics:
      - Nearly square aspect ratio (0.7 - 1.3)
      - Consistent size range (typical checkbox = 10-25px at 300 DPI)
      - Has 4 corners (approximate polygon)

    Classifies each as checked or unchecked by measuring ink density
    inside the box: checked boxes have an X/checkmark filling them.

    Returns a list of form_element dicts with:
      {type: "checkbox", checked: bool, bbox: [x0,y0,x1,y1] in PDF points}
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_h, img_w = gray.shape

    # Binarize
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE,
    )

    if hierarchy is None:
        return []

    candidates: List[Dict[str, Any]] = []

    # Typical checkbox size at 300 DPI: ~30-55 pixels (roughly 8-14pt box)
    min_side = 25
    max_side = 65

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Size filter — checkboxes are small, consistent squares
        if w < min_side or w > max_side or h < min_side or h > max_side:
            continue

        # Aspect ratio must be very close to square (stricter than before)
        aspect = w / max(h, 1)
        if aspect < 0.8 or aspect > 1.2:
            continue

        # Approximate polygon — checkboxes should have exactly 4 corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) < 4 or len(approx) > 6:
            continue

        # Must be a clean rectangle (high solidity = no irregular shape)
        cnt_area = cv2.contourArea(cnt)
        rect_area = w * h
        solidity = cnt_area / max(rect_area, 1)
        if solidity < 0.7:
            continue

        # The contour must be a hollow rectangle (border only), not a filled square.
        # Check that the border region has ink but the center is relatively empty
        # or has an X pattern.  A solid filled square is not a checkbox.
        border_mask = np.zeros_like(gray[y:y+h, x:x+w])
        cv2.drawContours(border_mask, [cnt - np.array([x, y])], -1, 255, 2)
        border_ink = np.count_nonzero(thresh[y:y+h, x:x+w] & border_mask)
        border_total = max(np.count_nonzero(border_mask), 1)
        border_coverage = border_ink / border_total
        # Border should have high ink coverage (it's drawn)
        if border_coverage < 0.4:
            continue

        # Measure ink density inside the box to determine checked/unchecked
        margin = max(4, int(w * 0.2))
        inner_y0 = min(y + margin, y + h - 1)
        inner_y1 = max(y + h - margin, y + 1)
        inner_x0 = min(x + margin, x + w - 1)
        inner_x1 = max(x + w - margin, x + 1)

        inner = thresh[inner_y0:inner_y1, inner_x0:inner_x1]
        if inner.size == 0:
            continue

        ink_ratio = np.count_nonzero(inner) / max(inner.size, 1)
        # Checked boxes have significant ink inside (X mark ~20-50%)
        # Unchecked boxes have very little ink inside (< 5%)
        # Anything in between is ambiguous (skip as likely not a checkbox)
        if 0.05 < ink_ratio < 0.15:
            continue  # ambiguous — probably not a checkbox

        checked = ink_ratio >= 0.15

        candidates.append({
            "type": "checkbox",
            "checked": checked,
            "bbox": [
                round(x * scale_x, 4),
                round(y * scale_y, 4),
                round((x + w) * scale_x, 4),
                round((y + h) * scale_y, 4),
            ],
            "bbox_px": [x, y, x + w, y + h],
            "ink_ratio": round(ink_ratio, 3),
            "side_px": (w + h) / 2,
        })

    # Checkboxes on a form are consistent in size.  Filter out candidates
    # whose size deviates too much from the most common size.
    if len(candidates) >= 3:
        sides = [c["side_px"] for c in candidates]
        median_side = float(np.median(sides))
        candidates = [
            c for c in candidates
            if abs(c["side_px"] - median_side) < median_side * 0.3
        ]

    checkboxes = candidates

    # Deduplicate overlapping detections (keep the one with highest solidity)
    if len(checkboxes) < 2:
        return checkboxes

    deduped: List[Dict[str, Any]] = []
    used = set()
    for i, cb in enumerate(checkboxes):
        if i in used:
            continue
        best = cb
        for j, other in enumerate(checkboxes):
            if j <= i or j in used:
                continue
            # Check overlap
            b1, b2 = cb["bbox_px"], other["bbox_px"]
            ox0 = max(b1[0], b2[0])
            oy0 = max(b1[1], b2[1])
            ox1 = min(b1[2], b2[2])
            oy1 = min(b1[3], b2[3])
            if ox0 < ox1 and oy0 < oy1:
                used.add(j)
        deduped.append(best)

    # Remove internal fields from output
    for cb in deduped:
        cb.pop("bbox_px", None)
        cb.pop("ink_ratio", None)
        cb.pop("side_px", None)

    return deduped


def _replace_checkbox_text_in_words(
    words: List[Dict[str, Any]],
    checkboxes: List[Dict[str, Any]],
    scale_x: float,
    scale_y: float,
) -> List[Dict[str, Any]]:
    """Replace OCR'd checkbox text (X, Xl, Cl, etc.) with semantic markers.

    When Tesseract reads a checked checkbox, it often produces 'X', '[X]',
    'Xl', etc.  When it reads an empty checkbox, it produces 'Cl', '[]',
    'O', etc.  This finds OCR words that overlap with detected checkbox
    bounding boxes and replaces their text with a semantic marker that
    the reconstruction can interpret.
    """
    if not checkboxes:
        return words

    updated: List[Dict[str, Any]] = []
    for w in words:
        # Convert word pixel bbox to PDF points for comparison
        w_pdf = [
            w["x0"] * scale_x, w["y0"] * scale_y,
            w["x1"] * scale_x, w["y1"] * scale_y,
        ]

        replaced = False
        for cb in checkboxes:
            cb_bbox = cb["bbox"]
            # Check if word center falls inside checkbox bbox
            wcx = (w_pdf[0] + w_pdf[2]) / 2
            wcy = (w_pdf[1] + w_pdf[3]) / 2
            if (cb_bbox[0] <= wcx <= cb_bbox[2] and
                    cb_bbox[1] <= wcy <= cb_bbox[3]):
                # This word is inside a checkbox — replace with marker
                marker = "\u2611" if cb["checked"] else "\u2610"  # ☑ or ☐
                w_copy = w.copy()
                w_copy["text"] = marker
                w_copy["is_checkbox"] = True
                updated.append(w_copy)
                replaced = True
                break

        if not replaced:
            updated.append(w)

    return updated


# ── Core: OCR-based rich extraction ─────────────────────────────────────────

def extract_rich_page_ocr(
    pdf_path: str,
    page_num: int,
    dpi: int = 300,
) -> Dict[str, Any]:
    """Extract the visual state of a scanned page via OCR + OpenCV.

    Returns the same schema as rich_extractor.extract_rich_page() so that
    reconstruct_pdf.py works identically on scanned and digital pages.
    """
    img, page_w, page_h = _render_page(pdf_path, page_num, dpi=dpi)
    img_h, img_w = img.shape[:2]

    scale_x = page_w / img_w   # pixel → PDF points
    scale_y = page_h / img_h

    logger.info(
        "OCR rich extraction p%d: %dx%d px @ %d DPI → %.1f×%.1f pts",
        page_num + 1, img_w, img_h, dpi, page_w, page_h,
    )

    # 1. OCR — get every word with position
    ocr_words = _ocr_page_detailed(img, dpi=dpi)
    logger.info("  OCR found %d words on page %d", len(ocr_words), page_num + 1)

    # 2. Detect checkboxes and replace OCR'd checkbox text with semantic markers
    checkboxes = _detect_checkboxes(img, scale_x, scale_y)
    logger.info("  Detected %d checkboxes on page %d", len(checkboxes), page_num + 1)
    if checkboxes:
        ocr_words = _replace_checkbox_text_in_words(
            ocr_words, checkboxes, scale_x, scale_y,
        )

    # 3. Build text blocks in rich-extractor schema
    text_blocks = _words_to_rich_text_blocks(ocr_words, scale_x, scale_y, dpi)

    # 4. Detect table grid lines → drawings
    drawings = _detect_table_drawings(img, scale_x, scale_y)
    logger.info("  Detected %d table line segments on page %d", len(drawings), page_num + 1)

    # 5. Detect actual figures (not the full-page scan)
    images = _detect_figures(img, ocr_words, scale_x, scale_y)
    logger.info("  Detected %d figures on page %d", len(images), page_num + 1)

    return {
        "version": "1.1-ocr",
        "page_num": page_num,
        "width": round(page_w, 4),
        "height": round(page_h, 4),
        "ctm_scale": 1.0,
        "text_blocks": text_blocks,
        "drawings": drawings,
        "images": images,
        "fonts_used": ["Helvetica"],
        "form_elements": checkboxes,
        "ocr_stats": {
            "total_words": len(ocr_words),
            "avg_confidence": (
                round(sum(w["conf"] for w in ocr_words) / len(ocr_words), 1)
                if ocr_words else 0.0
            ),
            "dpi": dpi,
        },
    }
