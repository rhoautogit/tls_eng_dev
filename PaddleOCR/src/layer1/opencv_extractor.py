"""Layer 1 – OpenCV image-based table and content extractor.

Renders each PDF page to a raster image, applies morphological operations to
detect table grids, reconstructs cell boundaries, and fills cells with text
from PyMuPDF character positions.  Handles both digital and scanned pages.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np

from ..models import (
    BoundingBox,
    ExtractionParameters,
    ImageElement,
    PageExtractionResult,
    Table,
    TextBlock,
)

logger = logging.getLogger(__name__)


# ─── Image Utilities ──────────────────────────────────────────────────────────

def render_page_to_image(
    pdf_path: str, page_num: int, dpi: int = 200
) -> Tuple[np.ndarray, float, float]:
    """Render a PDF page to a BGR numpy array.

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


def preprocess_for_table_detection(
    img: np.ndarray, params: ExtractionParameters
) -> np.ndarray:
    """Convert to grayscale and apply optional enhancement passes."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if params.opencv_use_noise_reduction:
        gray = cv2.fastNlMeansDenoising(gray, h=10)

    if params.opencv_use_clahe:
        clahe = cv2.createCLAHE(
            clipLimit=params.opencv_clahe_clip,
            tileGridSize=tuple(params.opencv_clahe_grid),
        )
        gray = clahe.apply(gray)

    if params.opencv_use_sharpening:
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)

    return gray


def detect_lines(
    gray: np.ndarray, params: ExtractionParameters
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (horizontal_mask, vertical_mask) binary images."""
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        params.opencv_threshold_block_size,
        params.opencv_threshold_constant,
    )
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(params.opencv_kernel_h))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, tuple(params.opencv_kernel_v))
    horizontal = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, h_kern, iterations=params.opencv_iterations
    )
    vertical = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, v_kern, iterations=params.opencv_iterations
    )
    return horizontal, vertical


# ─── Cell Detection ───────────────────────────────────────────────────────────

def find_cells_in_image(
    horizontal: np.ndarray,
    vertical: np.ndarray,
    min_cell_area: int = 50,
) -> List[BoundingBox]:
    """Find cell bounding boxes (in pixel coordinates) from line masks."""
    grid = cv2.add(horizontal, vertical)
    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cells: List[BoundingBox] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_cell_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        cells.append(BoundingBox(float(x), float(y), float(x + w), float(y + h)))
    return cells


def _bbox_2d_gap(a: BoundingBox, b: BoundingBox) -> float:
    """Return the max of x-gap and y-gap between two bboxes.

    Returns 0 when the boxes touch or overlap in both dimensions.
    """
    x_gap = max(0.0, max(a.x0, b.x0) - min(a.x1, b.x1))
    y_gap = max(0.0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return max(x_gap, y_gap)


def cluster_cells(
    cells: List[BoundingBox], img_width: int, gap_pct: float = 0.05
) -> List[List[BoundingBox]]:
    """Group cells that belong to the same table based on 2-D proximity.

    Two cells are considered adjacent when their bounding boxes are within
    ``gap`` pixels of each other in *both* the x and y axes simultaneously.
    This prevents cells that merely share an x-boundary from being grouped
    despite being far apart vertically.
    """
    if not cells:
        return []
    gap = img_width * gap_pct
    sorted_cells = sorted(cells, key=lambda c: (c.y0, c.x0))
    tables: List[List[BoundingBox]] = [[sorted_cells[0]]]
    for cell in sorted_cells[1:]:
        placed = False
        for cluster in tables:
            if any(_bbox_2d_gap(cell, c) < gap for c in cluster):
                cluster.append(cell)
                placed = True
                break
        if not placed:
            tables.append([cell])
    return tables


def cells_to_grid(
    cells: List[BoundingBox],
) -> Tuple[Optional[BoundingBox], List[List[Optional[BoundingBox]]]]:
    """Arrange a flat list of cells into a 2-D row/column grid."""
    if not cells:
        return None, []

    table_bbox = BoundingBox(
        min(c.x0 for c in cells),
        min(c.y0 for c in cells),
        max(c.x1 for c in cells),
        max(c.y1 for c in cells),
    )

    avg_h = sum(c.height for c in cells) / len(cells)
    avg_w = sum(c.width for c in cells) / len(cells)
    row_tol = max(avg_h * 0.4, 2.0)
    col_tol = max(avg_w * 0.4, 2.0)

    def snap(val: float, tolerance: float, seen: List[float]) -> float:
        for s in seen:
            if abs(val - s) <= tolerance:
                return s
        seen.append(val)
        return val

    row_centers: List[float] = []
    col_centers: List[float] = []
    for c in cells:
        snap((c.y0 + c.y1) / 2, row_tol, row_centers)
        snap((c.x0 + c.x1) / 2, col_tol, col_centers)

    row_centers.sort()
    col_centers.sort()
    num_rows, num_cols = len(row_centers), len(col_centers)
    grid: List[List[Optional[BoundingBox]]] = [
        [None] * num_cols for _ in range(num_rows)
    ]

    for c in cells:
        ry = snap((c.y0 + c.y1) / 2, row_tol, row_centers)
        rx = snap((c.x0 + c.x1) / 2, col_tol, col_centers)
        ri = row_centers.index(ry)
        ci = col_centers.index(rx)
        if grid[ri][ci] is None:
            grid[ri][ci] = c
    return table_bbox, grid


# ─── Text Extraction Helpers ──────────────────────────────────────────────────

def get_word_positions(pdf_path: str, page_num: int) -> List[Dict]:
    """Return word-level bounding boxes from PyMuPDF."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    words = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
    doc.close()
    return [{"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": w[4]} for w in words]


def text_in_bbox(words: List[Dict], bbox: BoundingBox) -> str:
    """Collect all words whose centres fall inside bbox."""
    matched = [
        w["text"]
        for w in words
        if bbox.contains_point((w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2)
    ]
    return " ".join(matched)


def _ocr_cell_from_image(img: np.ndarray, cell_px: BoundingBox) -> str:
    """Run Tesseract OCR on a cropped cell region of the page image."""
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        x0, y0 = int(cell_px.x0), int(cell_px.y0)
        x1, y1 = int(cell_px.x1), int(cell_px.y1)
        h, w = img.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        if x1 - x0 < 5 or y1 - y0 < 5:
            return ""
        crop = img[y0:y1, x0:x1]
        text = pytesseract.image_to_string(crop, config="--psm 6").strip()
        return text
    except Exception:
        return ""


def _ocr_full_page(img: np.ndarray) -> List[Dict]:
    """Run Tesseract OCR on the full page image and return word positions."""
    try:
        import pytesseract
        from PIL import Image as PILImage
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        data = pytesseract.image_to_data(pil_img, config="--psm 6", output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text or int(data["conf"][i]) < 30:
                continue
            x = data["left"][i]
            y = data["top"][i]
            w = data["width"][i]
            h = data["height"][i]
            words.append({"x0": float(x), "y0": float(y),
                          "x1": float(x + w), "y1": float(y + h), "text": text})
        return words
    except Exception as e:
        logger.debug("Full page OCR failed: %s", e)
        return []


# ─── Extractor ────────────────────────────────────────────────────────────────

class OpenCVExtractor:
    """Extracts tables from PDF pages via image processing."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        oc = config.get("extraction", {}).get("opencv", {})
        self._min_cell_area: int = int(oc.get("min_cell_area", 50))
        self._min_table_cells: int = int(oc.get("min_table_cells", 2))
        self._cluster_gap_pct: float = float(oc.get("table_cluster_gap_pct", 0.05))

    def extract_page(
        self,
        pdf_path: str,
        page_num: int,
        params: ExtractionParameters,
    ) -> PageExtractionResult:
        img, pdf_w, pdf_h = render_page_to_image(pdf_path, page_num, params.opencv_dpi)
        img_h, img_w = img.shape[:2]
        scale_x = pdf_w / img_w
        scale_y = pdf_h / img_h

        # Optionally crop to target region
        crop_x0, crop_y0 = 0.0, 0.0
        work_img = img
        if params.target_bbox is not None:
            tb = params.target_bbox
            px0 = max(0, int(tb.x0 / scale_x))
            py0 = max(0, int(tb.y0 / scale_y))
            px1 = min(img_w, int(tb.x1 / scale_x))
            py1 = min(img_h, int(tb.y1 / scale_y))
            work_img = img[py0:py1, px0:px1]
            crop_x0, crop_y0 = tb.x0, tb.y0
            # Recompute img dimensions for cropped region
            img_h, img_w = work_img.shape[:2]

        word_positions = get_word_positions(pdf_path, page_num)
        is_scanned = len(word_positions) < 20
        gray = preprocess_for_table_detection(work_img, params)
        horizontal, vertical = detect_lines(gray, params)

        # If scanned page (no text layer), use Tesseract OCR for word positions
        ocr_word_positions: List[Dict] = []
        if is_scanned:
            ocr_word_positions = _ocr_full_page(work_img)
            logger.debug("Page %d: scanned — OCR found %d words", page_num, len(ocr_word_positions))

        cells_px = find_cells_in_image(horizontal, vertical, self._min_cell_area)
        clusters = cluster_cells(cells_px, img_w, self._cluster_gap_pct)

        tables: List[Table] = []
        covered_bboxes: List[BoundingBox] = []

        for cluster in clusters:
            if len(cluster) < self._min_table_cells:
                continue
            table_bbox_px, grid_px = cells_to_grid(cluster)
            if not grid_px or table_bbox_px is None:
                continue

            # Convert pixel coords → PDF coords
            def to_pdf(bb: Optional[BoundingBox]) -> Optional[BoundingBox]:
                if bb is None:
                    return None
                return BoundingBox(
                    bb.x0 * scale_x + crop_x0,
                    bb.y0 * scale_y + crop_y0,
                    bb.x1 * scale_x + crop_x0,
                    bb.y1 * scale_y + crop_y0,
                )

            table_bbox_pdf = to_pdf(table_bbox_px)
            grid_pdf = [[to_pdf(cell) for cell in row] for row in grid_px]

            table_data: List[List[str]] = []
            for ri, row in enumerate(grid_pdf):
                row_data: List[str] = []
                for ci, cell_bb in enumerate(row):
                    if cell_bb is None:
                        row_data.append("")
                    elif is_scanned:
                        # Use OCR: try word positions first, fall back to cell crop
                        cell_text = text_in_bbox(ocr_word_positions, grid_px[ri][ci]) if ocr_word_positions else ""
                        if not cell_text.strip():
                            cell_text = _ocr_cell_from_image(work_img, grid_px[ri][ci])
                        row_data.append(cell_text)
                    else:
                        row_data.append(text_in_bbox(word_positions, cell_bb))
                table_data.append(row_data)

            total = sum(len(r) for r in table_data)
            filled = sum(1 for r in table_data for c in r if c.strip())
            conf = filled / total if total > 0 else 0.0

            tables.append(
                Table(
                    data=table_data,
                    bbox=table_bbox_pdf,
                    page_num=page_num,
                    confidence=conf,
                    source="opencv",
                )
            )
            if table_bbox_pdf:
                covered_bboxes.append(table_bbox_pdf)

        # Use OCR word positions for text blocks on scanned pages
        effective_words = ocr_word_positions if is_scanned and ocr_word_positions else word_positions
        text_blocks = self._build_text_blocks(
            effective_words, covered_bboxes, page_num
        )

        return PageExtractionResult(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=tables,
            images=[],
            source="opencv",
            is_scanned=is_scanned,
            page_width=pdf_w,
            page_height=pdf_h,
        )

    def _build_text_blocks(
        self,
        words: List[Dict],
        covered: List[BoundingBox],
        page_num: int,
    ) -> List[TextBlock]:
        """Build text blocks from words not covered by detected tables."""
        uncovered = [
            w
            for w in words
            if not any(
                bb.contains_point((w["x0"] + w["x1"]) / 2, (w["y0"] + w["y1"]) / 2)
                for bb in covered
            )
        ]
        if not uncovered:
            return []

        sorted_words = sorted(uncovered, key=lambda w: (w["y0"], w["x0"]))
        blocks: List[List[Dict]] = [[sorted_words[0]]]
        for w in sorted_words[1:]:
            last = blocks[-1][-1]
            same_line = abs(w["y0"] - last["y0"]) < 8
            close_below = (w["y0"] - last["y1"]) < 12
            if same_line or close_below:
                blocks[-1].append(w)
            else:
                blocks.append([w])

        result: List[TextBlock] = []
        for block in blocks:
            text = " ".join(w["text"] for w in block).strip()
            if not text:
                continue
            result.append(
                TextBlock(
                    text=text,
                    bbox=BoundingBox(
                        min(w["x0"] for w in block),
                        min(w["y0"] for w in block),
                        max(w["x1"] for w in block),
                        max(w["y1"] for w in block),
                    ),
                    page_num=page_num,
                    confidence=0.8,
                    source="opencv",
                )
            )
        return result
