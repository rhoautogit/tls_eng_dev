"""PaddleOCR engine wrapper for the TLS PDF Pipeline.

Provides text detection, text recognition, and table structure recognition
using PaddleOCR. Designed to run as a subprocess so GPU memory is cleanly
released when OCR is complete, leaving VRAM free for Qwen-VL verification.

Two execution modes:
  - GPU: single process, batches all scanned pages sequentially through one
    PaddleOCR instance. GPU handles internal parallelism.
  - CPU: multiple parallel worker processes, each handling a subset of pages.
    Worker count capped by available RAM (each PaddleOCR process ~500MB-1GB).

Usage (in-process, for testing):
    engine = PaddleOCREngine(config)
    result = engine.ocr_image(image_array)
    table = engine.detect_table_structure(image_array)

Usage (subprocess, for production):
    results = run_paddleocr_subprocess(image_paths, config)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Pre-register NVIDIA DLL paths before any PaddlePaddle import
def _register_nvidia_dlls() -> None:
    import os
    import site as _site
    try:
        dirs_to_add = []
        for sp in _site.getsitepackages() + [_site.getusersitepackages()]:
            for subdir in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
                dll_dir = Path(sp) / subdir
                if dll_dir.is_dir():
                    dirs_to_add.append(str(dll_dir))
                    os.add_dll_directory(str(dll_dir))
        # Also prepend to PATH for C++ runtime DLL lookups
        if dirs_to_add:
            os.environ["PATH"] = os.pathsep.join(dirs_to_add) + os.pathsep + os.environ.get("PATH", "")
    except (OSError, AttributeError, TypeError):
        pass

_register_nvidia_dlls()


# -- Data structures ----------------------------------------------------------

@dataclass
class OCRWord:
    """A single word/text region detected by PaddleOCR."""
    text: str
    confidence: float
    bbox: List[float]          # [x0, y0, x1, y1] in pixel coords
    polygon: List[List[float]] # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] quad points

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "polygon": self.polygon,
        }


@dataclass
class OCRLine:
    """A line of text (one or more words in reading order)."""
    text: str
    confidence: float
    bbox: List[float]
    words: List[OCRWord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "words": [w.to_dict() for w in self.words],
        }


@dataclass
class OCRTableCell:
    """A single cell in a detected table."""
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1
    text: str = ""
    bbox: List[float] = field(default_factory=list)


@dataclass
class OCRTable:
    """A table detected by PaddleOCR's table structure recognition."""
    bbox: List[float]
    num_rows: int = 0
    num_cols: int = 0
    cells: List[OCRTableCell] = field(default_factory=list)
    html: str = ""

    def to_grid(self) -> List[List[str]]:
        """Convert cells to a 2D grid of text values."""
        if not self.cells or self.num_rows == 0 or self.num_cols == 0:
            return []
        grid = [["" for _ in range(self.num_cols)] for _ in range(self.num_rows)]
        for cell in self.cells:
            if 0 <= cell.row < self.num_rows and 0 <= cell.col < self.num_cols:
                grid[cell.row][cell.col] = cell.text
        return grid


@dataclass
class OCRPageResult:
    """Complete OCR result for a single page image."""
    lines: List[OCRLine] = field(default_factory=list)
    tables: List[OCRTable] = field(default_factory=list)
    page_confidence: float = 0.0
    word_count: int = 0
    high_confidence_words: int = 0
    flagged_words: int = 0

    def all_text(self) -> str:
        return " ".join(line.text for line in self.lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lines": [l.to_dict() for l in self.lines],
            "tables": [asdict(t) for t in self.tables],
            "page_confidence": self.page_confidence,
            "word_count": self.word_count,
            "high_confidence_words": self.high_confidence_words,
            "flagged_words": self.flagged_words,
        }


# -- PaddleOCR engine (in-process) -------------------------------------------

class PaddleOCREngine:
    """Wraps PaddleOCR for text detection, recognition, and table structure."""

    _shared_instance: Optional["PaddleOCREngine"] = None

    @classmethod
    def get_shared(cls, config: Dict[str, Any]) -> "PaddleOCREngine":
        """Return a shared singleton engine to avoid re-initializing PaddleOCR."""
        if cls._shared_instance is None:
            cls._shared_instance = cls(config)
        return cls._shared_instance

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        paddle_cfg = config.get("scanned_path", {}).get("paddleocr", {})

        self.use_gpu = bool(paddle_cfg.get("use_gpu", False))
        self.lang = str(paddle_cfg.get("lang", "en"))
        self.det_model_dir = paddle_cfg.get("det_model_dir")
        self.rec_model_dir = paddle_cfg.get("rec_model_dir")
        self.table_model_dir = paddle_cfg.get("table_model_dir")
        self.use_angle_cls = bool(paddle_cfg.get("use_angle_cls", True))
        self.show_log = bool(paddle_cfg.get("show_log", False))

        self._ocr = None
        self._table_engine = None

    @staticmethod
    def _setup_dll_paths() -> None:
        """Add NVIDIA cuDNN/cuBLAS DLL paths so PaddlePaddle can find them."""
        import os
        import site
        try:
            # Search all site-packages directories for nvidia DLLs
            for sp in site.getsitepackages() + [site.getusersitepackages()]:
                for subdir in ("nvidia/cudnn/bin", "nvidia/cublas/bin"):
                    dll_dir = Path(sp) / subdir
                    if dll_dir.is_dir():
                        os.add_dll_directory(str(dll_dir))
        except (OSError, AttributeError, TypeError):
            pass

    def _get_ocr(self):
        """Lazy-initialize PaddleOCR instance."""
        if self._ocr is None:
            self._setup_dll_paths()
            from paddleocr import PaddleOCR
            kwargs = {
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
                "use_gpu": self.use_gpu,
                "show_log": self.show_log,
            }
            if self.det_model_dir:
                kwargs["det_model_dir"] = self.det_model_dir
            if self.rec_model_dir:
                kwargs["rec_model_dir"] = self.rec_model_dir
            self._ocr = PaddleOCR(**kwargs)
        return self._ocr

    def _get_table_engine(self):
        """PPStructure is disabled to avoid GPU OOM.

        PPStructure loads layout detection, table structure recognition, and
        formula recognition models alongside OCR, which exhausts VRAM.
        Table detection is handled by OpenCV instead.
        """
        return None

    def ocr_image(self, image: np.ndarray) -> OCRPageResult:
        """Run OCR on a single image array (BGR or grayscale).

        Returns an OCRPageResult with lines, confidence scores, and bboxes.
        """
        ocr = self._get_ocr()

        # PaddleOCR expects BGR or grayscale numpy array
        if len(image.shape) == 2:
            # Grayscale -- convert to BGR for PaddleOCR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        results = ocr.ocr(image, cls=self.use_angle_cls)

        if not results or not results[0]:
            return OCRPageResult()

        lines: List[OCRLine] = []
        all_confidences: List[float] = []

        for line_data in results[0]:
            polygon = line_data[0]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = line_data[1][0]
            confidence = float(line_data[1][1])

            # Convert polygon to axis-aligned bbox
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            word = OCRWord(
                text=text,
                confidence=confidence,
                bbox=bbox,
                polygon=polygon,
            )

            line = OCRLine(
                text=text,
                confidence=confidence,
                bbox=bbox,
                words=[word],
            )
            lines.append(line)
            all_confidences.append(confidence)

        # Compute page-level stats
        word_count = len(all_confidences)
        avg_confidence = (
            sum(all_confidences) / word_count if word_count > 0 else 0.0
        )
        high_conf = sum(1 for c in all_confidences if c >= 0.95)
        flagged = sum(1 for c in all_confidences if c < 0.85)

        return OCRPageResult(
            lines=lines,
            page_confidence=avg_confidence,
            word_count=word_count,
            high_confidence_words=high_conf,
            flagged_words=flagged,
        )

    def detect_table_structure(self, image: np.ndarray) -> List[OCRTable]:
        """Detect and recognize tables in an image.

        Uses PaddleOCR's PPStructure for table structure recognition.
        Returns list of OCRTable with cell data.
        """
        engine = self._get_table_engine()
        if engine is None:
            return []

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        try:
            results = engine(image)
        except Exception as e:
            logger.warning("Table structure detection failed: %s", e)
            return []

        tables: List[OCRTable] = []
        for region in results:
            if region.get("type") != "table":
                continue

            table_bbox = region.get("bbox", [0, 0, 0, 0])
            html = region.get("res", {}).get("html", "")

            table = OCRTable(
                bbox=list(table_bbox),
                html=html,
            )

            # Parse cells from the structure result if available
            cells_data = region.get("res", {}).get("cells", [])
            if cells_data:
                max_row = 0
                max_col = 0
                for cd in cells_data:
                    row = cd.get("row", [0, 0])
                    col = cd.get("col", [0, 0])
                    cell = OCRTableCell(
                        row=row[0] if isinstance(row, list) else row,
                        col=col[0] if isinstance(col, list) else col,
                        text=cd.get("text", ""),
                        bbox=cd.get("bbox", []),
                    )
                    table.cells.append(cell)
                    max_row = max(max_row, cell.row + 1)
                    max_col = max(max_col, cell.col + 1)
                table.num_rows = max_row
                table.num_cols = max_col

            tables.append(table)

        return tables


# -- Subprocess execution ----------------------------------------------------

# This script can be invoked as a subprocess:
#   python -m src.paddle_ocr_engine --images img1.png img2.png --config config.json --output results.json

def _run_ocr_worker(
    image_paths: List[str],
    config: Dict[str, Any],
    output_path: str,
) -> None:
    """Worker function: OCR a list of images, write results to JSON."""
    engine = PaddleOCREngine(config)
    results = {}

    for img_path in image_paths:
        try:
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Could not read image: %s", img_path)
                results[img_path] = OCRPageResult().to_dict()
                continue
            ocr_result = engine.ocr_image(img)
            # Also try table detection
            tables = engine.detect_table_structure(img)
            ocr_result.tables = tables
            results[img_path] = ocr_result.to_dict()
        except Exception as e:
            logger.warning("OCR failed on %s: %s", img_path, e)
            results[img_path] = OCRPageResult().to_dict()

    Path(output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")


def run_paddleocr_subprocess(
    image_paths: List[str],
    config: Dict[str, Any],
) -> Dict[str, OCRPageResult]:
    """Run PaddleOCR in a subprocess for clean GPU memory release.

    Spawns a separate Python process that loads PaddleOCR, processes
    all images, writes results to a temp JSON file, and exits.
    When the process ends, all GPU memory is released.

    Args:
        image_paths: List of image file paths to OCR
        config: Pipeline configuration dict

    Returns:
        Dict mapping image_path to OCRPageResult
    """
    if not image_paths:
        return {}

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="paddle_cfg_"
    ) as cfg_file:
        json.dump(config, cfg_file)
        cfg_path = cfg_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, prefix="paddle_out_"
    ) as out_file:
        out_path = out_file.name

    # Build subprocess command
    cmd = [
        sys.executable, "-m", "src.paddle_ocr_engine",
        "--images", *image_paths,
        "--config", cfg_path,
        "--output", out_path,
    ]

    logger.info(
        "Launching PaddleOCR subprocess for %d images...", len(image_paths)
    )

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
        if proc.returncode != 0:
            logger.error("PaddleOCR subprocess failed: %s", proc.stderr[:500])
            return {}

        # Read results
        raw = json.loads(Path(out_path).read_text(encoding="utf-8"))
    except subprocess.TimeoutExpired:
        logger.error("PaddleOCR subprocess timed out")
        return {}
    except Exception as e:
        logger.error("PaddleOCR subprocess error: %s", e)
        return {}
    finally:
        # Clean up temp files
        for p in (cfg_path, out_path):
            try:
                Path(p).unlink()
            except OSError:
                pass

    # Parse results back into OCRPageResult objects
    results: Dict[str, OCRPageResult] = {}
    for img_path, data in raw.items():
        lines = []
        for ld in data.get("lines", []):
            words = [
                OCRWord(**wd) for wd in ld.get("words", [])
            ]
            lines.append(OCRLine(
                text=ld["text"],
                confidence=ld["confidence"],
                bbox=ld["bbox"],
                words=words,
            ))

        tables = []
        for td in data.get("tables", []):
            cells = [OCRTableCell(**cd) for cd in td.get("cells", [])]
            tables.append(OCRTable(
                bbox=td["bbox"],
                num_rows=td.get("num_rows", 0),
                num_cols=td.get("num_cols", 0),
                cells=cells,
                html=td.get("html", ""),
            ))

        results[img_path] = OCRPageResult(
            lines=lines,
            tables=tables,
            page_confidence=data.get("page_confidence", 0.0),
            word_count=data.get("word_count", 0),
            high_confidence_words=data.get("high_confidence_words", 0),
            flagged_words=data.get("flagged_words", 0),
        )

    logger.info("PaddleOCR subprocess complete: %d images processed", len(results))
    return results


# -- CLI entry point for subprocess execution ---------------------------------

def main() -> None:
    """CLI entry point when run as subprocess."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _run_ocr_worker(args.images, config, args.output)


if __name__ == "__main__":
    main()
