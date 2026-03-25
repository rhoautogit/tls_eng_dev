"""Image intelligence module -- placeholder for Phase 2.

In the PaddleOCR pipeline, image intelligence (captioning, embedding) is
deferred to Phase 2 where Qwen-VL handles it through the RAG system.

Phase 1 uses PaddleOCR for text extraction from scanned pages and
Qwen-VL (via Ollama) for verification/correction of low-confidence OCR.
There is no need for CLIP, BLIP-2, or Tesseract in this pipeline.

This module provides stub functions so that any code importing from
image_intelligence continues to work without errors.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def process_image(
    image_bytes: bytes,
    run_ocr: bool = False,
    run_caption: bool = False,
    run_embedding: bool = False,
) -> Dict[str, Any]:
    """Stub -- image intelligence deferred to Phase 2 (Qwen-VL via RAG)."""
    return {
        "ocr_text": "",
        "caption": "",
        "embedding": None,
    }


def process_image_file(
    image_path: Path,
    run_ocr: bool = False,
    run_caption: bool = False,
    run_embedding: bool = False,
) -> Dict[str, Any]:
    """Stub -- image intelligence deferred to Phase 2 (Qwen-VL via RAG)."""
    return {
        "ocr_text": "",
        "caption": "",
        "embedding": None,
    }
