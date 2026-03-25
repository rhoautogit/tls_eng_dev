"""Image intelligence module — OCR, captioning, and embedding for extracted images.

Provides three levels of image searchability for the RAG pipeline:
  Level 1: Tesseract OCR   — extract text baked into raster images
  Level 2: BLIP-2 caption  — generate natural-language descriptions
  Level 3: CLIP embedding  — vector embeddings for similarity search

All models run locally (no external API calls).  GPU is used when available.
"""
from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Tesseract binary path (Windows default)
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ── Lazy model singletons ────────────────────────────────────────────────────
# Models are loaded once on first use, then cached for the session.

_tesseract_ready: Optional[bool] = None
_blip_model = None
_blip_processor = None
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None
_device = None


def _get_device():
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Image intelligence using device: %s", _device)
    return _device


def _ensure_tesseract():
    """Set up pytesseract with the correct binary path."""
    global _tesseract_ready
    if _tesseract_ready is not None:
        return _tesseract_ready
    try:
        import pytesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        # Quick test
        pytesseract.get_tesseract_version()
        _tesseract_ready = True
        logger.info("Tesseract OCR ready (v%s)", pytesseract.get_tesseract_version())
    except Exception as e:
        logger.warning("Tesseract not available: %s", e)
        _tesseract_ready = False
    return _tesseract_ready


def _ensure_blip():
    """Load BLIP-2 captioning model (first call only)."""
    global _blip_model, _blip_processor
    if _blip_model is not None:
        return True
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
        import torch

        model_name = "Salesforce/blip-image-captioning-base"
        logger.info("Loading BLIP captioning model (%s)...", model_name)
        _blip_processor = BlipProcessor.from_pretrained(model_name)
        _blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
        _blip_model = _blip_model.to(_get_device())
        _blip_model.eval()
        logger.info("BLIP model loaded on %s", _get_device())
        return True
    except Exception as e:
        logger.warning("BLIP model failed to load: %s", e)
        return False


def _ensure_clip():
    """Load CLIP model for image embeddings (first call only)."""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return True
    try:
        import open_clip

        model_name = "ViT-B-32"
        pretrained = "laion2b_s34b_b79k"
        logger.info("Loading CLIP model (%s/%s)...", model_name, pretrained)
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=_get_device()
        )
        _clip_tokenizer = open_clip.get_tokenizer(model_name)
        _clip_model.eval()
        logger.info("CLIP model loaded on %s", _get_device())
        return True
    except Exception as e:
        logger.warning("CLIP model failed to load: %s", e)
        return False


# ── Level 1: Tesseract OCR ───────────────────────────────────────────────────

def ocr_image(image: Image.Image) -> str:
    """Extract text from an image using Tesseract OCR.

    Returns the OCR'd text string (may be empty for non-text images).
    """
    if not _ensure_tesseract():
        return ""
    import pytesseract
    try:
        text = pytesseract.image_to_string(image, config="--psm 6")
        return text.strip()
    except Exception as e:
        logger.debug("OCR failed: %s", e)
        return ""


# ── Level 2: BLIP-2 Captioning ──────────────────────────────────────────────

def caption_image(image: Image.Image) -> str:
    """Generate a natural-language caption for an image using BLIP.

    Returns a descriptive string like 'a product photo of an LED luminaire'.
    """
    if not _ensure_blip():
        return ""
    import torch
    try:
        inputs = _blip_processor(images=image, return_tensors="pt").to(_get_device())
        with torch.no_grad():
            output = _blip_model.generate(**inputs, max_new_tokens=80)
        caption = _blip_processor.decode(output[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        logger.debug("Captioning failed: %s", e)
        return ""


# ── Level 3: CLIP Embedding ─────────────────────────────────────────────────

def embed_image(image: Image.Image) -> Optional[List[float]]:
    """Generate a CLIP embedding vector for an image.

    Returns a list of floats (512-dim) or None on failure.
    """
    if not _ensure_clip():
        return None
    import torch
    try:
        img_tensor = _clip_preprocess(image).unsqueeze(0).to(_get_device())
        with torch.no_grad():
            features = _clip_model.encode_image(img_tensor)
            features /= features.norm(dim=-1, keepdim=True)
        return features[0].cpu().tolist()
    except Exception as e:
        logger.debug("CLIP embedding failed: %s", e)
        return None


# ── Unified processing ───────────────────────────────────────────────────────

def process_image(
    image_bytes: bytes,
    run_ocr: bool = True,
    run_caption: bool = True,
    run_embedding: bool = True,
) -> Dict[str, Any]:
    """Run all requested intelligence levels on a single image.

    Args:
        image_bytes: Raw image file bytes (PNG, JPEG, etc.)
        run_ocr: Whether to run Tesseract OCR
        run_caption: Whether to run BLIP captioning
        run_embedding: Whether to run CLIP embedding

    Returns:
        Dict with keys: ocr_text, caption, embedding (each may be empty/None)
    """
    result: Dict[str, Any] = {
        "ocr_text": "",
        "caption": "",
        "embedding": None,
    }

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.debug("Could not open image: %s", e)
        return result

    # Skip very small images (likely icons/bullets, not meaningful content)
    if img.width < 20 or img.height < 20:
        return result

    if run_ocr:
        result["ocr_text"] = ocr_image(img)

    if run_caption:
        result["caption"] = caption_image(img)

    if run_embedding:
        result["embedding"] = embed_image(img)

    return result


def process_image_file(
    image_path: Path,
    run_ocr: bool = True,
    run_caption: bool = True,
    run_embedding: bool = True,
) -> Dict[str, Any]:
    """Convenience wrapper that reads from a file path."""
    try:
        image_bytes = image_path.read_bytes()
        return process_image(image_bytes, run_ocr, run_caption, run_embedding)
    except Exception as e:
        logger.debug("Could not read image file %s: %s", image_path, e)
        return {"ocr_text": "", "caption": "", "embedding": None}
