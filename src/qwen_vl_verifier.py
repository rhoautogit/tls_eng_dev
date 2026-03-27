"""Qwen-VL visual model verifier for scanned page OCR output.

Calls Qwen2.5-VL (7B) locally via Ollama to verify and correct OCR
extraction on pages flagged by the confidence gate. Runs entirely
local -- no external API calls.

Usage:
  verifier = QwenVLVerifier(config)
  result = verifier.verify_page(page_image_path, ocr_text, page_num)

The verifier sends the rendered page image alongside the OCR-extracted
text and asks the model to:
  1. Confirm whether the OCR text is accurate
  2. Identify any missing or incorrect content
  3. Provide corrected text where needed

The response is parsed into a structured VerificationResult that the
pipeline uses to update extraction data and verification status.
"""
from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz
import numpy as np

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TextCorrection:
    """A single correction identified by Qwen-VL."""
    original: str
    corrected: str
    location: str = ""       # description of where on the page
    confidence: float = 0.0  # model's confidence in the correction


@dataclass
class VerificationResponse:
    """Structured response from Qwen-VL verification."""
    is_accurate: bool = False
    confidence: float = 0.0
    corrections: List[TextCorrection] = field(default_factory=list)
    missing_text: List[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""

    @property
    def has_corrections(self) -> bool:
        return len(self.corrections) > 0 or len(self.missing_text) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_accurate": self.is_accurate,
            "confidence": self.confidence,
            "corrections": [
                {
                    "original": c.original,
                    "corrected": c.corrected,
                    "location": c.location,
                    "confidence": c.confidence,
                }
                for c in self.corrections
            ],
            "missing_text": self.missing_text,
            "summary": self.summary,
        }


# ── Ollama client ────────────────────────────────────────────────────────────

def _encode_image_base64(image_path: str) -> str:
    """Read an image file and return its base64 encoding."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _render_page_to_png(pdf_path: str, page_num: int, dpi: int = 200) -> bytes:
    """Render a PDF page to PNG bytes for sending to Qwen-VL."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    png_bytes = pix.tobytes("png")
    doc.close()
    return png_bytes


def _call_ollama(
    model: str,
    prompt: str,
    image_base64: str,
    host: str = "http://localhost:11434",
    num_ctx: int = -1,
    timeout: int = 300,
    keep_alive: str = "10m",
) -> str:
    """Call Ollama's generate API with an image and prompt.

    Returns the model's text response.
    """
    import requests

    url = f"{host}/api/generate"
    # Qwen3-VL: prepend /no_think to suppress chain-of-thought reasoning,
    # which otherwise produces prose instead of JSON.
    formatted_prompt = f"/no_think\n{prompt}"

    payload = {
        "model": model,
        "system": (
            "You are an OCR verification assistant. "
            "Output ONLY a single JSON object. "
            "No markdown fences, no explanation, no text before or after the JSON."
        ),
        "prompt": formatted_prompt,
        "images": [image_base64],
        "stream": False,
        "format": "json",
        "options": {},
        "keep_alive": keep_alive,
    }

    if num_ctx != -1:
        payload["options"]["num_ctx"] = num_ctx

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")
    except requests.ConnectionError:
        logger.error(
            "Cannot connect to Ollama at %s. Is Ollama running?", host
        )
        raise
    except requests.Timeout:
        logger.error("Ollama request timed out after %ds", timeout)
        raise
    except Exception as e:
        logger.error("Ollama API error: %s", e)
        raise


def _unload_ollama_models(host: str = "http://localhost:11434") -> None:
    """Unload all models from Ollama to free GPU VRAM.

    Sends keep_alive=0 to each loaded model so Ollama releases VRAM.
    Call this before PaddleOCR to give it full GPU, or after PaddleOCR
    to prepare for reloading the verification model cleanly.
    """
    import requests

    try:
        resp = requests.get(f"{host}/api/ps", timeout=5)
        if resp.status_code != 200:
            return
        running = resp.json().get("models", [])
        for m in running:
            model_name = m.get("name", "")
            if model_name:
                logger.info("Unloading Ollama model: %s", model_name)
                requests.post(
                    f"{host}/api/generate",
                    json={"model": model_name, "keep_alive": 0},
                    timeout=30,
                )
        if running:
            logger.info("Unloaded %d Ollama model(s) to free VRAM", len(running))
    except Exception as e:
        logger.debug("Could not unload Ollama models: %s", e)


def _release_paddle_gpu() -> None:
    """Aggressively release PaddlePaddle GPU memory.

    Goes beyond release_shared() by clearing the CUDA cache and forcing
    garbage collection. This is needed on 8GB GPUs where PaddlePaddle's
    CUDA context can hold 1-2GB even after the engine is released.
    """
    import gc

    try:
        from .paddle_ocr_engine import PaddleOCREngine
        PaddleOCREngine.release_shared()
    except Exception:
        pass

    gc.collect()

    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
    except Exception:
        pass

    gc.collect()


def _warmup_ollama(
    model: str,
    host: str = "http://localhost:11434",
    timeout: int = 180,
) -> bool:
    """Pre-load the model into GPU memory by sending a minimal request.

    Ollama loads models lazily on first request. After PaddleOCR releases
    GPU memory, we call this so the model is fully loaded before the real
    verification loop starts. This avoids the cold-start cost eating into
    the per-page timeout.

    Returns True if warmup succeeded.
    """
    import requests

    # First, ensure no stale Ollama models are hogging VRAM
    _unload_ollama_models(host)

    # Aggressively free PaddlePaddle GPU memory
    _release_paddle_gpu()

    import time
    time.sleep(2)  # brief pause to let VRAM settle

    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": "Hi",
        "stream": False,
        "keep_alive": "10m",
    }

    try:
        logger.info("Warming up Qwen-VL model (%s), loading into GPU...", model)
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        logger.info("Qwen-VL model loaded and ready")
        return True
    except Exception as e:
        logger.warning("Qwen-VL warmup failed: %s", e)
        return False


# ── Prompt templates ─────────────────────────────────────────────────────────

VERIFICATION_PROMPT = """Compare the OCR-extracted text below against the scanned document image.

=== EXTRACTED OCR TEXT ===
{ocr_text}
=== END OCR TEXT ===

Respond with ONLY this JSON (no other text before or after):

{{
  "is_accurate": true,
  "confidence": 0.95,
  "corrections": [],
  "missing_text": [],
  "summary": "OCR text matches the image"
}}

If there are errors, set is_accurate to false and list corrections:

{{
  "is_accurate": false,
  "confidence": 0.85,
  "corrections": [
    {{
      "original": "incorret word",
      "corrected": "incorrect word",
      "location": "top of page",
      "confidence": 0.9
    }}
  ],
  "missing_text": ["any significant text visible in image but missing from OCR"],
  "summary": "Found 1 misspelling"
}}

Rules:
- Only flag real errors, not formatting differences
- For numbers and technical values, be especially precise
- Include missing_text only for significant content (not headers/footers/page numbers)
- Do not explain. Do not add any text outside the JSON."""


TABLE_VERIFICATION_PROMPT = """Compare the OCR-extracted table data below against the scanned document image.

=== EXTRACTED TABLE DATA ===
{table_text}
=== END TABLE DATA ===

=== EXTRACTED TEXT ===
{ocr_text}
=== END TEXT ===

Check carefully: cell values, row/column alignment, missing rows or columns, merged cells.

Respond with ONLY this JSON (no other text before or after):

{{
  "is_accurate": true,
  "confidence": 0.95,
  "corrections": [],
  "missing_text": [],
  "summary": "Table data matches the image"
}}

If there are errors, set is_accurate to false and list corrections:

{{
  "is_accurate": false,
  "confidence": 0.8,
  "corrections": [
    {{
      "original": "incorrect value",
      "corrected": "correct value",
      "location": "table 1, row 2, column 3",
      "confidence": 0.9
    }}
  ],
  "missing_text": ["any content visible but not extracted"],
  "summary": "Found 1 incorrect cell value"
}}

Do not explain. Do not add any text outside the JSON."""


# ── Response parser ──────────────────────────────────────────────────────────

def _fix_json_quirks(text: str) -> str:
    """Fix common LLM JSON quirks before parsing."""
    # Replace Python-style True/False/None with JSON equivalents
    text = re.sub(r'\bTrue\b', 'true', text)
    text = re.sub(r'\bFalse\b', 'false', text)
    text = re.sub(r'\bNone\b', 'null', text)
    # Remove trailing commas before } or ]
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return text


def _parse_prose_fallback(raw: str) -> VerificationResponse:
    """Extract verification info from a prose (non-JSON) response.

    When Qwen-VL ignores the JSON instruction and returns natural language,
    we still try to extract useful signals: does it say the text is accurate,
    and does it mention any specific corrections.
    """
    result = VerificationResponse(raw_response=raw)
    lower = raw.lower()

    # Detect accuracy signal from prose
    accurate_signals = [
        "accurately reflects", "correctly captures", "matches the image",
        "text is accurate", "ocr text is accurate", "correctly extracted",
        "no significant errors", "no errors", "fully match",
    ]
    inaccurate_signals = [
        "does not fully match", "does not match", "discrepancies",
        "incorrectly", "missing", "errors found", "not accurate",
        "significant errors", "key issues",
    ]

    accurate_hits = sum(1 for s in accurate_signals if s in lower)
    inaccurate_hits = sum(1 for s in inaccurate_signals if s in lower)

    if accurate_hits > inaccurate_hits:
        result.is_accurate = True
        result.confidence = 0.85
    elif inaccurate_hits > 0:
        result.is_accurate = False
        result.confidence = 0.6
    else:
        result.is_accurate = False
        result.confidence = 0.5

    result.summary = raw[:300]

    logger.info(
        "Prose fallback: is_accurate=%s (accurate_signals=%d, inaccurate_signals=%d)",
        result.is_accurate, accurate_hits, inaccurate_hits,
    )
    return result


def _parse_verification_response(raw: str) -> VerificationResponse:
    """Parse Qwen-VL's JSON response into a VerificationResponse.

    Handles common LLM quirks: markdown code blocks, trailing commas,
    Python-style booleans, mixed text around JSON, etc.
    """
    result = VerificationResponse(raw_response=raw)

    # Strip markdown code blocks if present
    cleaned = raw.strip()
    # Handle ```json ... ``` or ``` ... ``` wrappers
    fence_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)```', cleaned)
    if fence_match:
        cleaned = fence_match.group(1).strip()

    cleaned = _fix_json_quirks(cleaned)

    # Try to parse JSON directly
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract the outermost JSON object from mixed text
        # Use a non-greedy approach: find first { and last }
        first_brace = cleaned.find('{')
        last_brace = cleaned.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            candidate = cleaned[first_brace:last_brace + 1]
            candidate = _fix_json_quirks(candidate)
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                logger.warning("Could not parse Qwen-VL response as JSON, using prose fallback")
                return _parse_prose_fallback(raw)
        else:
            logger.warning("No JSON found in Qwen-VL response, using prose fallback")
            return _parse_prose_fallback(raw)

    result.is_accurate = bool(data.get("is_accurate", False))
    result.confidence = float(data.get("confidence", 0.0))
    result.summary = str(data.get("summary", ""))
    result.missing_text = data.get("missing_text", [])

    for corr in data.get("corrections", []):
        if isinstance(corr, dict):
            result.corrections.append(TextCorrection(
                original=str(corr.get("original", "")),
                corrected=str(corr.get("corrected", "")),
                location=str(corr.get("location", "")),
                confidence=float(corr.get("confidence", 0.0)),
            ))

    return result


# ── Verifier class ───────────────────────────────────────────────────────────

class QwenVLVerifier:
    """Verifies and corrects OCR output using Qwen2.5-VL via Ollama."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        qwen_cfg = config.get("scanned_path", {}).get("qwen_vl", {})

        self.enabled = bool(qwen_cfg.get("enabled", True))
        self.model = str(qwen_cfg.get("model", "qwen3-vl:8b"))
        self.host = str(qwen_cfg.get("ollama_host", "http://localhost:11434"))
        self.gpu_num_ctx = int(qwen_cfg.get("gpu_num_ctx", -1))
        self.cpu_num_ctx = int(qwen_cfg.get("cpu_num_ctx", 0))
        self.timeout = int(qwen_cfg.get("timeout", 300))
        self._render_dpi = 200

        # Detect GPU availability to choose num_ctx
        self._num_ctx = self._detect_num_ctx()

    def _detect_num_ctx(self) -> int:
        """Choose num_ctx based on whether Ollama is using GPU.

        Queries the Ollama API to check if the model is loaded with GPU
        layers, rather than importing torch (which we don't need in the
        PaddleOCR pipeline).
        """
        try:
            import requests
            resp = requests.get(f"{self.host}/api/ps", timeout=5)
            if resp.status_code == 200:
                running = resp.json().get("models", [])
                for m in running:
                    # If any model is using GPU layers, assume GPU is available
                    details = m.get("details", {})
                    size_vram = m.get("size_vram", 0)
                    if size_vram > 0:
                        logger.info(
                            "Ollama GPU detected (VRAM in use), "
                            "using gpu_num_ctx=%d", self.gpu_num_ctx,
                        )
                        return self.gpu_num_ctx
        except Exception:
            pass

        # Fallback: check if CUDA is available via a lightweight check
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                logger.info(
                    "GPU detected via nvidia-smi, using gpu_num_ctx=%d",
                    self.gpu_num_ctx,
                )
                return self.gpu_num_ctx
        except Exception:
            pass

        logger.info("No GPU detected, using cpu_num_ctx=%d", self.cpu_num_ctx)
        return self.cpu_num_ctx

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is available."""
        if not self.enabled:
            return False

        import requests
        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=5)
            if resp.status_code != 200:
                return False
            models = resp.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            # Check if our model is available (with or without :latest tag)
            available = any(
                self.model in name or name.startswith(self.model.split(":")[0])
                for name in model_names
            )
            if not available:
                logger.warning(
                    "Model %s not found in Ollama. Available: %s",
                    self.model, model_names,
                )
            return available
        except Exception as e:
            logger.debug("Ollama not reachable: %s", e)
            return False

    def verify_page(
        self,
        pdf_path: str,
        page_num: int,
        ocr_text: str,
        table_text: str = "",
    ) -> VerificationResponse:
        """Verify OCR extraction for a single page.

        Args:
            pdf_path: Path to the PDF file
            page_num: Zero-indexed page number
            ocr_text: The OCR-extracted text to verify
            table_text: Optional table data as text

        Returns:
            VerificationResponse with accuracy assessment and corrections
        """
        if not self.enabled:
            return VerificationResponse(
                summary="Qwen-VL verification disabled"
            )

        # Render page to PNG
        try:
            png_bytes = _render_page_to_png(
                pdf_path, page_num, dpi=self._render_dpi
            )
            image_b64 = base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            logger.error("Failed to render page %d for Qwen-VL: %s", page_num + 1, e)
            return VerificationResponse(
                summary=f"Page render failed: {e}"
            )

        # Choose prompt based on whether tables are present
        if table_text:
            prompt = TABLE_VERIFICATION_PROMPT.format(
                table_text=table_text,
                ocr_text=ocr_text,
            )
        else:
            prompt = VERIFICATION_PROMPT.format(ocr_text=ocr_text)

        # Call Ollama
        try:
            logger.info(
                "Page %d: sending to Qwen-VL (%s) for verification...",
                page_num + 1, self.model,
            )
            raw_response = _call_ollama(
                model=self.model,
                prompt=prompt,
                image_base64=image_b64,
                host=self.host,
                num_ctx=self._num_ctx,
                timeout=self.timeout,
            )
            logger.info(
                "Page %d: Qwen-VL response received (%d chars)",
                page_num + 1, len(raw_response),
            )
        except Exception as e:
            logger.error(
                "Page %d: Qwen-VL call failed: %s", page_num + 1, e,
            )
            return VerificationResponse(
                summary=f"Qwen-VL call failed: {e}"
            )

        # Parse response
        result = _parse_verification_response(raw_response)

        logger.info(
            "Page %d: Qwen-VL verdict: %s (confidence=%.1f%%) "
            "corrections=%d missing=%d",
            page_num + 1,
            "ACCURATE" if result.is_accurate else "CORRECTIONS NEEDED",
            result.confidence * 100,
            len(result.corrections),
            len(result.missing_text),
        )

        return result

    def verify_flagged_pages(
        self,
        pdf_path: str,
        page_results: list,
    ) -> Dict[int, VerificationResponse]:
        """Verify all flagged pages in a document.

        Args:
            pdf_path: Path to the PDF
            page_results: List of PageResult objects

        Returns:
            Dict mapping page_num to VerificationResponse
        """
        results: Dict[int, VerificationResponse] = {}

        if not self.enabled:
            logger.info("Qwen-VL verification is disabled")
            return results

        if not self.is_available():
            logger.warning(
                "Qwen-VL not available (Ollama not running or model not pulled). "
                "Flagged pages will remain unverified."
            )
            return results

        flagged_pages = [
            pr for pr in page_results
            if (pr.extraction.confidence_gate
                and pr.extraction.confidence_gate.needs_qwen_vl)
        ]

        if not flagged_pages:
            logger.info("No pages flagged for Qwen-VL verification")
            return results

        logger.info(
            "Running Qwen-VL verification on %d flagged pages...",
            len(flagged_pages),
        )

        # Warmup: pre-load model into GPU before the verification loop.
        # After PaddleOCR released GPU memory, Ollama needs to load the
        # model fresh. This avoids cold-start cost on the first real page.
        _warmup_ollama(self.model, self.host)

        for pr in flagged_pages:
            ocr_text = pr.extraction.all_text()

            # Format table data if present
            table_text = ""
            if pr.extraction.tables:
                table_parts = []
                for ti, table in enumerate(pr.extraction.tables, 1):
                    table_parts.append(f"Table {ti}:")
                    if table.headers:
                        table_parts.append(" | ".join(table.headers))
                        table_parts.append("-" * 40)
                    for row in table.data:
                        table_parts.append(" | ".join(str(c) for c in row))
                    table_parts.append("")
                table_text = "\n".join(table_parts)

            response = self.verify_page(
                pdf_path, pr.page_num, ocr_text, table_text
            )
            results[pr.page_num] = response

        logger.info(
            "Qwen-VL verification complete: %d/%d pages verified",
            len(results), len(flagged_pages),
        )

        return results


# ── Correction applicator ────────────────────────────────────────────────────

def apply_corrections(
    extraction: Any,  # PageExtractionResult
    verification: VerificationResponse,
) -> Tuple[Any, bool]:
    """Apply Qwen-VL corrections to the extraction result.

    Returns (updated_extraction, was_modified).
    """
    if not verification.has_corrections:
        return extraction, False

    modified = False

    # Apply text corrections
    for corr in verification.corrections:
        if not corr.original or not corr.corrected:
            continue
        for block in extraction.text_blocks:
            if corr.original in block.text:
                block.text = block.text.replace(corr.original, corr.corrected)
                block.source = f"{block.source}+qwen_corrected"
                modified = True
                logger.debug(
                    "Applied correction: '%s' -> '%s'",
                    corr.original, corr.corrected,
                )

        # Check tables too
        for table in extraction.tables:
            for ri, row in enumerate(table.data):
                for ci, cell in enumerate(row):
                    if isinstance(cell, str) and corr.original in cell:
                        table.data[ri][ci] = cell.replace(
                            corr.original, corr.corrected
                        )
                        modified = True

    # Add missing text as new text blocks (if any)
    if verification.missing_text:
        from .models import BoundingBox, TextBlock
        for missing in verification.missing_text:
            if missing.strip():
                extraction.text_blocks.append(TextBlock(
                    text=missing.strip(),
                    bbox=BoundingBox(0, 0, 1, 1),  # placeholder bbox
                    page_num=extraction.page_num,
                    confidence=verification.confidence,
                    source="qwen_vl_missing",
                ))
                modified = True
                logger.debug("Added missing text: '%s'", missing[:50])

    return extraction, modified
