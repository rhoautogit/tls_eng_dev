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
    timeout: int = 120,
) -> str:
    """Call Ollama's generate API with an image and prompt.

    Returns the model's text response.
    """
    import requests

    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [image_base64],
        "stream": False,
        "options": {},
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


# ── Prompt templates ─────────────────────────────────────────────────────────

VERIFICATION_PROMPT = """You are verifying OCR-extracted text against the original scanned document image.

Below is the text that was extracted from this page using OCR. Compare it carefully against what you see in the image.

=== EXTRACTED OCR TEXT ===
{ocr_text}
=== END OCR TEXT ===

Analyze the image and the extracted text. Respond in the following JSON format ONLY (no other text):

{{
  "is_accurate": true/false,
  "confidence": 0.0 to 1.0,
  "corrections": [
    {{
      "original": "incorrect text from OCR",
      "corrected": "what it should be",
      "location": "description of where on the page",
      "confidence": 0.0 to 1.0
    }}
  ],
  "missing_text": ["any text visible in the image but missing from OCR"],
  "summary": "brief summary of verification findings"
}}

Rules:
- Only flag real errors, not formatting differences
- For numbers and technical values, be especially precise
- If the OCR text is accurate, set is_accurate to true and leave corrections empty
- Include missing_text only for significant content (not headers/footers/page numbers)
- Be concise in the summary"""


TABLE_VERIFICATION_PROMPT = """You are verifying OCR-extracted table data against the original scanned document image.

This page contains tables. Below is what was extracted. Compare against the image carefully, especially:
- Cell values (numbers, text)
- Row/column alignment
- Missing rows or columns
- Merged cell handling

=== EXTRACTED TABLE DATA ===
{table_text}
=== END TABLE DATA ===

=== EXTRACTED TEXT ===
{ocr_text}
=== END TEXT ===

Respond in JSON format ONLY:

{{
  "is_accurate": true/false,
  "confidence": 0.0 to 1.0,
  "corrections": [
    {{
      "original": "incorrect value",
      "corrected": "correct value",
      "location": "table N, row R, column C",
      "confidence": 0.0 to 1.0
    }}
  ],
  "missing_text": ["any content visible but not extracted"],
  "summary": "brief summary"
}}"""


# ── Response parser ──────────────────────────────────────────────────────────

def _parse_verification_response(raw: str) -> VerificationResponse:
    """Parse Qwen-VL's JSON response into a VerificationResponse.

    Handles common LLM quirks: markdown code blocks, trailing commas,
    partial JSON, etc.
    """
    result = VerificationResponse(raw_response=raw)

    # Strip markdown code blocks if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        # Remove ```json or ``` wrapper
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    # Try to parse JSON
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract JSON from mixed text
        json_match = re.search(r'\{[\s\S]*\}', cleaned)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError:
                logger.warning("Could not parse Qwen-VL response as JSON")
                result.summary = f"Unparseable response: {raw[:200]}"
                return result
        else:
            logger.warning("No JSON found in Qwen-VL response")
            result.summary = f"No JSON in response: {raw[:200]}"
            return result

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
