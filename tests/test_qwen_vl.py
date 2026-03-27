"""Tests for the Qwen-VL verifier module."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BoundingBox,
    ConfidenceGate,
    ConfidenceLevel,
    PageExtractionResult,
    PageResult,
    PageType,
    TextBlock,
    Table,
    VerificationStatus,
)
from src.qwen_vl_verifier import (
    QwenVLVerifier,
    VerificationResponse,
    TextCorrection,
    _parse_verification_response,
    apply_corrections,
)


# --- Response parsing ---

class TestParseVerificationResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "is_accurate": True,
            "confidence": 0.95,
            "corrections": [],
            "missing_text": [],
            "summary": "Text matches image",
        })
        result = _parse_verification_response(raw)
        assert result.is_accurate is True
        assert result.confidence == 0.95
        assert result.corrections == []
        assert result.summary == "Text matches image"

    def test_json_with_corrections(self):
        raw = json.dumps({
            "is_accurate": False,
            "confidence": 0.80,
            "corrections": [
                {
                    "original": "teh",
                    "corrected": "the",
                    "location": "paragraph 2",
                    "confidence": 0.99,
                },
            ],
            "missing_text": ["Footer text was missed"],
            "summary": "One typo found",
        })
        result = _parse_verification_response(raw)
        assert result.is_accurate is False
        assert len(result.corrections) == 1
        assert result.corrections[0].original == "teh"
        assert result.corrections[0].corrected == "the"
        assert len(result.missing_text) == 1
        assert result.has_corrections is True

    def test_markdown_code_block_wrapper(self):
        raw = "```json\n" + json.dumps({
            "is_accurate": True,
            "confidence": 0.9,
            "corrections": [],
            "missing_text": [],
            "summary": "OK",
        }) + "\n```"
        result = _parse_verification_response(raw)
        assert result.is_accurate is True

    def test_mixed_text_with_json(self):
        raw = "Here is my analysis:\n" + json.dumps({
            "is_accurate": False,
            "confidence": 0.7,
            "corrections": [],
            "missing_text": [],
            "summary": "Issues found",
        }) + "\nEnd of analysis."
        result = _parse_verification_response(raw)
        assert result.is_accurate is False

    def test_unparseable_response(self):
        raw = "I cannot process this image."
        result = _parse_verification_response(raw)
        assert result.is_accurate is False
        assert result.summary  # prose fallback populates summary

    def test_empty_response(self):
        result = _parse_verification_response("")
        assert result.is_accurate is False


# --- VerificationResponse ---

class TestVerificationResponse:
    def test_has_corrections_false_when_empty(self):
        r = VerificationResponse()
        assert r.has_corrections is False

    def test_has_corrections_true_with_corrections(self):
        r = VerificationResponse(
            corrections=[TextCorrection(original="a", corrected="b")]
        )
        assert r.has_corrections is True

    def test_has_corrections_true_with_missing(self):
        r = VerificationResponse(missing_text=["some text"])
        assert r.has_corrections is True

    def test_to_dict(self):
        r = VerificationResponse(
            is_accurate=True,
            confidence=0.9,
            corrections=[
                TextCorrection(original="x", corrected="y", location="p1", confidence=0.8)
            ],
            missing_text=["z"],
            summary="test",
        )
        d = r.to_dict()
        assert d["is_accurate"] is True
        assert len(d["corrections"]) == 1
        assert d["corrections"][0]["original"] == "x"
        assert d["missing_text"] == ["z"]


# --- apply_corrections ---

class TestApplyCorrections:
    def _extraction(self, text="hello teh world", table_data=None):
        blocks = [
            TextBlock(
                text=text,
                bbox=BoundingBox(10, 10, 200, 30),
                page_num=0,
                source="paddleocr",
            )
        ]
        tables = []
        if table_data:
            tables = [
                Table(data=table_data, bbox=BoundingBox(10, 50, 200, 100), page_num=0)
            ]
        return PageExtractionResult(
            page_num=0,
            text_blocks=blocks,
            tables=tables,
            images=[],
            source="paddleocr",
            page_width=612,
            page_height=792,
            page_type=PageType.SCANNED,
        )

    def test_apply_text_correction(self):
        extraction = self._extraction("hello teh world")
        verification = VerificationResponse(
            corrections=[
                TextCorrection(original="teh", corrected="the")
            ]
        )
        updated, modified = apply_corrections(extraction, verification)
        assert modified is True
        assert "the" in updated.text_blocks[0].text
        assert "teh" not in updated.text_blocks[0].text

    def test_apply_table_correction(self):
        extraction = self._extraction(
            text="intro",
            table_data=[["Valuee", "100"]],
        )
        verification = VerificationResponse(
            corrections=[
                TextCorrection(original="Valuee", corrected="Value")
            ]
        )
        updated, modified = apply_corrections(extraction, verification)
        assert modified is True
        assert updated.tables[0].data[0][0] == "Value"

    def test_add_missing_text(self):
        extraction = self._extraction("some text")
        verification = VerificationResponse(
            missing_text=["Footer: Page 1 of 5"]
        )
        updated, modified = apply_corrections(extraction, verification)
        assert modified is True
        assert len(updated.text_blocks) == 2
        assert updated.text_blocks[-1].source == "qwen_vl_missing"

    def test_no_corrections_returns_unmodified(self):
        extraction = self._extraction("correct text")
        verification = VerificationResponse(is_accurate=True)
        updated, modified = apply_corrections(extraction, verification)
        assert modified is False

    def test_empty_correction_skipped(self):
        extraction = self._extraction("text")
        verification = VerificationResponse(
            corrections=[TextCorrection(original="", corrected="")]
        )
        updated, modified = apply_corrections(extraction, verification)
        assert modified is False


# --- QwenVLVerifier ---

class TestQwenVLVerifier:
    def _config(self, enabled=True):
        return {
            "scanned_path": {
                "qwen_vl": {
                    "enabled": enabled,
                    "model": "qwen2.5-vl:7b",
                    "ollama_host": "http://localhost:11434",
                    "gpu_num_ctx": -1,
                    "cpu_num_ctx": 0,
                },
            },
        }

    def test_disabled_returns_empty(self):
        verifier = QwenVLVerifier(self._config(enabled=False))
        result = verifier.verify_page("fake.pdf", 0, "some text")
        assert "disabled" in result.summary

    def test_verify_page_calls_ollama(self):
        verifier = QwenVLVerifier(self._config())

        mock_response = json.dumps({
            "is_accurate": True,
            "confidence": 0.95,
            "corrections": [],
            "missing_text": [],
            "summary": "All good",
        })

        with patch(
            "src.qwen_vl_verifier._render_page_to_png",
            return_value=b"\x89PNG fake",
        ), patch(
            "src.qwen_vl_verifier._call_ollama",
            return_value=mock_response,
        ):
            result = verifier.verify_page("fake.pdf", 0, "OCR text here")

        assert result.is_accurate is True
        assert result.confidence == 0.95

    def test_verify_page_with_tables(self):
        verifier = QwenVLVerifier(self._config())

        mock_response = json.dumps({
            "is_accurate": False,
            "confidence": 0.80,
            "corrections": [
                {"original": "100", "corrected": "1000", "location": "table 1", "confidence": 0.9},
            ],
            "missing_text": [],
            "summary": "Number error in table",
        })

        with patch(
            "src.qwen_vl_verifier._render_page_to_png",
            return_value=b"\x89PNG fake",
        ), patch(
            "src.qwen_vl_verifier._call_ollama",
            return_value=mock_response,
        ):
            result = verifier.verify_page(
                "fake.pdf", 0, "text", table_text="100 | 200"
            )

        assert result.is_accurate is False
        assert len(result.corrections) == 1
        assert result.corrections[0].corrected == "1000"

    def test_ollama_connection_error_handled(self):
        verifier = QwenVLVerifier(self._config())

        with patch(
            "src.qwen_vl_verifier._render_page_to_png",
            return_value=b"\x89PNG fake",
        ), patch(
            "src.qwen_vl_verifier._call_ollama",
            side_effect=ConnectionError("Ollama not running"),
        ):
            result = verifier.verify_page("fake.pdf", 0, "text")

        assert "failed" in result.summary.lower()

    def test_is_available_when_ollama_down(self):
        verifier = QwenVLVerifier(self._config())
        import requests
        with patch(
            "requests.get",
            side_effect=requests.ConnectionError("not running"),
        ):
            assert verifier.is_available() is False

    def test_is_available_model_not_found(self):
        verifier = QwenVLVerifier(self._config())
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "llama3:latest"}]
        }
        with patch("requests.get", return_value=mock_resp):
            assert verifier.is_available() is False


# --- verify_flagged_pages ---

class TestVerifyFlaggedPages:
    def _config(self):
        return {
            "scanned_path": {
                "qwen_vl": {
                    "enabled": True,
                    "model": "qwen2.5-vl:7b",
                    "ollama_host": "http://localhost:11434",
                    "gpu_num_ctx": -1,
                    "cpu_num_ctx": 0,
                },
            },
        }

    def _page_result(self, page_num, needs_qwen=True):
        gate = ConfidenceGate(
            ocr_confidence=0.80,
            level=ConfidenceLevel.LOW,
            needs_qwen_vl=needs_qwen,
            needs_human_review=True,
            word_count=50,
        ) if needs_qwen else None

        extraction = PageExtractionResult(
            page_num=page_num,
            text_blocks=[
                TextBlock(
                    text="Test OCR text",
                    bbox=BoundingBox(10, 10, 200, 30),
                    page_num=page_num,
                )
            ],
            tables=[],
            images=[],
            source="paddleocr",
            page_width=612,
            page_height=792,
            page_type=PageType.SCANNED,
            confidence_gate=gate,
        )

        return PageResult(
            page_num=page_num,
            final_score=0.80,
            initial_score=0.80,
            passed=False,
            extraction=extraction,
            run_records=[],
            gap_map_paths=[],
            status="unresolved",
        )

    def test_skips_unflagged_pages(self):
        verifier = QwenVLVerifier(self._config())
        pages = [
            self._page_result(0, needs_qwen=False),
            self._page_result(1, needs_qwen=False),
        ]

        # Should not call Ollama at all
        with patch.object(verifier, "is_available", return_value=True):
            results = verifier.verify_flagged_pages("fake.pdf", pages)
        assert len(results) == 0

    def test_processes_flagged_pages(self):
        verifier = QwenVLVerifier(self._config())
        pages = [
            self._page_result(0, needs_qwen=False),
            self._page_result(1, needs_qwen=True),
        ]

        mock_response = json.dumps({
            "is_accurate": True,
            "confidence": 0.92,
            "corrections": [],
            "missing_text": [],
            "summary": "Verified",
        })

        with patch.object(verifier, "is_available", return_value=True), \
             patch(
                 "src.qwen_vl_verifier._render_page_to_png",
                 return_value=b"\x89PNG",
             ), \
             patch(
                 "src.qwen_vl_verifier._call_ollama",
                 return_value=mock_response,
             ):
            results = verifier.verify_flagged_pages("fake.pdf", pages)

        assert 1 in results
        assert results[1].is_accurate is True
        assert 0 not in results

    def test_returns_empty_when_unavailable(self):
        verifier = QwenVLVerifier(self._config())
        pages = [self._page_result(0, needs_qwen=True)]

        with patch.object(verifier, "is_available", return_value=False):
            results = verifier.verify_flagged_pages("fake.pdf", pages)
        assert len(results) == 0
