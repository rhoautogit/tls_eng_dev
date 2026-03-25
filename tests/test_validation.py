"""Tests for the 5-layer ValidationEngine (Stage 3)."""
from __future__ import annotations

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
    PageType,
    Table,
    TextBlock,
    ValidationResult,
)
from src.validation.validation_engine import ValidationEngine


# --- Config helper ---

def _cfg(
    w_cov=0.30, w_acc=0.25, w_comp=0.15, w_struct=0.15, w_cross=0.15,
    pass_thresh=0.95,
):
    return {
        "validation": {
            "pass_threshold": pass_thresh,
            "weights": {
                "coverage": w_cov,
                "accuracy": w_acc,
                "completeness": w_comp,
                "structural": w_struct,
                "cross_validation": w_cross,
            },
        },
        "coverage": {
            "header_margin_pct": 0.08,
            "footer_margin_pct": 0.08,
            "min_text_length": 3,
            "page_number_patterns": [r"^\d+$"],
        },
    }


def _make_extraction(
    text_blocks=None, tables=None, page_type=PageType.DIGITAL,
    page_width=612, page_height=792, confidence_gate=None,
):
    return PageExtractionResult(
        page_num=0,
        text_blocks=text_blocks or [],
        tables=tables or [],
        images=[],
        source="test",
        page_width=page_width,
        page_height=page_height,
        page_type=page_type,
        confidence_gate=confidence_gate,
    )


# --- ValidationResult structure ---

class TestValidationResult:
    def test_dataclass_fields(self):
        vr = ValidationResult(
            coverage_score=0.95,
            accuracy_score=0.90,
            completeness_score=0.85,
            structural_score=1.0,
            cross_validation_score=0.92,
            composite_score=0.92,
            passed=False,
        )
        assert vr.coverage_score == 0.95
        assert vr.passed is False


# --- V4: Structural checks ---

class TestStructuralValidation:
    def test_empty_extraction_returns_one(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()
        score = engine._v4_structural(extraction)
        assert score == 1.0

    def test_consistent_table_scores_high(self):
        engine = ValidationEngine(_cfg())
        table = Table(
            data=[["a", "b", "c"], ["d", "e", "f"], ["g", "h", "i"]],
            bbox=BoundingBox(10, 10, 300, 200),
            page_num=0,
        )
        extraction = _make_extraction(tables=[table])
        score = engine._v4_structural(extraction)
        assert score >= 0.9

    def test_inconsistent_table_scores_lower(self):
        engine = ValidationEngine(_cfg())
        table = Table(
            data=[["a", "b", "c"], ["d"], ["e", "f"]],
            bbox=BoundingBox(10, 10, 300, 200),
            page_num=0,
        )
        extraction = _make_extraction(tables=[table])
        score = engine._v4_structural(extraction)
        assert score < 1.0

    def test_overlapping_text_blocks_penalized(self):
        engine = ValidationEngine(_cfg())
        blocks = [
            TextBlock(text="A", bbox=BoundingBox(0, 0, 100, 50), page_num=0),
            TextBlock(text="B", bbox=BoundingBox(10, 5, 95, 45), page_num=0),
        ]
        extraction = _make_extraction(text_blocks=blocks)
        score = engine._v4_structural(extraction)
        assert score < 1.0

    def test_non_overlapping_blocks_not_penalized(self):
        engine = ValidationEngine(_cfg())
        blocks = [
            TextBlock(text="A", bbox=BoundingBox(0, 0, 100, 50), page_num=0),
            TextBlock(text="B", bbox=BoundingBox(0, 100, 100, 150), page_num=0),
        ]
        extraction = _make_extraction(text_blocks=blocks)
        score = engine._v4_structural(extraction)
        assert score >= 0.9


# --- V2: Accuracy ---

class TestAccuracyValidation:
    def test_ocr_confidence_used_for_scanned(self):
        engine = ValidationEngine(_cfg())
        gate = ConfidenceGate(
            ocr_confidence=0.88,
            level=ConfidenceLevel.MEDIUM,
            needs_qwen_vl=True,
            needs_human_review=False,
            word_count=100,
            high_confidence_words=80,
            flagged_words=20,
        )
        extraction = _make_extraction(
            page_type=PageType.SCANNED,
            confidence_gate=gate,
        )
        score = engine._accuracy_from_ocr_confidence(extraction)
        assert score == pytest.approx(0.88)

    def test_ocr_no_gate_falls_back_to_block_confidence(self):
        engine = ValidationEngine(_cfg())
        blocks = [
            TextBlock(text="A", bbox=BoundingBox(0, 0, 100, 50), page_num=0, confidence=0.7),
            TextBlock(text="B", bbox=BoundingBox(0, 60, 100, 100), page_num=0, confidence=0.9),
        ]
        extraction = _make_extraction(
            text_blocks=blocks,
            page_type=PageType.SCANNED,
        )
        score = engine._accuracy_from_ocr_confidence(extraction)
        assert score == pytest.approx(0.8)

    def test_digital_accuracy_perfect_match(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction(
            text_blocks=[
                TextBlock(
                    text="hello world test",
                    bbox=BoundingBox(10, 50, 200, 70),
                    page_num=0,
                ),
            ]
        )
        # Mock the baseline to return same text
        with patch(
            "src.validation.validation_engine.extract_calibrated_baseline",
            return_value="hello world test",
        ):
            score = engine._accuracy_from_text_match("fake.pdf", 0, extraction)

        assert score == pytest.approx(1.0)

    def test_digital_accuracy_empty_baseline(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()
        with patch(
            "src.validation.validation_engine.extract_calibrated_baseline",
            return_value="",
        ):
            score = engine._accuracy_from_text_match("fake.pdf", 0, extraction)
        assert score == 1.0  # nothing to validate


# --- Composite score ---

class TestCompositeScore:
    def test_all_perfect_scores_pass(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()

        with patch.object(engine, "_v1_coverage", return_value=1.0), \
             patch.object(engine, "_v2_accuracy", return_value=1.0), \
             patch.object(engine, "_v3_completeness", return_value=1.0), \
             patch.object(engine, "_v4_structural", return_value=1.0):
            result = engine.validate("fake.pdf", 0, extraction)

        assert result.composite_score == pytest.approx(1.0)
        assert result.passed is True

    def test_low_scores_fail(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()

        with patch.object(engine, "_v1_coverage", return_value=0.5), \
             patch.object(engine, "_v2_accuracy", return_value=0.5), \
             patch.object(engine, "_v3_completeness", return_value=0.5), \
             patch.object(engine, "_v4_structural", return_value=0.5):
            result = engine.validate("fake.pdf", 0, extraction)

        assert result.composite_score < 0.95
        assert result.passed is False

    def test_weights_sum_correctly(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()

        # With all scores at 0.8, composite should be 0.8
        with patch.object(engine, "_v1_coverage", return_value=0.8), \
             patch.object(engine, "_v2_accuracy", return_value=0.8), \
             patch.object(engine, "_v3_completeness", return_value=0.8), \
             patch.object(engine, "_v4_structural", return_value=0.8):
            result = engine.validate("fake.pdf", 0, extraction)

        assert result.composite_score == pytest.approx(0.8, abs=0.01)

    def test_custom_pass_threshold(self):
        engine = ValidationEngine(_cfg(pass_thresh=0.50))
        extraction = _make_extraction()

        with patch.object(engine, "_v1_coverage", return_value=0.6), \
             patch.object(engine, "_v2_accuracy", return_value=0.6), \
             patch.object(engine, "_v3_completeness", return_value=0.6), \
             patch.object(engine, "_v4_structural", return_value=0.6):
            result = engine.validate("fake.pdf", 0, extraction)

        assert result.passed is True  # 0.6 > 0.50


# --- Flagged items ---

class TestFlaggedItems:
    def test_low_accuracy_flagged(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()
        validation = ValidationResult(
            coverage_score=1.0,
            accuracy_score=0.80,
            completeness_score=1.0,
            structural_score=1.0,
            cross_validation_score=0.95,
            composite_score=0.95,
            passed=True,
        )
        flagged = engine.get_flagged_items("fake.pdf", 0, extraction, validation)
        types = [f["type"] for f in flagged]
        assert "low_accuracy" in types

    def test_high_scores_no_flags(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()
        validation = ValidationResult(
            coverage_score=1.0,
            accuracy_score=0.98,
            completeness_score=0.99,
            structural_score=1.0,
            cross_validation_score=0.99,
            composite_score=0.99,
            passed=True,
        )
        flagged = engine.get_flagged_items("fake.pdf", 0, extraction, validation)
        assert len(flagged) == 0

    def test_low_completeness_flagged(self):
        engine = ValidationEngine(_cfg())
        extraction = _make_extraction()
        validation = ValidationResult(
            coverage_score=1.0,
            accuracy_score=0.98,
            completeness_score=0.75,
            structural_score=1.0,
            cross_validation_score=0.95,
            composite_score=0.92,
            passed=False,
        )
        flagged = engine.get_flagged_items("fake.pdf", 0, extraction, validation)
        types = [f["type"] for f in flagged]
        assert "missing_content" in types
        # Should be high severity since < 0.80
        severity = next(f["severity"] for f in flagged if f["type"] == "missing_content")
        assert severity == "high"

    def test_ocr_confidence_flag(self):
        engine = ValidationEngine(_cfg())
        gate = ConfidenceGate(
            ocr_confidence=0.82,
            level=ConfidenceLevel.LOW,
            needs_qwen_vl=True,
            needs_human_review=True,
            word_count=100,
            flagged_words=30,
        )
        extraction = _make_extraction(
            page_type=PageType.SCANNED,
            confidence_gate=gate,
        )
        validation = ValidationResult(
            coverage_score=0.9,
            accuracy_score=0.82,
            completeness_score=0.9,
            structural_score=0.9,
            cross_validation_score=0.88,
            composite_score=0.88,
            passed=False,
        )
        flagged = engine.get_flagged_items("fake.pdf", 0, extraction, validation)
        types = [f["type"] for f in flagged]
        assert "low_ocr_confidence" in types
