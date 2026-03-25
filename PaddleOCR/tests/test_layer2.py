"""Tests for Layer 2: Coverage Scorer."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer2.coverage_scorer import (
    _count_meaningful_chars,
    calculate_coverage,
)
from src.models import (
    BoundingBox,
    ImageElement,
    PageExtractionResult,
    Table,
    TextBlock,
)


# ─── Character counting ───────────────────────────────────────────────────────

class TestCountMeaningfulChars:
    def test_empty(self):
        assert _count_meaningful_chars("") == 0

    def test_whitespace_only(self):
        assert _count_meaningful_chars("   \n\t  ") == 0

    def test_normal_text(self):
        assert _count_meaningful_chars("hello world") == 10

    def test_mixed(self):
        assert _count_meaningful_chars("  a b c  ") == 3


# ─── Coverage calculation ─────────────────────────────────────────────────────

def _make_result(text="", table_data=None):
    blocks = [TextBlock(text=text, bbox=BoundingBox(0, 0, 100, 20), page_num=0)] if text else []
    tables = []
    if table_data:
        tables = [
            Table(data=table_data, bbox=BoundingBox(0, 30, 100, 100), page_num=0)
        ]
    return PageExtractionResult(
        page_num=0,
        text_blocks=blocks,
        tables=tables,
        images=[],
        source="test",
        page_width=600,
        page_height=800,
    )


class TestCalculateCoverage:
    def test_empty_baseline_returns_one(self):
        result = _make_result()
        score = calculate_coverage(result, "")
        assert score == 1.0

    def test_full_coverage(self):
        baseline = "hello world foo bar"
        result = _make_result(text="hello world foo bar")
        score = calculate_coverage(result, baseline)
        assert score == pytest.approx(1.0)

    def test_partial_coverage(self):
        baseline = "abcdefghij"   # 10 chars
        result = _make_result(text="abcde")   # 5 chars
        score = calculate_coverage(result, baseline)
        assert score == pytest.approx(0.5)

    def test_over_extraction_capped_at_one(self):
        baseline = "abc"
        result = _make_result(text="abcdefghijklmno")
        score = calculate_coverage(result, baseline)
        assert score == pytest.approx(1.0)

    def test_table_content_counted(self):
        baseline = "name value unit"  # 13 chars
        result = _make_result(table_data=[["name", "value", "unit"]])
        score = calculate_coverage(result, baseline)
        assert score >= 0.9

    def test_threshold_95(self):
        baseline = "a" * 100
        result = _make_result(text="a" * 95)
        score = calculate_coverage(result, baseline)
        assert score >= 0.95


# ─── CoverageScorer integration (mocked fitz) ─────────────────────────────────

class TestCoverageScorerMocked:
    def _config(self):
        return {
            "coverage": {
                "header_margin_pct": 0.08,
                "footer_margin_pct": 0.08,
                "min_text_length": 3,
                "page_number_patterns": [r"^\d+$"],
            }
        }

    def test_score_page_calls_fitz(self):
        from src.layer2.coverage_scorer import CoverageScorer

        mock_block = (10, 50, 200, 70, "Engineering calculation data", 0, 0)
        mock_page = MagicMock()
        mock_page.rect.height = 800
        mock_page.get_text.return_value = [mock_block]

        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        result = _make_result(text="Engineering calculation data")

        with patch("fitz.open", return_value=mock_doc):
            scorer = CoverageScorer(self._config())
            score = scorer.score_page("fake.pdf", 0, result)

        assert 0.0 <= score <= 1.0

    def test_baseline_filters_page_number(self):
        from src.layer2.coverage_scorer import extract_calibrated_baseline

        # block at top (header region): y0=5, y1=15 → header_cutoff=64 → filtered
        # block in body: text="42" → matches page-number pattern → filtered
        # block in body: text="actual content" → kept
        blocks = [
            (10, 5, 200, 15, "Company Header", 0, 0),    # header region y0<64
            (10, 100, 200, 120, "42", 1, 0),              # page number
            (10, 200, 200, 220, "actual content", 2, 0),  # kept
        ]

        mock_page = MagicMock()
        mock_page.rect.height = 800.0
        mock_page.get_text.return_value = blocks

        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            baseline = extract_calibrated_baseline(
                "fake.pdf", 0, self._config()
            )

        assert "actual content" in baseline
        assert "Company Header" not in baseline
        assert "42" not in baseline
