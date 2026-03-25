"""Tests for PageClassifier (Stage 1: page type detection)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import BoundingBox, PageType
from src.page_classifier import PageClassifier


# --- Config helper ---

def _cfg(digital_thresh=100, scanned_thresh=20, img_coverage_thresh=0.4):
    return {
        "page_classification": {
            "digital_char_threshold": digital_thresh,
            "scanned_char_threshold": scanned_thresh,
            "image_coverage_threshold": img_coverage_thresh,
            "min_image_size": 50,
        },
        "extraction": {
            "merger": {
                "digital_threshold": digital_thresh,
                "scanned_threshold": scanned_thresh,
            }
        },
    }


# --- PageType determination ---

class TestPageTypeDetermination:
    def test_digital_high_chars_no_images(self):
        pc = PageClassifier(_cfg())
        assert pc._determine_type(500, 0, 0.0) == PageType.DIGITAL

    def test_scanned_low_chars_high_image(self):
        pc = PageClassifier(_cfg())
        assert pc._determine_type(5, 1, 0.8) == PageType.SCANNED

    def test_hybrid_high_chars_high_image(self):
        pc = PageClassifier(_cfg())
        assert pc._determine_type(500, 2, 0.5) == PageType.HYBRID

    def test_scanned_zero_chars(self):
        pc = PageClassifier(_cfg())
        assert pc._determine_type(0, 0, 0.0) == PageType.SCANNED

    def test_digital_moderate_chars_low_image(self):
        pc = PageClassifier(_cfg())
        assert pc._determine_type(200, 1, 0.1) == PageType.DIGITAL

    def test_threshold_boundary_digital(self):
        pc = PageClassifier(_cfg(digital_thresh=100))
        # Exactly at threshold with no images -> digital
        assert pc._determine_type(100, 0, 0.0) == PageType.DIGITAL

    def test_threshold_boundary_scanned(self):
        pc = PageClassifier(_cfg(scanned_thresh=20))
        # Below scanned threshold -> scanned
        assert pc._determine_type(19, 0, 0.0) == PageType.SCANNED


# --- Image coverage computation ---

class TestImageCoverage:
    def test_no_images(self):
        pc = PageClassifier(_cfg())
        coverage = pc._compute_image_coverage([], 100000)
        assert coverage == 0.0

    def test_full_coverage(self):
        pc = PageClassifier(_cfg())
        bboxes = [BoundingBox(0, 0, 100, 100)]
        coverage = pc._compute_image_coverage(bboxes, 10000)
        assert coverage == pytest.approx(1.0)

    def test_partial_coverage(self):
        pc = PageClassifier(_cfg())
        bboxes = [BoundingBox(0, 0, 50, 50)]
        coverage = pc._compute_image_coverage(bboxes, 10000)
        assert coverage == pytest.approx(0.25)


# --- classify_page with mocked fitz ---

class TestClassifyPageMocked:
    def _mock_page(self, text_blocks, image_info, width=612, height=792):
        page = MagicMock()
        page.rect.width = width
        page.rect.height = height

        # get_text("text") returns a string, get_text("blocks") returns blocks
        full_text = " ".join(b[4] for b in text_blocks if b[6] == 0)
        def _get_text(mode="text"):
            if mode == "text":
                return full_text
            return text_blocks
        page.get_text.side_effect = _get_text

        # get_images(full=True) returns list of image tuples (for counting)
        page.get_images.return_value = [(i,) for i in range(len(image_info))]
        # _get_image_bboxes uses page.get_image_info()
        page.get_image_info.return_value = image_info
        return page

    def test_digital_page(self):
        pc = PageClassifier(_cfg())
        text_blocks = [
            (10, 50, 200, 70, "This is a long enough text block for testing", 0, 0),
            (10, 80, 200, 100, "Another block of text with enough characters here", 1, 0),
            (10, 110, 200, 130, "Yet more text content to exceed threshold easily", 2, 0),
        ]
        mock_page = self._mock_page(text_blocks, [])
        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            result = pc.classify_page("fake.pdf", 0)

        assert result.page_type == PageType.DIGITAL
        assert result.embedded_char_count > 100

    def test_scanned_page(self):
        pc = PageClassifier(_cfg())
        # No text, large image covering most of the page
        mock_page = self._mock_page(
            [],
            [{"bbox": (0, 0, 600, 780)}],
        )
        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            result = pc.classify_page("fake.pdf", 0)

        assert result.page_type == PageType.SCANNED

    def test_hybrid_page(self):
        pc = PageClassifier(_cfg())
        text_blocks = [
            (10, 50, 200, 70, "Enough text to exceed the digital threshold by a mile", 0, 0),
            (10, 80, 200, 100, "More text here to make it clearly digital territory", 1, 0),
            (10, 110, 200, 130, "Even more text blocks for good measure in this test", 2, 0),
        ]
        # Large image covering >40% of the page
        mock_page = self._mock_page(
            text_blocks,
            [{"bbox": (0, 400, 600, 780)}],
        )
        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            result = pc.classify_page("fake.pdf", 0)

        assert result.page_type == PageType.HYBRID

    def test_classification_fields(self):
        pc = PageClassifier(_cfg())
        text_blocks = [
            (10, 50, 200, 70, "Text content here for classification", 0, 0),
            (10, 80, 200, 100, "More text content to reach threshold easily enough", 1, 0),
            (10, 110, 200, 130, "Third text block with enough content for digital", 2, 0),
        ]
        mock_page = self._mock_page(text_blocks, [])
        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            result = pc.classify_page("fake.pdf", 0)

        assert result.page_num == 0
        assert result.page_width == 612
        assert result.page_height == 792


# --- classify_document ---

class TestClassifyDocument:
    def test_multi_page(self):
        pc = PageClassifier(_cfg())

        # Page 0: digital (lots of text, no images)
        p0 = MagicMock()
        p0.rect.width = 612
        p0.rect.height = 792
        p0_text = "Lots of text " * 10
        p0.get_text.side_effect = lambda mode="text": p0_text if mode == "text" else []
        p0.get_images.return_value = []
        p0.get_image_info.return_value = []

        # Page 1: scanned (no text, big image)
        p1 = MagicMock()
        p1.rect.width = 612
        p1.rect.height = 792
        p1.get_text.side_effect = lambda mode="text": "" if mode == "text" else []
        p1.get_images.return_value = [(0,)]
        p1.get_image_info.return_value = [{"bbox": (0, 0, 600, 780)}]

        pages = [p0, p1]

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=lambda i: pages[i])
        mock_doc.close = MagicMock()

        with patch("fitz.open", return_value=mock_doc):
            results = pc.classify_document("fake.pdf")

        assert len(results) == 2
        assert results[0].page_type == PageType.DIGITAL
        assert results[1].page_type == PageType.SCANNED


# --- Fallback to merger config ---

class TestConfigFallback:
    def test_uses_merger_thresholds_when_no_page_classification(self):
        cfg = {
            "extraction": {
                "merger": {
                    "digital_threshold": 150,
                    "scanned_threshold": 30,
                }
            }
        }
        pc = PageClassifier(cfg)
        # 150 chars should be digital with the merger threshold
        assert pc._determine_type(150, 0, 0.0) == PageType.DIGITAL
        # 29 chars should be scanned
        assert pc._determine_type(29, 0, 0.0) == PageType.SCANNED
