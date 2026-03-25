"""Tests for path executors and region splitter."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.models import (
    BoundingBox,
    ConfidenceGate,
    ConfidenceLevel,
    ExtractionParameters,
    PageClassification,
    PageExtractionResult,
    PageResult,
    PageType,
    RegionInfo,
    RunRecord,
    Table,
    TextBlock,
)


# --- Region Splitter ---

class TestRegionSplitter:
    def test_text_overlap_ratio_no_overlap(self):
        from src.paths.region_splitter import RegionSplitter
        rs = RegionSplitter()
        img_bb = BoundingBox(0, 0, 100, 100)
        text_regions = [BoundingBox(200, 200, 300, 300)]
        ratio = rs._text_overlap_ratio(img_bb, text_regions)
        assert ratio == pytest.approx(0.0)

    def test_text_overlap_ratio_full_overlap(self):
        from src.paths.region_splitter import RegionSplitter
        rs = RegionSplitter()
        img_bb = BoundingBox(0, 0, 100, 100)
        text_regions = [BoundingBox(0, 0, 100, 100)]
        ratio = rs._text_overlap_ratio(img_bb, text_regions)
        assert ratio == pytest.approx(1.0)

    def test_text_overlap_ratio_partial(self):
        from src.paths.region_splitter import RegionSplitter
        rs = RegionSplitter()
        img_bb = BoundingBox(0, 0, 100, 100)
        text_regions = [BoundingBox(50, 50, 150, 150)]
        # Intersection = 50*50 = 2500, img area = 10000
        ratio = rs._text_overlap_ratio(img_bb, text_regions)
        assert ratio == pytest.approx(0.25)

    def test_text_overlap_ratio_zero_area(self):
        from src.paths.region_splitter import RegionSplitter
        rs = RegionSplitter()
        img_bb = BoundingBox(0, 0, 0, 0)
        ratio = rs._text_overlap_ratio(img_bb, [])
        assert ratio == 0.0

    def test_split_page_mocked(self):
        from src.paths.region_splitter import RegionSplitter
        rs = RegionSplitter()

        mock_page = MagicMock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        # Text blocks in upper half
        mock_page.get_text.return_value = [
            (10, 50, 200, 70, "Text content here", 0, 0),
            (10, 80, 200, 100, "More text content", 1, 0),
        ]
        # Large image in lower half (no text overlap)
        mock_page.get_image_info.return_value = [
            {"bbox": (0, 400, 600, 780)},
        ]

        mock_doc = MagicMock()
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        with patch("fitz.open", return_value=mock_doc):
            regions = rs.split_page("fake.pdf", 0)

        digital_count = sum(1 for r in regions if r.region_type == PageType.DIGITAL)
        scanned_count = sum(1 for r in regions if r.region_type == PageType.SCANNED)
        assert digital_count >= 1
        assert scanned_count >= 1


# --- Digital Path ---

class TestDigitalPath:
    def _make_classification(self, page_type=PageType.DIGITAL):
        return PageClassification(
            page_num=0,
            page_type=page_type,
            embedded_char_count=500,
            image_region_count=0,
            image_coverage_ratio=0.0,
            page_width=612,
            page_height=792,
        )

    def test_execute_returns_page_result(self, tmp_path):
        from src.paths.digital_path import DigitalPathExecutor

        config = {
            "accuracy_threshold": 0.95,
            "max_retries": 3,
            "early_termination_threshold": 0.01,
            "retry_strategies": {
                "retry_1": {"tolerance_multiplier": 1.5, "target_gap_region": True},
                "retry_2": {"clahe": {"clip_limit": 2.0}, "sharpening": True, "target_gap_region": True},
                "retry_3": {"combine_all": True},
            },
        }

        extraction = PageExtractionResult(
            page_num=0,
            text_blocks=[TextBlock(text="Test", bbox=BoundingBox(10, 50, 200, 70), page_num=0)],
            tables=[],
            images=[],
            source="pdfplumber",
            page_width=612,
            page_height=792,
        )

        pp = MagicMock()
        pp.extract_page.return_value = extraction
        custom = MagicMock()
        custom.process.return_value = extraction
        scorer = MagicMock()
        scorer.score_page.return_value = 0.97
        scorer.score_page_detailed.return_value = {
            "total_score": 0.97,
            "pdfplumber_pct": 0.90,
            "rich_pct": 0.07,
        }
        gap_analyzer = MagicMock()
        adjuster = MagicMock()

        with patch("src.paths.digital_path.extract_rich_page", return_value={"text_blocks": []}), \
             patch("src.paths.digital_path.save_rich_page"):
            executor = DigitalPathExecutor(
                config, pp, custom, scorer, gap_analyzer, adjuster,
            )
            result = executor.execute(
                "fake.pdf", 0, self._make_classification(),
                ExtractionParameters(), tmp_path,
            )

        assert isinstance(result, PageResult)
        assert result.passed is True
        assert result.final_score >= 0.95


# --- Hybrid Path merge ---

class TestHybridPathMerge:
    def test_merge_results_no_duplicates(self):
        from src.paths.hybrid_path import HybridPathExecutor

        config = {"accuracy_threshold": 0.95}
        executor = HybridPathExecutor(
            config,
            MagicMock(),  # pp
            MagicMock(),  # cv
            MagicMock(),  # custom
            MagicMock(),  # scorer
        )

        digital = PageExtractionResult(
            page_num=0,
            text_blocks=[
                TextBlock(text="Digital text", bbox=BoundingBox(10, 10, 200, 30), page_num=0),
            ],
            tables=[],
            images=[],
            source="pdfplumber",
            page_width=612,
            page_height=792,
        )

        scanned = PageExtractionResult(
            page_num=0,
            text_blocks=[
                # This overlaps with digital
                TextBlock(text="Digital text", bbox=BoundingBox(12, 12, 198, 28), page_num=0),
                # This is unique
                TextBlock(text="OCR text", bbox=BoundingBox(10, 400, 200, 420), page_num=0),
            ],
            tables=[],
            images=[],
            source="paddleocr",
            page_width=612,
            page_height=792,
        )

        merged = executor._merge_results(digital, scanned, 0, 612, 792)
        # Should have digital text + unique OCR text, not duplicated overlap
        assert len(merged.text_blocks) == 2
        texts = {b.text for b in merged.text_blocks}
        assert "Digital text" in texts
        assert "OCR text" in texts

    def test_merge_opencv_tables_dedup(self):
        from src.paths.hybrid_path import HybridPathExecutor

        executor = HybridPathExecutor(
            {"accuracy_threshold": 0.95},
            MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )

        extraction = PageExtractionResult(
            page_num=0,
            text_blocks=[],
            tables=[
                Table(data=[["a"]], bbox=BoundingBox(10, 10, 200, 100), page_num=0),
            ],
            images=[],
            source="pdfplumber",
            page_width=612,
            page_height=792,
        )

        cv_result = PageExtractionResult(
            page_num=0,
            text_blocks=[],
            tables=[
                # Overlapping table (should be skipped)
                Table(data=[["a"]], bbox=BoundingBox(12, 12, 198, 98), page_num=0),
                # Non-overlapping table (should be added)
                Table(data=[["b"]], bbox=BoundingBox(10, 300, 200, 400), page_num=0),
            ],
            images=[],
            source="opencv",
            page_width=612,
            page_height=792,
        )

        merged = executor._merge_opencv_tables(extraction, cv_result)
        assert len(merged.tables) == 2  # original + non-overlapping


# --- Scanned Path retry logic ---

class TestScannedPathPreprocessRetry:
    def test_preprocess_for_retry_applies_clahe(self):
        from src.paths.scanned_path import preprocess_for_retry

        gray = np.full((100, 100), 128, dtype=np.uint8)
        strategy = {
            "clahe": {"clip_limit": 3.0, "tile_grid_size": [8, 8]},
            "deskew": False,
            "line_removal": False,
            "denoise": False,
        }
        result = preprocess_for_retry(gray, strategy)
        assert result.shape == gray.shape

    def test_preprocess_for_retry_adaptive_binarize(self):
        from src.paths.scanned_path import preprocess_for_retry

        gray = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        strategy = {
            "deskew": False,
            "line_removal": False,
            "denoise": False,
            "adaptive_binarize": True,
            "adaptive_block_size": 15,
            "adaptive_constant": 5,
        }
        result = preprocess_for_retry(gray, strategy)
        # Should be binary (0 or 255 only)
        unique = set(np.unique(result))
        assert unique.issubset({0, 255})

    def test_preprocess_for_retry_morphological_open(self):
        from src.paths.scanned_path import preprocess_for_retry

        gray = np.full((100, 100), 200, dtype=np.uint8)
        strategy = {
            "deskew": False,
            "line_removal": False,
            "denoise": False,
            "morphological_open": True,
            "morphological_kernel": [3, 3],
        }
        result = preprocess_for_retry(gray, strategy)
        assert result.shape == gray.shape


class TestScannedPathRetryLoop:
    def _config(self):
        return {
            "accuracy_threshold": 0.95,
            "scanned_path": {
                "ocr": {"dpi": 300},
                "preprocessing": {},
                "confidence_gate": {
                    "high_threshold": 0.95,
                    "medium_threshold": 0.85,
                },
                "retry": {
                    "max_retries": 3,
                    "early_termination_threshold": 0.01,
                    "strategies": {
                        "retry_1": {
                            "description": "Higher DPI",
                            "dpi": 400,
                            "deskew": False,
                            "line_removal": False,
                            "denoise": False,
                        },
                        "retry_2": {
                            "description": "Max DPI + CLAHE",
                            "dpi": 450,
                            "clahe": {"clip_limit": 4.0, "tile_grid_size": [4, 4]},
                            "deskew": False,
                            "line_removal": False,
                            "denoise": False,
                        },
                        "retry_3": {
                            "description": "Full combined",
                            "dpi": 600,
                            "clahe": {"clip_limit": 5.0, "tile_grid_size": [4, 4]},
                            "deskew": False,
                            "line_removal": False,
                            "denoise": False,
                        },
                    },
                },
            },
        }

    def test_scanned_executor_has_retry_config(self):
        from src.paths.scanned_path import ScannedPathExecutor
        executor = ScannedPathExecutor(
            self._config(), MagicMock(), MagicMock()
        )
        assert executor._max_retries == 3
        assert executor._early_stop_delta == 0.01
        assert executor._threshold == 0.95

    def test_retry_loop_improves_confidence(self):
        """Test that retry loop keeps the best result when confidence improves."""
        from src.paths.scanned_path import ScannedPathExecutor

        executor = ScannedPathExecutor(
            self._config(), MagicMock(), MagicMock()
        )

        initial_extraction = PageExtractionResult(
            page_num=0, text_blocks=[], tables=[], images=[],
            source="paddleocr", is_scanned=True,
            page_width=612, page_height=792, page_type=PageType.SCANNED,
        )
        initial_gate = ConfidenceGate(
            ocr_confidence=0.80, level=ConfidenceLevel.LOW,
            needs_qwen_vl=True, needs_human_review=True,
            word_count=100, high_confidence_words=60, flagged_words=40,
        )

        # Mock the retry functions to return improving confidence
        improving_gates = [
            ConfidenceGate(
                ocr_confidence=0.88, level=ConfidenceLevel.MEDIUM,
                needs_qwen_vl=True, needs_human_review=False,
                word_count=100, high_confidence_words=70, flagged_words=30,
            ),
            ConfidenceGate(
                ocr_confidence=0.93, level=ConfidenceLevel.MEDIUM,
                needs_qwen_vl=True, needs_human_review=False,
                word_count=100, high_confidence_words=85, flagged_words=15,
            ),
            ConfidenceGate(
                ocr_confidence=0.96, level=ConfidenceLevel.HIGH,
                needs_qwen_vl=False, needs_human_review=False,
                word_count=100, high_confidence_words=95, flagged_words=5,
            ),
        ]
        gate_iter = iter(improving_gates)

        # Mock PaddleOCR engine on the executor
        from src.paddle_ocr_engine import OCRPageResult
        mock_ocr_result = OCRPageResult(
            lines=[], page_confidence=0.0, word_count=100,
            high_confidence_words=95, flagged_words=5,
        )

        def mock_ocr_image(*a, **k):
            return mock_ocr_result

        executor._engine = MagicMock()
        executor._engine.ocr_image = mock_ocr_image
        executor._engine.detect_table_structure = MagicMock(return_value=[])

        with patch(
            "src.paths.scanned_path.render_page_image",
            return_value=(np.zeros((100, 100, 3), dtype=np.uint8), 612.0, 792.0),
        ), patch(
            "src.paths.scanned_path.preprocess_for_retry",
            return_value=np.zeros((100, 100), dtype=np.uint8),
        ), patch(
            "src.paths.scanned_path.compute_confidence_gate",
            side_effect=lambda *a, **k: next(gate_iter),
        ):
            best_ext, best_gate, best_score, records = executor._retry_loop(
                "fake.pdf", 0, 612, 792,
                initial_extraction, initial_gate, 0.5,
                ExtractionParameters(), Path("/tmp"),
            )

        # Should have reached threshold and stopped
        assert best_gate.ocr_confidence >= 0.95
        assert len(records) == 3  # all 3 retries ran, 3rd one passed
        assert records[-1].score_after >= 0.95

    def test_retry_loop_early_stop_on_plateau(self):
        """Test that retry loop stops early when confidence plateaus."""
        from src.paths.scanned_path import ScannedPathExecutor

        executor = ScannedPathExecutor(
            self._config(), MagicMock(), MagicMock()
        )

        initial_extraction = PageExtractionResult(
            page_num=0, text_blocks=[], tables=[], images=[],
            source="paddleocr", is_scanned=True,
            page_width=612, page_height=792, page_type=PageType.SCANNED,
        )
        initial_gate = ConfidenceGate(
            ocr_confidence=0.80, level=ConfidenceLevel.LOW,
            needs_qwen_vl=True, needs_human_review=True,
            word_count=100, high_confidence_words=60, flagged_words=40,
        )

        # Plateau: confidence barely moves
        plateau_gates = [
            ConfidenceGate(
                ocr_confidence=0.805, level=ConfidenceLevel.LOW,
                needs_qwen_vl=True, needs_human_review=True,
                word_count=100,
            ),
            ConfidenceGate(
                ocr_confidence=0.808, level=ConfidenceLevel.LOW,
                needs_qwen_vl=True, needs_human_review=True,
                word_count=100,
            ),
            ConfidenceGate(
                ocr_confidence=0.809, level=ConfidenceLevel.LOW,
                needs_qwen_vl=True, needs_human_review=True,
                word_count=100,
            ),
        ]
        gate_iter = iter(plateau_gates)

        # Mock PaddleOCR engine on the executor
        from src.paddle_ocr_engine import OCRPageResult
        mock_ocr_result = OCRPageResult(
            lines=[], page_confidence=0.0, word_count=100,
        )

        executor._engine = MagicMock()
        executor._engine.ocr_image = MagicMock(return_value=mock_ocr_result)
        executor._engine.detect_table_structure = MagicMock(return_value=[])

        with patch(
            "src.paths.scanned_path.render_page_image",
            return_value=(np.zeros((100, 100, 3), dtype=np.uint8), 612.0, 792.0),
        ), patch(
            "src.paths.scanned_path.preprocess_for_retry",
            return_value=np.zeros((100, 100), dtype=np.uint8),
        ), patch(
            "src.paths.scanned_path.compute_confidence_gate",
            side_effect=lambda *a, **k: next(gate_iter),
        ):
            best_ext, best_gate, best_score, records = executor._retry_loop(
                "fake.pdf", 0, 612, 792,
                initial_extraction, initial_gate, 0.5,
                ExtractionParameters(), Path("/tmp"),
            )

        # Should early stop after 2 retries (both deltas < 0.01)
        assert len(records) == 2
        assert best_gate.ocr_confidence < 0.95


# --- Hybrid Path retry logic ---

class TestHybridPathRetry:
    def test_hybrid_has_retry_config(self):
        from src.paths.hybrid_path import HybridPathExecutor
        config = {
            "accuracy_threshold": 0.95,
            "scanned_path": {
                "retry": {
                    "max_retries": 3,
                    "early_termination_threshold": 0.01,
                },
            },
        }
        executor = HybridPathExecutor(
            config, MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )
        assert executor._max_retries == 3
        assert executor._early_stop_delta == 0.01


# --- RegionInfo dataclass ---

class TestRegionInfo:
    def test_region_info_fields(self):
        ri = RegionInfo(
            bbox=BoundingBox(0, 0, 100, 100),
            region_type=PageType.DIGITAL,
            char_count=500,
        )
        assert ri.region_type == PageType.DIGITAL
        assert ri.char_count == 500
        assert ri.bbox.area == 10000
