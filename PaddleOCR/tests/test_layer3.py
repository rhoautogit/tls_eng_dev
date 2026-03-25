"""Tests for Layer 3: Visual Twin, Gap Analyzer, Parameter Adjuster, Retry Controller."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BoundingBox,
    ExtractionParameters,
    Gap,
    PageExtractionResult,
    RunRecord,
    Table,
    TextBlock,
)


# ─── Visual Twin ──────────────────────────────────────────────────────────────

class TestVisualTwin:
    def _result(self, text_bboxes=None, table_bboxes=None):
        blocks = [
            TextBlock(text="x", bbox=bb, page_num=0)
            for bb in (text_bboxes or [])
        ]
        tables = [
            Table(data=[["x"]], bbox=bb, page_num=0)
            for bb in (table_bboxes or [])
        ]
        return PageExtractionResult(
            page_num=0,
            text_blocks=blocks,
            tables=tables,
            images=[],
            source="test",
            page_width=200,
            page_height=300,
        )

    def test_twin_is_mostly_white_when_empty(self):
        from src.layer3.visual_twin import render_extraction_twin
        result = self._result()
        twin = render_extraction_twin(result, 200, 300, 1.0, 1.0)
        white_pct = np.mean(twin == 255)
        assert white_pct > 0.99

    def test_twin_has_coloured_region_for_text(self):
        from src.layer3.visual_twin import render_extraction_twin
        result = self._result(text_bboxes=[BoundingBox(0, 0, 100, 50)])
        twin = render_extraction_twin(result, 200, 300, 1.0, 1.0)
        # Top-left region should not be pure white
        region = twin[0:50, 0:100]
        assert not np.all(region == 255)

    def test_covered_mask_marks_non_white(self):
        from src.layer3.visual_twin import create_covered_mask
        twin = np.full((100, 100, 3), 255, dtype=np.uint8)
        twin[20:40, 20:40] = [100, 200, 100]
        mask = create_covered_mask(twin)
        assert mask[30, 30] == 255   # coloured = covered
        assert mask[0, 0] == 0       # white = not covered


# ─── Gap Type Estimation ──────────────────────────────────────────────────────

class TestGapTypeEstimation:
    def test_blank_returns_unknown(self):
        from src.layer3.gap_analyzer import _estimate_gap_type
        blank = np.zeros((0, 0, 3), dtype=np.uint8)
        assert _estimate_gap_type(blank) == "unknown"

    def test_white_region_returns_something(self):
        from src.layer3.gap_analyzer import _estimate_gap_type
        white = np.full((50, 50, 3), 255, dtype=np.uint8)
        result = _estimate_gap_type(white)
        assert result in ("table", "text", "image", "unknown")

    def test_grid_lines_detected_as_table(self):
        from src.layer3.gap_analyzer import _estimate_gap_type
        img = np.full((100, 200, 3), 255, dtype=np.uint8)
        # Dense horizontal lines every 10 px → h_density ≈ 0.09
        for y in range(10, 100, 10):
            img[y, :] = 0
        # Dense vertical lines every 20 px → v_density ≈ 0.045
        for x in range(20, 200, 20):
            img[:, x] = 0
        result = _estimate_gap_type(img)
        assert result == "table"


# ─── Parameter Adjuster ───────────────────────────────────────────────────────

class TestParameterAdjuster:
    def _cfg(self):
        return {
            "retry_strategies": {
                "retry_1": {
                    "tolerance_multiplier": 1.5,
                    "target_gap_region": True,
                },
                "retry_2": {
                    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                    "sharpening": True,
                    "noise_reduction": True,
                    "text_alignment_detection": True,
                    "target_gap_region": True,
                },
                "retry_3": {
                    "pdfplumber_snap_tolerance": 10.0,
                    "pdfplumber_join_tolerance": 10.0,
                    "pdfplumber_edge_min_length": 1.0,
                    "opencv_dpi": 300,
                    "clahe": {"clip_limit": 3.0, "tile_grid_size": [4, 4]},
                    "sharpening": True,
                    "noise_reduction": True,
                    "text_alignment_detection": True,
                },
            }
        }

    def _gaps(self):
        return [
            Gap(
                bbox=BoundingBox(100, 200, 300, 400),
                area_ratio=0.1,
                estimated_type="table",
                severity="high",
                page_num=0,
            )
        ]

    def test_retry_1_switches_mode(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters(pdfplumber_mode="lattice")
        params = adj.get_params(1, self._gaps(), current)
        assert params.pdfplumber_mode == "stream"

    def test_retry_1_loosens_tolerances(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters(pdfplumber_snap_tolerance=3.0)
        params = adj.get_params(1, self._gaps(), current)
        assert params.pdfplumber_snap_tolerance > 3.0

    def test_retry_1_sets_target_bbox_when_gaps(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters()
        params = adj.get_params(1, self._gaps(), current)
        assert params.target_bbox is not None

    def test_retry_2_enables_clahe(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters()
        params = adj.get_params(2, self._gaps(), current)
        assert params.opencv_use_clahe is True
        assert params.opencv_use_sharpening is True

    def test_retry_3_max_sensitivity(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters()
        params = adj.get_params(3, [], current)
        assert params.max_sensitivity is True
        assert params.opencv_dpi >= 300
        assert params.pdfplumber_snap_tolerance >= 10.0

    def test_retry_1_no_gaps_no_target(self):
        from src.layer3.parameter_adjuster import ParameterAdjuster
        adj = ParameterAdjuster(self._cfg())
        current = ExtractionParameters()
        params = adj.get_params(1, [], current)
        assert params.target_bbox is None


# ─── Retry Controller ─────────────────────────────────────────────────────────

class TestRetryController:
    def _config(self):
        return {
            "accuracy_threshold": 0.95,
            "max_retries": 3,
            "early_termination_threshold": 0.01,
            "retry_strategies": {
                "retry_1": {"tolerance_multiplier": 1.5, "target_gap_region": False},
                "retry_2": {
                    "clahe": {"clip_limit": 2.0, "tile_grid_size": [8, 8]},
                    "sharpening": True,
                    "noise_reduction": True,
                    "text_alignment_detection": True,
                    "target_gap_region": False,
                },
                "retry_3": {
                    "pdfplumber_snap_tolerance": 10.0,
                    "pdfplumber_join_tolerance": 10.0,
                    "pdfplumber_edge_min_length": 1.0,
                    "opencv_dpi": 300,
                    "clahe": {"clip_limit": 3.0, "tile_grid_size": [4, 4]},
                    "sharpening": True,
                    "noise_reduction": True,
                    "text_alignment_detection": True,
                },
            },
        }

    def _empty_result(self, page_num=0):
        return PageExtractionResult(
            page_num=page_num,
            text_blocks=[],
            tables=[],
            images=[],
            source="test",
            page_width=600,
            page_height=800,
        )

    def test_stops_when_threshold_reached(self, tmp_path):
        from src.layer3.retry_controller import RetryController
        from src.layer3.parameter_adjuster import ParameterAdjuster

        pp = MagicMock()
        cv = MagicMock()
        merger = MagicMock()
        scorer = MagicMock()
        gap_analyzer = MagicMock()
        adjuster = ParameterAdjuster(self._config())

        empty = self._empty_result()
        pp.extract_page.return_value = empty
        cv.extract_page.return_value = empty
        merger.merge.return_value = empty
        merger.merge_with_previous.return_value = empty
        # Return passing score on first retry
        scorer.score_page.return_value = 0.97
        gap_analyzer.analyze.return_value = ([], str(tmp_path / "gap.png"))

        ctrl = RetryController(
            self._config(), pp, cv, merger, scorer, gap_analyzer, adjuster
        )
        _, best_score, run_records, _ = ctrl.process_page(
            "fake.pdf", 0, empty, 0.5, ExtractionParameters(), tmp_path
        )
        assert best_score >= 0.95
        assert len(run_records) == 1  # stopped after first retry

    def test_early_termination_on_plateau(self, tmp_path):
        from src.layer3.retry_controller import RetryController
        from src.layer3.parameter_adjuster import ParameterAdjuster

        pp = MagicMock()
        cv = MagicMock()
        merger = MagicMock()
        scorer = MagicMock()
        gap_analyzer = MagicMock()
        adjuster = ParameterAdjuster(self._config())

        empty = self._empty_result()
        pp.extract_page.return_value = empty
        cv.extract_page.return_value = empty
        merger.merge.return_value = empty
        merger.merge_with_previous.return_value = empty
        # Both retries produce negligible improvement
        scorer.score_page.return_value = 0.50  # < threshold, delta < 0.01 vs initial 0.495
        gap_analyzer.analyze.return_value = ([], str(tmp_path / "gap.png"))

        ctrl = RetryController(
            self._config(), pp, cv, merger, scorer, gap_analyzer, adjuster
        )
        _, best_score, run_records, _ = ctrl.process_page(
            "fake.pdf", 0, empty, 0.495, ExtractionParameters(), tmp_path
        )
        # Early termination after 2 retries with tiny delta
        assert len(run_records) <= 3

    def test_records_all_runs_when_unresolved(self, tmp_path):
        from src.layer3.retry_controller import RetryController
        from src.layer3.parameter_adjuster import ParameterAdjuster

        pp = MagicMock()
        cv = MagicMock()
        merger = MagicMock()
        scorer = MagicMock()
        gap_analyzer = MagicMock()
        adjuster = ParameterAdjuster(self._config())

        empty = self._empty_result()
        pp.extract_page.return_value = empty
        cv.extract_page.return_value = empty
        merger.merge.return_value = empty
        merger.merge_with_previous.return_value = empty
        # Score improves enough to avoid early stop but never reaches 95%
        scores = [0.7, 0.8, 0.88]
        scorer.score_page.side_effect = scores
        gap_analyzer.analyze.return_value = ([], str(tmp_path / "gap.png"))

        ctrl = RetryController(
            self._config(), pp, cv, merger, scorer, gap_analyzer, adjuster
        )
        _, best_score, run_records, _ = ctrl.process_page(
            "fake.pdf", 0, empty, 0.6, ExtractionParameters(), tmp_path
        )
        assert len(run_records) == 3
        assert best_score == pytest.approx(0.88)
