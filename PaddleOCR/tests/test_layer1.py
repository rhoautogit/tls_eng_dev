"""Tests for Layer 1: Core Extraction Engine."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import BoundingBox, ExtractionParameters, PageExtractionResult, Table, TextBlock


# ─── BoundingBox tests ────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_area(self):
        bb = BoundingBox(0, 0, 10, 5)
        assert bb.area == 50.0

    def test_zero_area(self):
        assert BoundingBox(5, 5, 5, 5).area == 0.0

    def test_iou_identical(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert bb.iou(bb) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = BoundingBox(0, 0, 5, 5)
        b = BoundingBox(10, 10, 20, 20)
        assert a.iou(b) == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        a = BoundingBox(0, 0, 10, 10)
        b = BoundingBox(5, 5, 15, 15)
        # intersection = 5*5=25, union = 100+100-25=175
        assert a.iou(b) == pytest.approx(25 / 175)

    def test_contains_point(self):
        bb = BoundingBox(0, 0, 10, 10)
        assert bb.contains_point(5, 5)
        assert not bb.contains_point(11, 5)

    def test_to_tuple(self):
        bb = BoundingBox(1, 2, 3, 4)
        assert bb.to_tuple() == (1, 2, 3, 4)


# ─── pdfplumber extractor helpers ─────────────────────────────────────────────

class TestWordsToBlocks:
    def _words(self, entries):
        return [
            {"text": t, "x0": x0, "top": top, "x1": x1, "bottom": bot}
            for t, x0, top, x1, bot in entries
        ]

    def test_single_word(self):
        from src.layer1.pdfplumber_extractor import _words_to_blocks
        words = self._words([("hello", 10, 10, 50, 20)])
        blocks = _words_to_blocks(words)
        assert len(blocks) == 1
        assert blocks[0]["text"] == "hello"

    def test_two_words_same_line(self):
        from src.layer1.pdfplumber_extractor import _words_to_blocks
        words = self._words([
            ("hello", 10, 10, 50, 20),
            ("world", 55, 10, 90, 20),
        ])
        blocks = _words_to_blocks(words)
        assert len(blocks) == 1
        assert "hello" in blocks[0]["text"]
        assert "world" in blocks[0]["text"]

    def test_two_separate_blocks(self):
        from src.layer1.pdfplumber_extractor import _words_to_blocks
        words = self._words([
            ("title", 10, 10, 80, 20),
            ("body",  10, 50, 80, 60),   # far below
        ])
        blocks = _words_to_blocks(words, block_gap=5.0)
        assert len(blocks) == 2

    def test_empty_words(self):
        from src.layer1.pdfplumber_extractor import _words_to_blocks
        assert _words_to_blocks([]) == []


class TestBuildTableSettings:
    def _params(self, mode="lattice", text_align=False):
        p = ExtractionParameters()
        p.pdfplumber_mode = mode
        p.pdfplumber_use_text_alignment = text_align
        p.pdfplumber_snap_tolerance = 3.0
        p.pdfplumber_join_tolerance = 3.0
        p.pdfplumber_edge_min_length = 3.0
        return p

    def test_lattice_mode(self):
        from src.layer1.pdfplumber_extractor import _build_table_settings
        s = _build_table_settings(self._params("lattice"))
        assert s["vertical_strategy"] == "lines"
        assert s["horizontal_strategy"] == "lines"

    def test_stream_mode(self):
        from src.layer1.pdfplumber_extractor import _build_table_settings
        s = _build_table_settings(self._params("stream"))
        assert s["vertical_strategy"] == "text"

    def test_text_alignment_mode(self):
        from src.layer1.pdfplumber_extractor import _build_table_settings
        s = _build_table_settings(self._params(text_align=True))
        assert s["vertical_strategy"] == "text"
        assert s["horizontal_strategy"] == "text"


# ─── OpenCV extractor helpers ─────────────────────────────────────────────────

class TestCellsToGrid:
    def test_simple_2x2(self):
        from src.layer1.opencv_extractor import cells_to_grid
        cells = [
            BoundingBox(0, 0, 50, 25),
            BoundingBox(50, 0, 100, 25),
            BoundingBox(0, 25, 50, 50),
            BoundingBox(50, 25, 100, 50),
        ]
        bbox, grid = cells_to_grid(cells)
        assert bbox is not None
        assert len(grid) == 2
        assert len(grid[0]) == 2

    def test_empty_cells(self):
        from src.layer1.opencv_extractor import cells_to_grid
        bbox, grid = cells_to_grid([])
        assert bbox is None
        assert grid == []


class TestClusterCells:
    def test_two_groups(self):
        from src.layer1.opencv_extractor import cluster_cells
        cells_a = [BoundingBox(0, 0, 50, 25), BoundingBox(50, 0, 100, 25)]
        cells_b = [BoundingBox(0, 200, 50, 225), BoundingBox(50, 200, 100, 225)]
        clusters = cluster_cells(cells_a + cells_b, img_width=200, gap_pct=0.05)
        assert len(clusters) == 2

    def test_single_cluster(self):
        from src.layer1.opencv_extractor import cluster_cells
        cells = [BoundingBox(0, 0, 50, 25), BoundingBox(50, 0, 100, 25)]
        clusters = cluster_cells(cells, img_width=200, gap_pct=0.1)
        assert len(clusters) == 1


# ─── Result merger ────────────────────────────────────────────────────────────

class TestResultMerger:
    def _make_result(self, source, texts=None, tables=None, scanned=False):
        texts = texts or []
        tables = tables or []
        return PageExtractionResult(
            page_num=0,
            text_blocks=texts,
            tables=tables,
            images=[],
            source=source,
            is_scanned=scanned,
            page_width=600,
            page_height=800,
        )

    def _cfg(self):
        return {
            "extraction": {
                "merger": {
                    "digital_threshold": 100,
                    "scanned_threshold": 20,
                    "digital_pdfplumber_weight": 0.7,
                    "digital_opencv_weight": 0.3,
                    "scanned_pdfplumber_weight": 0.3,
                    "scanned_opencv_weight": 0.7,
                    "table_iou_threshold": 0.4,
                    "text_iou_threshold": 0.5,
                }
            }
        }

    def test_merge_produces_result(self):
        from src.layer1.result_merger import ResultMerger
        merger = ResultMerger(self._cfg())
        pp = self._make_result("pdfplumber")
        cv = self._make_result("opencv")
        out = merger.merge(pp, cv)
        assert isinstance(out, PageExtractionResult)
        assert out.source == "merged"

    def test_deduplication_removes_overlap(self):
        from src.layer1.result_merger import ResultMerger, _dedupe_tables
        t1 = Table(data=[["a"]], bbox=BoundingBox(0, 0, 100, 50), page_num=0, confidence=0.9)
        t2 = Table(data=[["a"]], bbox=BoundingBox(10, 5, 110, 55), page_num=0, confidence=0.7)
        result = _dedupe_tables([t1, t2], iou_threshold=0.3)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # higher confidence kept

    def test_merge_with_previous_target(self):
        from src.layer1.result_merger import ResultMerger
        merger = ResultMerger(self._cfg())
        prev_block = TextBlock(
            text="outside",
            bbox=BoundingBox(0, 0, 100, 20),
            page_num=0,
            confidence=1.0,
        )
        prev = PageExtractionResult(
            page_num=0, text_blocks=[prev_block], tables=[], images=[],
            source="merged", page_width=600, page_height=800,
        )
        retry_block = TextBlock(
            text="inside",
            bbox=BoundingBox(200, 200, 300, 250),
            page_num=0,
            confidence=1.0,
        )
        retry = PageExtractionResult(
            page_num=0, text_blocks=[retry_block], tables=[], images=[],
            source="opencv", page_width=600, page_height=800,
        )
        target = BoundingBox(150, 150, 350, 350)
        result = merger.merge_with_previous(prev, retry, target)
        texts = {b.text for b in result.text_blocks}
        assert "outside" in texts   # kept (outside target)
        assert "inside" in texts    # added from retry


# ─── Custom table logic ───────────────────────────────────────────────────────

class TestCustomTableLogic:
    def _block(self, text, x0, y0, x1, y1):
        return TextBlock(
            text=text,
            bbox=BoundingBox(x0, y0, x1, y1),
            page_num=0,
        )

    def test_detect_implicit_table(self):
        from src.layer1.custom_table_logic import detect_implicit_tables
        # Simulate a 2x3 grid of text blocks
        blocks = [
            self._block("A1", 0,   0,  80, 20),
            self._block("B1", 100, 0, 180, 20),
            self._block("A2", 0,  30,  80, 50),
            self._block("B2", 100, 30, 180, 50),
            self._block("A3", 0,  60,  80, 80),
            self._block("B3", 100, 60, 180, 80),
        ]
        tables, remaining = detect_implicit_tables(
            blocks, page_width=600, page_height=800,
            min_cluster_size=4, alignment_tolerance=10.0,
        )
        assert len(tables) >= 1

    def test_parse_nested_headers(self):
        from src.layer1.custom_table_logic import parse_nested_headers
        table = Table(
            data=[
                ["Group A", "", "Group B", ""],     # level-0 headers
                ["col1", "col2", "col3", "col4"],    # level-1 headers
                ["1", "2", "3", "4"],
            ],
            bbox=BoundingBox(0, 0, 400, 100),
            page_num=0,
        )
        result = parse_nested_headers(table)
        # data rows should be only ["1","2","3","4"]
        assert len(result.data) == 1
