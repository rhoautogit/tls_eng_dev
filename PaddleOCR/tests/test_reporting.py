"""Tests for the PDF Digestion Report generator."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    BoundingBox,
    DocumentResult,
    ExtractionParameters,
    PageExtractionResult,
    PageResult,
    RunRecord,
    Gap,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_doc_result(num_pages: int = 2) -> DocumentResult:
    pages = []
    for i in range(num_pages):
        extraction = PageExtractionResult(
            page_num=i,
            text_blocks=[],
            tables=[],
            images=[],
            source="test",
            page_width=600,
            page_height=800,
        )
        is_unresolved = i == 1

        run_records = []
        if is_unresolved:
            run_records = [
                RunRecord(
                    run_number=1,
                    parameters={"pdfplumber_mode": "stream"},
                    score_before=0.6,
                    score_after=0.72,
                    delta=0.12,
                    gap_map_path="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    gaps=[
                        Gap(
                            bbox=BoundingBox(100, 200, 300, 400),
                            area_ratio=0.08,
                            estimated_type="table",
                            severity="high",
                            page_num=i,
                        )
                    ],
                ),
                RunRecord(
                    run_number=2,
                    parameters={"opencv_use_clahe": True},
                    score_before=0.72,
                    score_after=0.80,
                    delta=0.08,
                    gap_map_path="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    gaps=[],
                ),
                RunRecord(
                    run_number=3,
                    parameters={"max_sensitivity": True},
                    score_before=0.80,
                    score_after=0.85,
                    delta=0.05,
                    gap_map_path="",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    gaps=[],
                ),
            ]

        pages.append(
            PageResult(
                page_num=i,
                final_score=0.97 if not is_unresolved else 0.85,
                initial_score=0.97 if not is_unresolved else 0.6,
                passed=not is_unresolved,
                extraction=extraction,
                run_records=run_records,
                gap_map_paths=[],
                status="passed_initial" if not is_unresolved else "unresolved",
            )
        )

    return DocumentResult(
        pdf_path="/fake/path/document.pdf",
        total_pages=num_pages,
        pages=pages,
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_score=sum(p.final_score for p in pages) / num_pages,
    )


def _cfg(tmp_path: Path) -> dict:
    return {
        "output": {
            "reports_dir": str(tmp_path / "reports"),
        },
        "reporting": {
            "formats": ["json", "html", "csv"],
            "embed_images_in_html": False,
        },
    }


# ─── DocumentResult serialisation ────────────────────────────────────────────

class TestDocumentResultDict:
    def test_structure(self):
        doc = _make_doc_result(2)
        d = doc.to_dict()
        assert "pdf_path" in d
        assert "overall_score" in d
        assert "summary" in d
        assert "pages" in d
        assert len(d["pages"]) == 2

    def test_summary_fields(self):
        doc = _make_doc_result(2)
        s = doc.to_dict()["summary"]
        assert "passed_pages" in s
        assert "unresolved_pages" in s
        assert "pass_rate" in s
        assert "confidence_breakdown" in s

    def test_page_has_run_records(self):
        doc = _make_doc_result(2)
        page1 = doc.to_dict()["pages"][1]  # unresolved page
        assert len(page1["run_records"]) == 3

    def test_run_record_structure(self):
        doc = _make_doc_result(2)
        rr = doc.to_dict()["pages"][1]["run_records"][0]
        assert "run_number" in rr
        assert "score_before" in rr
        assert "score_after" in rr
        assert "delta" in rr
        assert "parameters" in rr
        assert "gaps" in rr

    def test_gap_structure(self):
        doc = _make_doc_result(2)
        gap = doc.to_dict()["pages"][1]["run_records"][0]["gaps"][0]
        assert "bbox" in gap
        assert "area_ratio" in gap
        assert "estimated_type" in gap
        assert "severity" in gap


# ─── JSON report ──────────────────────────────────────────────────────────────

class TestJSONReport:
    def test_writes_valid_json(self, tmp_path):
        from src.reporting.report_generator import write_json
        doc = _make_doc_result()
        out = tmp_path / "report.json"
        write_json(doc, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "pages" in data

    def test_json_contains_overall_score(self, tmp_path):
        from src.reporting.report_generator import write_json
        doc = _make_doc_result()
        out = tmp_path / "report.json"
        write_json(doc, out)
        data = json.loads(out.read_text())
        assert "overall_score" in data
        assert 0 <= data["overall_score"] <= 1


# ─── CSV report ───────────────────────────────────────────────────────────────

class TestCSVReport:
    def test_writes_csv_with_header(self, tmp_path):
        from src.reporting.report_generator import write_csv
        doc = _make_doc_result(2)
        out = tmp_path / "report.csv"
        write_csv(doc, out)
        assert out.exists()
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert "page" in rows[0]
        assert "final_score" in rows[0]
        assert "status" in rows[0]

    def test_unresolved_flagged(self, tmp_path):
        from src.reporting.report_generator import write_csv
        doc = _make_doc_result(2)
        out = tmp_path / "report.csv"
        write_csv(doc, out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        unresolved_row = rows[1]
        assert unresolved_row["flagged_for_review"] == "True"

    def test_passed_not_flagged(self, tmp_path):
        from src.reporting.report_generator import write_csv
        doc = _make_doc_result(2)
        out = tmp_path / "report.csv"
        write_csv(doc, out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        passed_row = rows[0]
        assert passed_row["flagged_for_review"] == "False"


# ─── HTML report ──────────────────────────────────────────────────────────────

class TestHTMLReport:
    def test_writes_valid_html(self, tmp_path):
        from src.reporting.report_generator import write_html
        doc = _make_doc_result()
        out = tmp_path / "report.html"
        write_html(doc, out, embed_images=False)
        assert out.exists()
        content = out.read_text()
        assert "<!DOCTYPE html>" in content
        assert "TLS PDF Validation Report" in content

    def test_html_contains_page_scores(self, tmp_path):
        from src.reporting.report_generator import write_html
        doc = _make_doc_result(2)
        out = tmp_path / "report.html"
        write_html(doc, out, embed_images=False)
        content = out.read_text()
        assert "unresolved" in content.lower()
        assert "passed" in content.lower()

    def test_html_marks_manual_review(self, tmp_path):
        from src.reporting.report_generator import write_html
        doc = _make_doc_result(2)
        out = tmp_path / "report.html"
        write_html(doc, out, embed_images=False)
        content = out.read_text()
        assert "Manual Review" in content


# ─── ReportGenerator (integration) ───────────────────────────────────────────

class TestReportGenerator:
    def test_generates_all_formats(self, tmp_path):
        from src.reporting.report_generator import ReportGenerator
        doc = _make_doc_result()
        gen = ReportGenerator(_cfg(tmp_path))
        paths = gen.generate(doc, "test_doc")
        assert "json" in paths
        assert "html" in paths
        assert "csv" in paths
        for p in paths.values():
            assert Path(p).exists()
