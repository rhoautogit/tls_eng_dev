"""Reporting -- PDF Digestion Report generator.

Produces three formats from a DocumentResult:
  - JSON  : full machine-readable report (all runs, all gaps, all scores)
  - HTML  : human-readable dashboard with 5-layer validation metrics,
            page type breakdown, verification status, and per-page details
  - CSV   : summary table across all pages (suitable for spreadsheet review)
"""
from __future__ import annotations

import base64
import csv
import json
import logging
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..models import DocumentResult, PageResult, RunRecord

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _b64_image(path: str) -> str:
    """Return a base64-encoded data URI for a PNG file, or empty string."""
    try:
        with open(path, "rb") as fh:
            data = base64.b64encode(fh.read()).decode()
        return f"data:image/png;base64,{data}"
    except Exception:
        return ""


def _status_class(status: str) -> str:
    return {
        "passed_initial": "passed",
        "resolved": "resolved",
        "unresolved": "unresolved",
    }.get(status, "")


def _pct(val: float) -> str:
    """Format a 0-1 float as a percentage string like '52.3%'."""
    return f"{val * 100:.1f}%"


def _score_color(val: float) -> str:
    """Return a CSS color based on score value."""
    if val >= 0.95:
        return "#27ae60"
    if val >= 0.85:
        return "#f39c12"
    return "#e74c3c"


# ── JSON ─────────────────────────────────────────────────────────────────────

def write_json(doc: DocumentResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(doc.to_dict(), fh, indent=2, ensure_ascii=False)
    logger.info("JSON report -> %s", out_path)


# ── CSV ──────────────────────────────────────────────────────────────────────

def write_csv(doc: DocumentResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "pdf_file",
        "page",
        "page_type",
        "status",
        "verification_status",
        "initial_score",
        "final_score",
        "coverage_score",
        "accuracy_score",
        "completeness_score",
        "structural_score",
        "composite_score",
        "ocr_confidence",
        "pdfplumber_pct",
        "opencv_pct",
        "rich_extractor_pct",
        "paddleocr_pct",
        "num_tables",
        "num_text_blocks",
        "num_images",
        "retries_used",
        "flagged_for_review",
    ]
    pdf_name = Path(doc.pdf_path).name
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for page in doc.pages:
            sc = page.source_contributions
            val = page.validation
            gate = page.extraction.confidence_gate
            writer.writerow(
                {
                    "pdf_file": pdf_name,
                    "page": page.page_num + 1,
                    "page_type": page.extraction.page_type.value,
                    "status": page.status,
                    "verification_status": page.extraction.verification_status.value,
                    "initial_score": f"{page.initial_score:.4f}",
                    "final_score": f"{page.final_score:.4f}",
                    "coverage_score": f"{val.coverage_score:.4f}" if val else "",
                    "accuracy_score": f"{val.accuracy_score:.4f}" if val else "",
                    "completeness_score": f"{val.completeness_score:.4f}" if val else "",
                    "structural_score": f"{val.structural_score:.4f}" if val else "",
                    "composite_score": f"{val.composite_score:.4f}" if val else "",
                    "ocr_confidence": f"{gate.ocr_confidence:.4f}" if gate else "",
                    "pdfplumber_pct": f"{sc.get('pdfplumber', 0.0):.4f}",
                    "opencv_pct": f"{sc.get('opencv', 0.0):.4f}",
                    "rich_extractor_pct": f"{sc.get('rich_extractor', 0.0):.4f}",
                    "paddleocr_pct": f"{sc.get('paddleocr', 0.0):.4f}",
                    "num_tables": len(page.extraction.tables),
                    "num_text_blocks": len(page.extraction.text_blocks),
                    "num_images": len(page.extraction.images),
                    "retries_used": len(page.run_records),
                    "flagged_for_review": page.status == "unresolved",
                }
            )
    logger.info("CSV report -> %s", out_path)


# ── HTML ─────────────────────────────────────────────────────────────────────

_HTML_CSS = """
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; color: #333; }
  h1, h2, h3 { color: #2c3e50; }
  .container { max-width: 1200px; margin: 0 auto; }
  .card { background: #fff; border-radius: 6px; padding: 20px; margin: 16px 0;
          box-shadow: 0 1px 4px rgba(0,0,0,.1); }
  table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
  th { background: #2c3e50; color: #fff; padding: 8px 12px; text-align: left; }
  td { padding: 7px 12px; border-bottom: 1px solid #eee; }
  tr:hover td { background: #f9f9f9; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; font-weight: bold; }
  .passed    { background: #d4edda; color: #155724; }
  .resolved  { background: #fff3cd; color: #856404; }
  .unresolved{ background: #f8d7da; color: #721c24; }
  .high   { background: #cce5ff; color: #004085; }
  .medium { background: #fff3cd; color: #856404; }
  .low    { background: #f8d7da; color: #721c24; }
  .digital  { background: #d1ecf1; color: #0c5460; }
  .scanned  { background: #e2d5f1; color: #432874; }
  .hybrid   { background: #ffeeba; color: #856404; }
  .auto_verified  { background: #d4edda; color: #155724; }
  .qwen_verified  { background: #cce5ff; color: #004085; }
  .qwen_corrected { background: #fff3cd; color: #856404; }
  .pending_human  { background: #f8d7da; color: #721c24; }
  .not_verified   { background: #e2e3e5; color: #383d41; }
  details > summary { cursor: pointer; padding: 8px; background: #ecf0f1;
                      border-radius: 4px; margin-bottom: 8px; }
  details[open] > summary { background: #d5dbdb; }
  .run-record { background: #f8f9fa; border-left: 4px solid #adb5bd;
                padding: 10px 14px; margin: 8px 0; border-radius: 0 4px 4px 0; }
  .run-record.improved { border-left-color: #28a745; }
  .gap-map { max-width: 100%; height: auto; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; }
  .stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
  .stat-box { background: #ecf0f1; border-radius: 6px; padding: 14px; text-align: center; }
  .stat-box .value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
  .stat-box .label { font-size: 0.8em; color: #7f8c8d; margin-top: 4px; }
  .score-bar { height: 20px; border-radius: 10px; background: #ecf0f1; overflow: hidden; margin: 2px 0; }
  .score-fill { height: 100%; border-radius: 10px; transition: width 0.3s; }
  .contrib-bar { display: flex; height: 24px; border-radius: 4px; overflow: hidden; margin: 4px 0; }
  .contrib-bar .seg { display: flex; align-items: center; justify-content: center;
                      font-size: 0.75em; font-weight: bold; color: #fff; min-width: 0; }
  .seg-pp  { background: #3498db; }
  .seg-cv  { background: #e67e22; }
  .seg-rich{ background: #2ecc71; }
  .seg-ocr { background: #9b59b6; }
  .contrib-legend { display: flex; gap: 16px; margin: 8px 0; font-size: 0.85em; flex-wrap: wrap; }
  .contrib-legend span { display: inline-flex; align-items: center; gap: 4px; }
  .legend-dot { width: 12px; height: 12px; border-radius: 2px; display: inline-block; }
  .val-layer { display: flex; align-items: center; gap: 8px; margin: 4px 0; }
  .val-layer .label { min-width: 120px; font-size: 0.85em; }
  .val-layer .bar { flex: 1; }
  .val-layer .score { min-width: 50px; text-align: right; font-weight: bold; font-size: 0.85em; }
  .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 768px) { .two-col { grid-template-columns: 1fr; } }
</style>
"""


def _score_bar_html(score: float, label: str = "") -> str:
    """Render a horizontal score bar with color gradient."""
    color = _score_color(score)
    width = max(2, score * 100)
    return f"""
    <div class="val-layer">
      <div class="label">{label}</div>
      <div class="bar"><div class="score-bar">
        <div class="score-fill" style="width:{width:.1f}%;background:{color}"></div>
      </div></div>
      <div class="score" style="color:{color}">{_pct(score)}</div>
    </div>"""


def _contribution_bar_html(sc: Dict[str, float]) -> str:
    """Render a stacked horizontal bar showing per-tool contributions."""
    pp = sc.get("pdfplumber", 0.0)
    cv = sc.get("opencv", 0.0)
    ri = sc.get("rich_extractor", 0.0)
    ocr = sc.get("paddleocr", 0.0)
    total = pp + cv + ri + ocr
    if total <= 0:
        return ""

    pp_w = (pp / total * 100)
    cv_w = (cv / total * 100)
    ri_w = (ri / total * 100)
    ocr_w = (ocr / total * 100)

    pp_label = _pct(pp) if pp >= 0.01 else ""
    cv_label = _pct(cv) if cv >= 0.01 else ""
    ri_label = _pct(ri) if ri >= 0.01 else ""
    ocr_label = _pct(ocr) if ocr >= 0.01 else ""

    return f"""
    <div class="contrib-bar" title="pdfplumber: {_pct(pp)} | OpenCV: {_pct(cv)} | Rich: {_pct(ri)} | OCR: {_pct(ocr)}">
      <div class="seg seg-pp" style="width:{pp_w:.1f}%">{pp_label}</div>
      <div class="seg seg-cv" style="width:{cv_w:.1f}%">{cv_label}</div>
      <div class="seg seg-rich" style="width:{ri_w:.1f}%">{ri_label}</div>
      <div class="seg seg-ocr" style="width:{ocr_w:.1f}%">{ocr_label}</div>
    </div>"""


_CONTRIB_LEGEND = """
<div class="contrib-legend">
  <span><span class="legend-dot" style="background:#3498db"></span> pdfplumber</span>
  <span><span class="legend-dot" style="background:#e67e22"></span> OpenCV</span>
  <span><span class="legend-dot" style="background:#2ecc71"></span> Rich Extractor (PyMuPDF)</span>
  <span><span class="legend-dot" style="background:#9b59b6"></span> PaddleOCR</span>
</div>"""


def _run_record_html(run: RunRecord, embed: bool) -> str:
    improved = run.delta > 0
    cls = "run-record improved" if improved else "run-record"
    delta_str = f"+{run.delta:.1%}" if run.delta >= 0 else f"{run.delta:.1%}"
    params_json = json.dumps(run.parameters, indent=2)
    gaps_html = ""
    if run.gaps:
        rows = "".join(
            f"<tr><td>{g.estimated_type}</td><td>{g.severity}</td>"
            f"<td>{g.area_ratio:.1%}</td></tr>"
            for g in run.gaps
        )
        gaps_html = f"""
        <details><summary>Gaps found ({len(run.gaps)})</summary>
        <table><tr><th>Type</th><th>Severity</th><th>Area</th></tr>{rows}</table>
        </details>"""

    img_html = ""
    if run.gap_map_path:
        if embed:
            src = _b64_image(run.gap_map_path)
        else:
            src = run.gap_map_path.replace("\\", "/")
        if src:
            img_html = f'<img class="gap-map" src="{src}" alt="Gap map retry {run.run_number}">'

    return f"""
    <div class="{cls}">
      <strong>Retry {run.run_number}</strong> &nbsp;
      Score: {run.score_before:.1%} &rarr; <strong>{run.score_after:.1%}</strong>
      ({delta_str}) &nbsp; <em>{run.timestamp}</em>
      <details><summary>Parameters</summary><pre style="font-size:.8em;overflow:auto">{params_json}</pre></details>
      {gaps_html}
      {img_html}
    </div>"""


def _validation_section_html(page: PageResult) -> str:
    """Render the 5-layer validation section for a page."""
    val = page.validation
    if not val:
        return ""

    return f"""
      <h4>5-Layer Validation</h4>
      {_score_bar_html(val.coverage_score, "V1 Coverage")}
      {_score_bar_html(val.accuracy_score, "V2 Accuracy")}
      {_score_bar_html(val.completeness_score, "V3 Completeness")}
      {_score_bar_html(val.structural_score, "V4 Structural")}
      {_score_bar_html(val.composite_score, "V5 Composite")}
      <div style="margin-top:8px">
        <strong>Verdict:</strong>
        <span class="badge {'passed' if val.passed else 'unresolved'}">
          {'PASS' if val.passed else 'FAIL'} ({_pct(val.composite_score)})
        </span>
      </div>"""


def _confidence_gate_html(page: PageResult) -> str:
    """Render the OCR confidence gate section for scanned/hybrid pages."""
    gate = page.extraction.confidence_gate
    if not gate:
        return ""

    level_cls = gate.level.value
    return f"""
      <h4>OCR Confidence Gate</h4>
      <table style="max-width:400px">
        <tr><td>OCR Confidence</td>
            <td><strong style="color:{_score_color(gate.ocr_confidence)}">{_pct(gate.ocr_confidence)}</strong></td></tr>
        <tr><td>Level</td>
            <td><span class="badge {level_cls}">{gate.level.value.upper()}</span></td></tr>
        <tr><td>Total Words</td><td>{gate.word_count}</td></tr>
        <tr><td>High Confidence Words</td><td>{gate.high_confidence_words}</td></tr>
        <tr><td>Flagged Words</td><td>{gate.flagged_words}</td></tr>
        <tr><td>Qwen-VL Review</td>
            <td>{'Yes' if gate.needs_qwen_vl else 'No'}</td></tr>
        <tr><td>Human Review</td>
            <td>{'Yes' if gate.needs_human_review else 'No'}</td></tr>
      </table>"""


def _verification_status_html(page: PageResult) -> str:
    """Render the verification status badge."""
    status = page.extraction.verification_status
    cls = status.value.replace("_", "_")  # CSS class matches enum value
    label = status.value.replace("_", " ").title()
    return f'<span class="badge {status.value}">{label}</span>'


def _page_section_html(page: PageResult, embed: bool) -> str:
    status_cls = _status_class(page.status)
    page_type = page.extraction.page_type.value
    page_type_cls = page_type
    flag = " &#128681; Manual Review Required" if page.status == "unresolved" else ""

    runs_html = "".join(_run_record_html(r, embed) for r in page.run_records)

    initial_gap_maps = ""
    if page.gap_map_paths:
        path = page.gap_map_paths[0]
        src = _b64_image(path) if embed else path.replace("\\", "/")
        if src:
            initial_gap_maps = (
                f'<img class="gap-map" src="{src}" alt="Initial gap map">'
            )

    sc = page.source_contributions
    contrib_bar = _contribution_bar_html(sc)
    validation_html = _validation_section_html(page)
    confidence_gate_html = _confidence_gate_html(page)
    verification_badge = _verification_status_html(page)

    return f"""
  <details>
    <summary>
      Page {page.page_num + 1}
      &nbsp;<span class="badge {page_type_cls}">{page_type}</span>
      &nbsp;<span class="badge {status_cls}">{page.status.replace('_', ' ')}</span>
      &nbsp;{verification_badge}
      &nbsp;<strong>{_pct(page.final_score)}</strong>
      {flag}
    </summary>
    <div style="padding:8px 0">
      <div class="two-col">
        <div>
          {validation_html}
        </div>
        <div>
          <h4>Tool Contributions</h4>
          {contrib_bar}
          <table style="max-width:400px">
            <tr><th>Tool</th><th>Contribution</th></tr>
            <tr><td>pdfplumber</td><td>{_pct(sc.get('pdfplumber', 0.0))}</td></tr>
            <tr><td>OpenCV</td><td>{_pct(sc.get('opencv', 0.0))}</td></tr>
            <tr><td>Rich Extractor</td><td>{_pct(sc.get('rich_extractor', 0.0))}</td></tr>
            <tr><td>PaddleOCR</td><td>{_pct(sc.get('paddleocr', 0.0))}</td></tr>
            <tr style="font-weight:bold"><td>Total</td><td>{_pct(sc.get('total', 0.0))}</td></tr>
          </table>
          {confidence_gate_html}
        </div>
      </div>

      <h4>Page Metrics</h4>
      <table>
        <tr><th>Metric</th><th>Value</th></tr>
        <tr><td>Page Type</td><td><span class="badge {page_type_cls}">{page_type}</span></td></tr>
        <tr><td>Verification</td><td>{verification_badge}</td></tr>
        <tr><td>Initial Score</td><td>{_pct(page.initial_score)}</td></tr>
        <tr><td>Final Score</td><td><strong>{_pct(page.final_score)}</strong></td></tr>
        <tr><td>Retries Used</td><td>{len(page.run_records)}</td></tr>
        <tr><td>Tables Extracted</td><td>{len(page.extraction.tables)}</td></tr>
        <tr><td>Text Blocks</td><td>{len(page.extraction.text_blocks)}</td></tr>
        <tr><td>Images</td><td>{len(page.extraction.images)}</td></tr>
      </table>
      {initial_gap_maps}
      {runs_html}
    </div>
  </details>"""


def write_html(doc: DocumentResult, out_path: Path, embed_images: bool = True) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    d = doc.to_dict()
    summary = d["summary"]
    pdf_name = Path(doc.pdf_path).name

    page_types = summary.get("page_types", {})
    verification = summary.get("verification_status", {})
    conf = summary.get("confidence_breakdown", {})

    # Top-level stats
    stat_boxes = f"""
    <div class="stat-grid">
      <div class="stat-box">
        <div class="value" style="color:{_score_color(doc.overall_score)}">{_pct(doc.overall_score)}</div>
        <div class="label">Avg. Extraction Score</div></div>
      <div class="stat-box"><div class="value">{doc.total_pages}</div>
        <div class="label">Total Pages</div></div>
      <div class="stat-box"><div class="value" style="color:#27ae60">{summary['passed_pages']}</div>
        <div class="label">Passed</div></div>
      <div class="stat-box"><div class="value" style="color:#e74c3c">{summary['unresolved_pages']}</div>
        <div class="label">Unresolved</div></div>
    </div>"""

    # Page type and verification breakdown (side by side)
    breakdown_html = f"""
    <div class="two-col">
      <div>
        <h3>Page Types</h3>
        <table style="max-width:300px">
          <tr><th>Type</th><th>Pages</th></tr>
          <tr><td><span class="badge digital">Digital</span></td><td>{page_types.get('digital', 0)}</td></tr>
          <tr><td><span class="badge scanned">Scanned</span></td><td>{page_types.get('scanned', 0)}</td></tr>
          <tr><td><span class="badge hybrid">Hybrid</span></td><td>{page_types.get('hybrid', 0)}</td></tr>
        </table>

        <h3 style="margin-top:16px">Confidence Breakdown</h3>
        <table style="max-width:300px">
          <tr><th>Level</th><th>Pages</th></tr>
          <tr><td><span class="badge high">High</span></td><td>{conf.get('high', 0)}</td></tr>
          <tr><td><span class="badge medium">Medium</span></td><td>{conf.get('medium', 0)}</td></tr>
          <tr><td><span class="badge low">Low</span></td><td>{conf.get('low', 0)}</td></tr>
        </table>
      </div>
      <div>
        <h3>Verification Status</h3>
        <table style="max-width:350px">
          <tr><th>Status</th><th>Pages</th></tr>
          <tr><td><span class="badge auto_verified">Auto Verified</span></td>
              <td>{verification.get('auto_verified', 0)}</td></tr>
          <tr><td><span class="badge qwen_verified">Qwen-VL Verified</span></td>
              <td>{verification.get('qwen_verified', 0)}</td></tr>
          <tr><td><span class="badge qwen_corrected">Qwen-VL Corrected</span></td>
              <td>{verification.get('qwen_corrected', 0)}</td></tr>
          <tr><td><span class="badge pending_human">Pending Human Review</span></td>
              <td>{verification.get('pending_human_review', 0)}</td></tr>
        </table>
      </div>
    </div>"""

    # Tool contributions summary
    tc = summary.get("tool_contributions", {})
    avg_bar = _contribution_bar_html(tc) if tc else ""
    tool_section = ""
    if tc:
        tool_section = f"""
    <h3>Average Tool Contributions</h3>
    {_CONTRIB_LEGEND}
    {avg_bar}
    <table style="max-width:500px">
      <tr><th>Tool</th><th>Avg. Contribution</th></tr>
      <tr><td>pdfplumber</td><td>{_pct(tc.get('pdfplumber', 0.0))}</td></tr>
      <tr><td>OpenCV</td><td>{_pct(tc.get('opencv', 0.0))}</td></tr>
      <tr><td>Rich Extractor (PyMuPDF)</td><td>{_pct(tc.get('rich_extractor', 0.0))}</td></tr>
      <tr><td>PaddleOCR</td><td>{_pct(tc.get('paddleocr', 0.0))}</td></tr>
      <tr style="font-weight:bold"><td>Total</td><td>{_pct(tc.get('total', 0.0))}</td></tr>
    </table>"""

    # Average validation scores across all pages
    avg_val_html = ""
    val_pages = [p for p in doc.pages if p.validation]
    if val_pages:
        n = len(val_pages)
        avg_cov = sum(p.validation.coverage_score for p in val_pages) / n
        avg_acc = sum(p.validation.accuracy_score for p in val_pages) / n
        avg_comp = sum(p.validation.completeness_score for p in val_pages) / n
        avg_struct = sum(p.validation.structural_score for p in val_pages) / n
        avg_composite = sum(p.validation.composite_score for p in val_pages) / n

        avg_val_html = f"""
    <h3>Average Validation Scores ({n} pages)</h3>
    {_score_bar_html(avg_cov, "V1 Coverage")}
    {_score_bar_html(avg_acc, "V2 Accuracy")}
    {_score_bar_html(avg_comp, "V3 Completeness")}
    {_score_bar_html(avg_struct, "V4 Structural")}
    {_score_bar_html(avg_composite, "V5 Composite")}"""

    pages_html = "\n".join(
        _page_section_html(page, embed_images) for page in doc.pages
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TLS PDF Validation Report -- {pdf_name}</title>
  {_HTML_CSS}
</head>
<body>
<div class="container">
  <div class="card">
    <h1>PDF Validation Report</h1>
    <p><strong>File:</strong> {doc.pdf_path}</p>
    <p><strong>Generated:</strong> {doc.timestamp}</p>
    {stat_boxes}
  </div>
  <div class="card">
    {breakdown_html}
  </div>
  <div class="card">
    {avg_val_html}
    {tool_section}
  </div>
  <div class="card">
    <h2>Per-Page Details</h2>
    {_CONTRIB_LEGEND}
    {pages_html}
  </div>
</div>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info("HTML report -> %s", out_path)


# ── Accuracy Report (PDF) ───────────────────────────────────────────────────

# Validation dimension descriptions for the accuracy report key
_DIMENSION_KEY = [
    (
        "V1: Coverage (30%)",
        "Measures how much of the page's content area was successfully "
        "extracted. Compares the spatial extent of extracted text and tables "
        "against the baseline content area (excluding headers/footers).",
    ),
    (
        "V2: Accuracy (25%)",
        "Measures the correctness of extracted text by comparing it against "
        "the PDF's embedded text baseline. Evaluates character-level "
        "similarity, catching OCR misreads, garbled fonts, and encoding errors.",
    ),
    (
        "V3: Completeness (15%)",
        "Checks whether all content blocks present in the original page were "
        "captured. Identifies missing text blocks, tables, or images that "
        "were present in the source but absent from the extraction.",
    ),
    (
        "V4: Structural (15%)",
        "Evaluates whether the document's structure was preserved: reading "
        "order, table row/column alignment, heading hierarchy, and spatial "
        "relationships between elements.",
    ),
    (
        "V5: Cross-validation (15%)",
        "Compares results from multiple extraction tools (pdfplumber, "
        "PyMuPDF, OpenCV, OCR) against each other. High agreement across "
        "tools increases confidence; disagreement flags potential errors.",
    ),
]


def write_accuracy_pdf(doc: DocumentResult, out_path: Path) -> None:
    """Generate a formal accuracy report as a PDF using PyMuPDF."""
    import fitz

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_name = Path(doc.pdf_path).name
    pdf_doc = fitz.open()

    # ── Helpers ──────────────────────────────────────────────────────────
    PAGE_W, PAGE_H = 612, 792  # Letter size in points
    MARGIN = 50
    CONTENT_W = PAGE_W - 2 * MARGIN

    def new_page():
        return pdf_doc.new_page(width=PAGE_W, height=PAGE_H)

    def draw_text(page, x, y, text, fontsize=10, fontname="helv", color=(0, 0, 0)):
        page.insert_text(fitz.Point(x, y), text, fontsize=fontsize,
                         fontname=fontname, color=color)
        return y + fontsize + 4

    def draw_line(page, y, color=(0.8, 0.8, 0.8)):
        page.draw_line(fitz.Point(MARGIN, y), fitz.Point(PAGE_W - MARGIN, y),
                       color=color, width=0.5)
        return y + 8

    def check_page_break(page, y, needed=60):
        if y + needed > PAGE_H - MARGIN:
            page = new_page()
            return page, MARGIN + 20
        return page, y

    # ── Page 1: Title + Key ──────────────────────────────────────────────
    page = new_page()
    y = MARGIN + 10

    # Title
    y = draw_text(page, MARGIN, y, "Accuracy Report", fontsize=20, fontname="hebo")
    y = draw_text(page, MARGIN, y + 4, f"File: {pdf_name}", fontsize=10)
    y = draw_text(page, MARGIN, y, f"Generated: {doc.timestamp}", fontsize=10)
    y = draw_text(page, MARGIN, y,
                  f"Total Pages: {doc.total_pages}    "
                  f"Avg. Extraction Score: {_pct(doc.overall_score)}",
                  fontsize=10)
    y += 10
    y = draw_line(page, y, color=(0.3, 0.3, 0.3))

    # Dimension key
    y += 4
    y = draw_text(page, MARGIN, y, "Validation Dimensions", fontsize=14, fontname="hebo")
    y += 6

    for dim_name, dim_desc in _DIMENSION_KEY:
        page, y = check_page_break(page, y, needed=50)
        y = draw_text(page, MARGIN, y, dim_name, fontsize=10, fontname="hebo")
        y += 2
        # Word-wrap the description
        words = dim_desc.split()
        line = ""
        for word in words:
            test = f"{line} {word}".strip()
            if fitz.get_text_length(test, fontname="helv", fontsize=9) > CONTENT_W - 10:
                y = draw_text(page, MARGIN + 10, y, line, fontsize=9)
                line = word
            else:
                line = test
        if line:
            y = draw_text(page, MARGIN + 10, y, line, fontsize=9)
        y += 6

    y += 4
    y = draw_line(page, y, color=(0.3, 0.3, 0.3))
    y += 4
    y = draw_text(page, MARGIN, y, "Pass Threshold: 95% composite score",
                  fontsize=10, fontname="hebo", color=(0.15, 0.15, 0.15))

    # ── Per-page accuracy table ──────────────────────────────────────────
    page = new_page()
    y = MARGIN + 10
    y = draw_text(page, MARGIN, y, "Per-Page Accuracy Scores", fontsize=14, fontname="hebo")
    y += 8

    # Table header
    cols = [
        (MARGIN, 35, "Page"),
        (MARGIN + 38, 45, "Type"),
        (MARGIN + 86, 40, "Status"),
        (MARGIN + 132, 52, "Coverage"),
        (MARGIN + 188, 52, "Accuracy"),
        (MARGIN + 244, 58, "Complete"),
        (MARGIN + 306, 55, "Structural"),
        (MARGIN + 367, 60, "Composite"),
        (MARGIN + 432, 50, "Verdict"),
    ]

    def draw_table_header(page, y):
        for cx, cw, label in cols:
            draw_text(page, cx, y, label, fontsize=8, fontname="hebo")
        y += 12
        y = draw_line(page, y)
        return y

    y = draw_table_header(page, y)

    for pr in doc.pages:
        page, y = check_page_break(page, y, needed=16)
        if y == MARGIN + 20:
            y = draw_table_header(page, y)

        val = pr.validation
        page_type = pr.extraction.page_type.value[:3].upper()
        status = pr.extraction.verification_status.value.replace("_", " ")[:10]

        if val:
            cov = _pct(val.coverage_score)
            acc = _pct(val.accuracy_score)
            comp = _pct(val.completeness_score)
            struc = _pct(val.structural_score)
            composite = _pct(val.composite_score)
            verdict = "PASS" if val.passed else "FAIL"
            verdict_color = (0.15, 0.68, 0.38) if val.passed else (0.91, 0.30, 0.24)
        else:
            cov = acc = comp = struc = composite = "N/A"
            verdict = "N/A"
            verdict_color = (0.5, 0.5, 0.5)

        row_data = [
            (cols[0][0], str(pr.page_num + 1)),
            (cols[1][0], page_type),
            (cols[2][0], status),
            (cols[3][0], cov),
            (cols[4][0], acc),
            (cols[5][0], comp),
            (cols[6][0], struc),
            (cols[7][0], composite),
        ]

        for cx, text in row_data:
            draw_text(page, cx, y, text, fontsize=8)
        draw_text(page, cols[8][0], y, verdict, fontsize=8,
                  fontname="hebo", color=verdict_color)
        y += 14

    # ── Flagged pages (below threshold) with gap map overlays ────────────
    flagged_pages = [
        p for p in doc.pages
        if (p.validation and not p.validation.passed) or (not p.validation and not p.passed)
    ]
    if flagged_pages:
        page = new_page()
        y = MARGIN + 10
        y = draw_text(page, MARGIN, y, "Flagged Pages (Below 95% Threshold)",
                      fontsize=14, fontname="hebo")
        y += 8

        for pr in flagged_pages:
            page, y = check_page_break(page, y, needed=80)
            val = pr.validation

            y = draw_text(page, MARGIN, y,
                          f"Page {pr.page_num + 1} "
                          f"({pr.extraction.page_type.value})",
                          fontsize=10, fontname="hebo",
                          color=(0.91, 0.30, 0.24))

            if val:
                y = draw_text(page, MARGIN + 10, y,
                              f"Composite: {_pct(val.composite_score)}  |  "
                              f"Coverage: {_pct(val.coverage_score)}  |  "
                              f"Accuracy: {_pct(val.accuracy_score)}  |  "
                              f"Completeness: {_pct(val.completeness_score)}  |  "
                              f"Structural: {_pct(val.structural_score)}",
                              fontsize=8)

            v_status = pr.extraction.verification_status.value.replace("_", " ").title()
            y = draw_text(page, MARGIN + 10, y,
                          f"Verification: {v_status}  |  "
                          f"Retries: {len(pr.run_records)}",
                          fontsize=8)
            y += 4

            # Embed gap map overlay (green=extracted, red=missed)
            gap_map_file = None
            if pr.gap_map_paths:
                for gm in reversed(pr.gap_map_paths):
                    if gm and Path(gm).exists():
                        gap_map_file = gm
                        break
            if not gap_map_file and pr.run_records:
                for rr in reversed(pr.run_records):
                    if rr.gap_map_path and Path(rr.gap_map_path).exists():
                        gap_map_file = rr.gap_map_path
                        break

            if gap_map_file:
                try:
                    img_rect_w = CONTENT_W * 0.85
                    pix = fitz.Pixmap(gap_map_file)
                    iw, ih = pix.width, pix.height
                    pix = None
                except Exception:
                    iw, ih = 4, 3

                aspect = ih / iw if iw > 0 else 0.75
                img_h = min(img_rect_w * aspect, 300)
                if img_h < img_rect_w * aspect:
                    img_rect_w = img_h / aspect

                page, y = check_page_break(page, y, needed=img_h + 20)
                img_rect = fitz.Rect(
                    MARGIN + 10, y,
                    MARGIN + 10 + img_rect_w, y + img_h,
                )
                try:
                    page.insert_image(img_rect, filename=gap_map_file)
                    y += img_h + 8
                except Exception:
                    pass

            y += 6
            y = draw_line(page, y)

    pdf_doc.save(str(out_path), garbage=4, deflate=True)
    pdf_doc.close()
    logger.info("Accuracy PDF report -> %s", out_path)


# ── Main Generator ───────────────────────────────────────────────────────────

class ReportGenerator:
    """Generates all report formats from a DocumentResult."""

    def __init__(self, config: Dict[str, Any]) -> None:
        rep_cfg = config.get("reporting", {})
        self._formats: List[str] = rep_cfg.get("formats", ["json", "html", "csv"])
        self._embed: bool = bool(rep_cfg.get("embed_images_in_html", True))
        self._reports_dir = Path(config.get("output", {}).get("reports_dir", "reports"))

    def generate(self, doc: DocumentResult, pdf_stem: str) -> Dict[str, str]:
        """Write all configured report formats.

        Returns:
            Dict mapping format name -> output file path.
        """
        paths: Dict[str, str] = {}
        base = self._reports_dir / pdf_stem / pdf_stem

        if "json" in self._formats:
            p = base.with_suffix(".json")
            write_json(doc, p)
            paths["json"] = str(p)

        if "html" in self._formats:
            p = base.with_suffix(".html")
            write_html(doc, p, embed_images=self._embed)
            paths["html"] = str(p)

        if "csv" in self._formats:
            p = base.with_suffix(".csv")
            write_csv(doc, p)
            paths["csv"] = str(p)

        # Always generate accuracy report PDF
        p = base.parent / f"{pdf_stem}_accuracy.pdf"
        write_accuracy_pdf(doc, p)
        paths["accuracy_pdf"] = str(p)

        return paths
