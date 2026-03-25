"""Main pipeline orchestrator.

Branched architecture (v6): classifies each page as digital, scanned,
or hybrid, then routes to the appropriate extraction path.

  Stage 1  Page type detection (digital / scanned / hybrid)
  Stage 2  Path execution:
           - Digital:  pdfplumber + PyMuPDF -> coverage scoring -> retry loop
           - Scanned:  Tesseract OCR (CPU) + OpenCV -> confidence gate ->
                       inline Qwen-VL verification (GPU) if below threshold
           - Hybrid:   region split -> digital tools + scanned tools ->
                       inline Qwen-VL on scanned regions -> merge
  Stage 3  5-layer validation engine (coverage, accuracy, completeness,
           structural, cross-validation)
  Stage 4  Reporting (JSON + HTML + CSV) with validation metrics

Saves extracted content (text, tables, images) to organised output directories
and generates a Validation Report with per-page metrics and flagged items.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
import pandas as pd
from tqdm import tqdm

from .config_loader import default_params_from_config, load_config
from .layer1.custom_table_logic import CustomTableLogic, stitch_multipage_tables
from .layer1.opencv_extractor import OpenCVExtractor
from .layer1.pdfplumber_extractor import PDFPlumberExtractor
from .layer2.coverage_scorer import CoverageScorer
from .layer3.gap_analyzer import GapAnalyzer
from .layer3.parameter_adjuster import ParameterAdjuster
from .models import (
    DocumentResult,
    ExtractionParameters,
    PageExtractionResult,
    PageResult,
    PageType,
    VerificationStatus,
)
from .page_classifier import PageClassifier
from .paths import DigitalPathExecutor, ScannedPathExecutor, HybridPathExecutor
from .qwen_vl_verifier import QwenVLVerifier
from .reporting.report_generator import ReportGenerator
from .validation import ValidationEngine
from .image_intelligence import process_image_file

logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _safe_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", name)


# ─── Output Writers ───────────────────────────────────────────────────────────

def _save_page_output(
    result: PageExtractionResult,
    output_base: Path,
    pdf_path: str = "",
) -> None:
    """Write extracted text (Markdown + JSON), tables (CSV), and image files."""
    import json

    page_label = f"page_{result.page_num + 1:03d}"

    # ── Text ──────────────────────────────────────────────────────────────────
    text_dir = output_base / "text"
    text_dir.mkdir(parents=True, exist_ok=True)

    md_lines: List[str] = [f"# Page {result.page_num + 1}\n"]
    for block in result.text_blocks:
        md_lines.append(block.text)
        md_lines.append("")
    (text_dir / f"{page_label}.md").write_text(
        "\n".join(md_lines), encoding="utf-8"
    )

    text_meta = {
        "page_num": result.page_num,
        "source": result.source,
        "is_scanned": result.is_scanned,
        "page_type": result.page_type.value,
        "verification_status": result.verification_status.value,
        "num_text_blocks": len(result.text_blocks),
        "text_blocks": [b.to_dict() for b in result.text_blocks],
    }
    if result.confidence_gate:
        text_meta["confidence_gate"] = result.confidence_gate.to_dict()

    (text_dir / f"{page_label}_metadata.json").write_text(
        json.dumps(text_meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Tables ────────────────────────────────────────────────────────────────
    tables_dir = output_base / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for ti, table in enumerate(result.tables, start=1):
        tname = f"{page_label}_table_{ti:02d}"
        try:
            df = pd.DataFrame(table.data)
            if table.headers and len(table.headers) == df.shape[1]:
                df.columns = table.headers
            df.to_csv(tables_dir / f"{tname}.csv", index=False, encoding="utf-8")
        except Exception as e:
            logger.warning("Could not write CSV for %s: %s", tname, e)

        table_meta = table.to_dict()
        (tables_dir / f"{tname}_metadata.json").write_text(
            json.dumps(table_meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # ── Images: extract pixels + write metadata ───────────────────────────────
    if result.images:
        images_dir = output_base / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        if pdf_path:
            try:
                _extract_embedded_images(
                    pdf_path, result.page_num, images_dir, result.images, page_label
                )
            except Exception as e:
                logger.debug("Image pixel extraction failed p%d: %s", result.page_num, e)

        for img_elem in result.images:
            if img_elem.image_path and Path(img_elem.image_path).exists():
                try:
                    intel = process_image_file(
                        Path(img_elem.image_path),
                        run_ocr=True,
                        run_caption=False,
                        run_embedding=False,
                    )
                    img_elem.ocr_text = intel.get("ocr_text", "")
                except Exception as e:
                    logger.debug("OCR failed: %s", e)

        import json as _json
        img_meta = [img.to_dict() for img in result.images]
        (images_dir / f"{page_label}_images_metadata.json").write_text(
            _json.dumps(img_meta, indent=2), encoding="utf-8"
        )


def _extract_embedded_images(
    pdf_path: str,
    page_num: int,
    images_dir: Path,
    image_elements,
    page_label: str,
) -> None:
    """Extract embedded image pixels from the PDF page and save as PNG files."""
    doc = fitz.open(pdf_path)
    page = doc[page_num]

    img_list = page.get_images(full=True)

    saved_paths: list[Path] = []
    for img_index, img_info in enumerate(img_list):
        xref = img_info[0]
        try:
            rects = page.get_image_rects(xref)
            display_rect = rects[0] if rects else None

            img_data = doc.extract_image(xref)
            ext = img_data.get("ext", "png")
            img_bytes = img_data["image"]

            out_name = f"{page_label}_image_{img_index + 1:02d}.{ext}"
            out_path = images_dir / out_name
            out_path.write_bytes(img_bytes)
            saved_paths.append((out_path, display_rect))
        except Exception as exc:
            logger.debug("Could not extract image xref=%d p%d: %s", xref, page_num, exc)

    doc.close()

    for elem in image_elements:
        if elem.image_path:
            continue
        best_path = None
        best_iou = -1.0
        elem_bb = elem.bbox
        for (saved_path, display_rect) in saved_paths:
            if display_rect is None:
                if not best_path:
                    best_path = saved_path
                continue
            from .models import BoundingBox as _BB
            dr_bb = _BB(display_rect.x0, display_rect.y0, display_rect.x1, display_rect.y1)
            iou = elem_bb.iou(dr_bb)
            if iou > best_iou:
                best_iou = iou
                best_path = saved_path
        if best_path:
            elem.image_path = str(best_path)

    unmatched_elems = [e for e in image_elements if not e.image_path]
    unmatched_paths = [p for p, _ in saved_paths if not any(e.image_path == str(p) for e in image_elements)]
    for elem, path in zip(unmatched_elems, unmatched_paths):
        elem.image_path = str(path)


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class PDFPipeline:
    """Branched PDF extraction and validation pipeline.

    Classifies each page as digital/scanned/hybrid, routes to the
    appropriate extraction path, then validates with the 5-layer engine.
    """

    def __init__(self, config_path: str = "config/pipeline_config.yaml") -> None:
        self.config = load_config(config_path)
        self._threshold: float = float(self.config.get("accuracy_threshold", 0.95))

        # Page classifier (Stage 1)
        self._classifier = PageClassifier(self.config)

        # Extractors (shared across paths)
        self._pp = PDFPlumberExtractor(self.config)
        self._cv = OpenCVExtractor(self.config)
        self._custom = CustomTableLogic(self.config)
        self._scorer = CoverageScorer(self.config)
        self._gap_analyzer = GapAnalyzer(self.config)
        self._adjuster = ParameterAdjuster(self.config)

        # Qwen-VL verifier (GPU, runs inline during scanned/hybrid extraction)
        self._qwen_verifier = QwenVLVerifier(self.config)

        # Path executors (Stage 2)
        self._digital_path = DigitalPathExecutor(
            self.config,
            self._pp,
            self._custom,
            self._scorer,
            self._gap_analyzer,
            self._adjuster,
        )
        self._scanned_path = ScannedPathExecutor(
            self.config,
            self._cv,
            self._scorer,
            qwen_verifier=self._qwen_verifier,
        )
        self._hybrid_path = HybridPathExecutor(
            self.config,
            self._pp,
            self._cv,
            self._custom,
            self._scorer,
            qwen_verifier=self._qwen_verifier,
        )

        # Validation engine (Stage 3)
        self._validator = ValidationEngine(self.config)

        # Reporting (Stage 4)
        self._reporter = ReportGenerator(self.config)

        out_cfg = self.config.get("output", {})
        self._base_out = Path(out_cfg.get("base_dir", "output"))
        self._reports_dir = Path(out_cfg.get("reports_dir", "reports"))

    # ── Public API ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> DocumentResult:
        """Run the full pipeline on a single PDF.

        Returns the DocumentResult and writes all output files.
        """
        pipeline_start = time.perf_counter()
        stage_times: Dict[str, float] = {}

        pdf_path = str(Path(pdf_path).resolve())
        pdf_stem = _safe_filename(Path(pdf_path).stem)
        output_base = self._base_out / pdf_stem

        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        print(f"\n{'='*60}")
        print(f"  Processing: {Path(pdf_path).name} ({total_pages} pages)")
        print(f"{'='*60}\n")

        # Stage 1: Classify all pages upfront
        t0 = time.perf_counter()
        classifications = self._classifier.classify_document(pdf_path)
        stage_times["Classification"] = time.perf_counter() - t0

        # Log classification summary
        type_counts = {t: 0 for t in PageType}
        for c in classifications:
            type_counts[c.page_type] += 1
        print(
            f"  Page types: {type_counts[PageType.DIGITAL]} digital, "
            f"{type_counts[PageType.SCANNED]} scanned, "
            f"{type_counts[PageType.HYBRID]} hybrid "
            f"({stage_times['Classification']:.1f}s)\n"
        )

        default_params = default_params_from_config(self.config)
        page_results: List[PageResult] = []
        all_flagged_items: List[Dict[str, Any]] = []

        # Stage 2 + 3: Extract and validate each page
        t0 = time.perf_counter()
        page_bar = tqdm(
            range(total_pages),
            desc="  Extracting & validating",
            unit="page",
            bar_format="  {desc}: {bar:30} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
        )
        for page_num in page_bar:
            classification = classifications[page_num]
            page_bar.set_postfix_str(
                f"p{page_num + 1} [{classification.page_type.value}]"
            )

            # Stage 2: Route to appropriate path
            page_result = self._process_page(
                pdf_path, page_num, classification, default_params, output_base,
            )

            # Stage 3: Validation
            validation = self._validator.validate(
                pdf_path, page_num, page_result.extraction,
            )
            page_result.validation = validation
            page_result.final_score = validation.composite_score
            page_result.passed = validation.passed

            # Collect flagged items
            flagged = self._validator.get_flagged_items(
                pdf_path, page_num, page_result.extraction, validation,
            )
            all_flagged_items.extend(flagged)

            page_results.append(page_result)
            _save_page_output(page_result.extraction, output_base, pdf_path)

        stage_times["Extraction + Validation"] = time.perf_counter() - t0

        # Multi-page table stitching
        t0 = time.perf_counter()
        extractions = [r.extraction for r in page_results]
        stitch_multipage_tables(extractions)
        stage_times["Table stitching"] = time.perf_counter() - t0

        # Save flagged items
        if all_flagged_items:
            self._save_flagged_items(all_flagged_items, output_base)

        overall = (
            sum(r.final_score for r in page_results) / len(page_results)
            if page_results
            else 0.0
        )

        doc_result = DocumentResult(
            pdf_path=pdf_path,
            total_pages=total_pages,
            pages=page_results,
            timestamp=datetime.now(timezone.utc).isoformat(),
            overall_score=overall,
        )

        # Stage 4: Reporting
        t0 = time.perf_counter()
        report_paths = self._reporter.generate(doc_result, pdf_stem)
        stage_times["Reporting"] = time.perf_counter() - t0

        total_time = time.perf_counter() - pipeline_start

        # Print summary
        print(f"\n{'='*60}")
        print(f"  PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  Overall score:  {overall * 100:.1f}%")
        print(f"  Flagged items:  {len(all_flagged_items)}")
        print(f"  Total time:     {total_time:.1f}s")
        print(f"\n  Stage breakdown:")
        for stage, elapsed in stage_times.items():
            pct = (elapsed / total_time * 100) if total_time > 0 else 0
            print(f"    {stage:<28} {elapsed:>6.1f}s  ({pct:.0f}%)")
        print(f"\n  Reports: {report_paths}")
        print(f"  Output:  {output_base}")
        print(f"{'='*60}\n")

        return doc_result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _process_page(
        self,
        pdf_path: str,
        page_num: int,
        classification,
        default_params: ExtractionParameters,
        output_base: Path,
    ) -> PageResult:
        """Route a page to the appropriate extraction path."""
        if classification.page_type == PageType.DIGITAL:
            return self._digital_path.execute(
                pdf_path, page_num, classification, default_params, output_base,
            )
        elif classification.page_type == PageType.SCANNED:
            return self._scanned_path.execute(
                pdf_path, page_num, classification, default_params, output_base,
            )
        else:  # HYBRID
            return self._hybrid_path.execute(
                pdf_path, page_num, classification, default_params, output_base,
            )

    def _save_flagged_items(
        self, flagged_items: List[Dict[str, Any]], output_base: Path
    ) -> None:
        """Save flagged items to a JSON file for review."""
        import json
        flagged_dir = output_base / "validation"
        flagged_dir.mkdir(parents=True, exist_ok=True)
        (flagged_dir / "flagged_items.json").write_text(
            json.dumps(flagged_items, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(
            "Saved %d flagged items to %s",
            len(flagged_items),
            flagged_dir / "flagged_items.json",
        )
