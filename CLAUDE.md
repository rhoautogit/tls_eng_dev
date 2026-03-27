# TLS Engineering PDF Pipeline

## Project Overview
Three-layer PDF ingestion and evaluation pipeline for TLS Engineering.
Processes ~500 engineering PDFs/quarter. No external API calls (data compliance).
All processing is local -- PaddleOCR (GPU) for OCR, Ollama for Qwen-VL, no cloud services.

## Quick Start
```bash
pip install -r requirements.txt
python run_pipeline.py path/to/file.pdf
python run_pipeline.py path/to/folder/ --workers 4
pytest tests/ -v
```

## Architecture (Pipeline v7)

Pipeline stages: Input -> Stage 1 (Page Classification) -> Stage 2 (Path Execution, includes inline Qwen-VL) -> Stage 3 (5-Layer Validation) -> Stage 4 (Reporting)

### Entry Point
- `run_pipeline.py` -- CLI entry point
- `src/pipeline.py` -- main orchestrator (PDFPipeline class)

### Shared
- `src/models.py` -- all shared dataclasses (BoundingBox, Table, Gap, RunRecord, etc.)
- `src/config_loader.py` -- YAML config loader + default params builder
- `config/pipeline_config.yaml` -- all tunable thresholds/modes

### Stage 1: Page Classification
- `src/page_classifier.py` -- classifies pages as digital, scanned, or hybrid (text density + image coverage + heuristic rules)

### Stage 2: Path Execution
Three branched paths based on page classification:

**Digital Path**
- `src/paths/digital_path.py` -- digital PDF processing
- `src/layer1/pdfplumber_extractor.py` -- text + table extraction (lattice/stream modes)
- `src/layer2/coverage_scorer.py` -- PyMuPDF baseline coverage check
- Rich extractor fallback if pdfplumber text is garbled (< 80% similarity)
- Retry loop (max 3): switch mode+loosen, CLAHE+sharpen, full combined

**Scanned Path**
- `src/paths/scanned_path.py` -- scanned PDF processing
- OpenCV preprocessing (gray + CLAHE + threshold) -> PaddleOCR (GPU) -> confidence gate
- Confidence gate: HIGH (>= 95%), MEDIUM (85-95%), LOW (< 85%)
- OCR retry loop (max 3): DPI 400 -> 450 -> 600 with escalating CLAHE + adaptive binarize + morphological cleanup
- Early stop if delta < 1% for 2 consecutive retries
- If confidence still below threshold after retries, runs Qwen-VL (GPU) inline to verify/correct
- Output statuses: AUTO_VERIFIED, QWEN_VERIFIED, QWEN_CORRECTED, or PENDING_HUMAN

**Hybrid Path**
- `src/paths/hybrid_path.py` -- hybrid page processing
- `src/paths/region_splitter.py` -- splits page into digital + scanned regions
- Digital regions -> pdfplumber, scanned regions -> PaddleOCR (GPU)
- Confidence gate on scanned regions, retry loop reuses scanned path strategies
- If scanned region confidence still below threshold, runs Qwen-VL (GPU) inline on scanned regions
- Merge + dedup results

**Shared Extraction Modules**
- `src/paddle_ocr_engine.py` -- PaddleOCR engine wrapper (GPU, shared singleton with periodic memory flush)
- `src/layer1/opencv_extractor.py` -- image-based grid detection via morphological ops
- `src/layer1/result_merger.py` -- parallel results merged; digital prefers pdfplumber (0.7w), scanned prefers opencv (0.7w)
- `src/layer1/custom_table_logic.py` -- implicit table detection, merged cells, nested headers, multi-page stitch
- `src/ocr_rich_extractor.py` -- OCR extraction with rich metadata
- `src/rich_extractor.py` -- rich extraction for digital pages
- `src/image_intelligence.py` -- image captioning/classification (placeholder for Phase 2)

### Stage 3: 5-Layer Validation
- `src/validation/validation_engine.py` -- composite scoring:
  - V1: Coverage (30%) -- spatial coverage
  - V2: Accuracy (25%) -- text correctness
  - V3: Completeness (15%) -- missing content inventory
  - V4: Structural (15%) -- document structure integrity
  - V5: Cross-validation (15%) -- cross-source agreement
- **Pass threshold: 95% composite score**

### Qwen-VL Verifier (inline, GPU)
- `src/qwen_vl_verifier.py` -- called inline by scanned/hybrid paths when OCR confidence is below threshold
- PaddleOCR runs on GPU, Qwen-VL runs on GPU via Ollama
- Renders page as PNG (200 DPI) -> sends image + OCR text to Qwen3-VL (8B) via Ollama -> JSON corrections
- Output statuses: `QWEN_VERIFIED`, `QWEN_CORRECTED`, or `PENDING_HUMAN`

### Stage 4: Reporting
- `src/reporting/report_generator.py` -- JSON + HTML dashboard + CSV + accuracy PDF per document

### Layer 2 -- Coverage Scoring
- `src/layer2/coverage_scorer.py` -- PyMuPDF baseline (excl. header 8% + footer 8% + page numbers)
- Score = extracted_chars / baseline_chars, capped at 1.0

### Layer 3 -- Visual Twin + Retry Loop
- `src/layer3/visual_twin.py` -- renders extraction twin (green=text, blue=table, yellow=image)
- `src/layer3/gap_analyzer.py` -- content_mask AND NOT covered_mask -> gap regions -> type + severity
- `src/layer3/parameter_adjuster.py` -- retry parameter adjustments per strategy
- `src/layer3/retry_controller.py` -- max 3 retries, early stop if delta<1% for 2 consecutive

## Output Structure
```
output/{pdf_stem}/text/          -- .md + metadata .json per page
output/{pdf_stem}/tables/        -- .csv + metadata .json per table
output/{pdf_stem}/images/        -- PNG + metadata .json
output/{pdf_stem}/gap_maps/      -- PNG gap maps (green=extracted, red=missed)
output/{pdf_stem}/rich/          -- rich visual JSON per page
output/{pdf_stem}/verification/  -- Qwen-VL results
output/{pdf_stem}/validation/    -- flagged items
reports/{pdf_stem}/{pdf_stem}.json/.html/.csv
reports/{pdf_stem}/{pdf_stem}_accuracy.pdf
```

## Coordinate Systems
- Both pdfplumber and fitz use top-left origin (y increases downward)
- pdfplumber: (x0, top, x1, bottom); fitz: (x0, y0, x1, y1) -- same semantics

## Key Thresholds (in pipeline_config.yaml)
- Validation pass threshold: 95% (composite score)
- Coverage accuracy threshold: 95% (per-page extraction)
- Max retries: 3
- Digital page: >= 100 embedded chars
- Scanned page: < 20 embedded chars
- OCR confidence gate: HIGH >= 95%, MEDIUM >= 85%, LOW < 85%
- GPU memory flush: every 50 OCR pages

## Style Rules
- Do not use em dashes in any output text. Use commas, periods, or parentheses instead.
- Do not add Co-Authored-By lines in git commit messages.

## Tests
```bash
pytest tests/ -v
```
Test files mirror the source structure: `test_layer1.py`, `test_layer2.py`, `test_layer3.py`, `test_page_classifier.py`, `test_paths.py`, `test_qwen_vl.py`, `test_reporting.py`, `test_validation.py`.
