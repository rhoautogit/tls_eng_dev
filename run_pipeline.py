#!/usr/bin/env python3
"""CLI entry point for the TLS PDF Ingestion & Evaluation Pipeline.

Usage
-----
Single PDF:
  python run_pipeline.py path/to/document.pdf

Directory of PDFs:
  python run_pipeline.py path/to/pdf_folder/

Custom config:
  python run_pipeline.py path/to/document.pdf --config config/my_config.yaml

Options
-------
  --config PATH   Path to YAML config file (default: config/pipeline_config.yaml)
  --log-level     Logging verbosity: DEBUG | INFO | WARNING (default: INFO)
  --workers N     Number of PDFs to process in parallel (default: 1)
"""
from __future__ import annotations

import os
# PaddlePaddle GPU memory: must be set BEFORE importing paddle.
# auto_growth allocates on demand instead of pre-allocating a huge pool,
# allowing empty_cache() to actually release memory between flushes.
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.7")

import argparse
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )


def _process_one(pdf_path: str, config_path: str, log_level: str = "INFO") -> dict:
    """Worker function -- runs in a subprocess so GPU memory is freed on exit."""
    import logging as _logging
    _logging.basicConfig(
        level=getattr(_logging, log_level.upper(), _logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    _log = _logging.getLogger("worker")
    _log.info("Starting PDF: %s", pdf_path)

    from src.pipeline import PDFPipeline

    pipeline = PDFPipeline(config_path)
    result = pipeline.process_pdf(pdf_path)
    return {
        "pdf": pdf_path,
        "pages": result.total_pages,
        "overall_score": result.overall_score,
        "passed": sum(1 for p in result.pages if p.passed),
        "unresolved": sum(1 for p in result.pages if p.status == "unresolved"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="TLS PDF Ingestion & Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Path to a single PDF file or a directory containing PDFs.",
    )
    parser.add_argument(
        "--config",
        default="config/pipeline_config.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of PDFs to process concurrently (default: 1).",
    )
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_pipeline")

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input path does not exist: %s", input_path)
        return 1

    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error("Input file is not a PDF: %s", input_path)
            return 1
        pdf_files = [input_path]
    else:
        pdf_files = sorted(input_path.glob("**/*.pdf"))
        if not pdf_files:
            logger.error("No PDF files found in: %s", input_path)
            return 1

    logger.info("Found %d PDF(s) to process.", len(pdf_files))
    config_path = str(Path(args.config).resolve())

    results = []
    errors = []

    if args.workers == 1 or len(pdf_files) == 1:
        for pdf in pdf_files:
            try:
                res = _process_one(str(pdf), config_path, args.log_level)
                results.append(res)
                logger.info(
                    "ok %s  score=%.1f%%  passed=%d/%d  unresolved=%d",
                    res["pdf"], res["overall_score"] * 100,
                    res["passed"], res["pages"], res["unresolved"],
                )
            except Exception as exc:
                logger.error("FAIL %s  ERROR: %s", pdf, exc, exc_info=True)
                errors.append((str(pdf), str(exc)))
            finally:
                # Release PaddleOCR engine between PDFs to free GPU memory
                try:
                    from src.paddle_ocr_engine import PaddleOCREngine
                    PaddleOCREngine.release_shared()
                except Exception:
                    pass
                import gc
                gc.collect()
                try:
                    import paddle
                    if paddle.device.is_compiled_with_cuda():
                        paddle.device.cuda.empty_cache()
                except Exception:
                    pass
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one, str(pdf), config_path, args.log_level): pdf
                for pdf in pdf_files
            }
            for future in as_completed(futures):
                pdf = futures[future]
                try:
                    res = future.result()
                    results.append(res)
                    logger.info(
                        "ok %s  score=%.1f%%  passed=%d/%d  unresolved=%d",
                        res["pdf"], res["overall_score"] * 100,
                        res["passed"], res["pages"], res["unresolved"],
                    )
                except Exception as exc:
                    logger.error("FAIL %s  ERROR: %s", pdf, exc, exc_info=True)
                    errors.append((str(pdf), str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Processed : {len(results) + len(errors)} PDFs")
    print(f"  Succeeded : {len(results)}")
    print(f"  Failed    : {len(errors)}")
    if results:
        avg_score = sum(r["overall_score"] for r in results) / len(results)
        total_unresolved = sum(r["unresolved"] for r in results)
        print(f"  Avg score : {avg_score:.1%}")
        print(f"  Unresolved pages : {total_unresolved}")
    if errors:
        print("\n  Errors:")
        for path, msg in errors:
            print(f"    {path}: {msg}")
    print("=" * 60)

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
