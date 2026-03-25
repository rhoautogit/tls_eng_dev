"""Layer 3 – Retry Controller.

Manages the per-page retry loop:
  1. Run gap analysis on the current best extraction.
  2. Obtain new parameters from ParameterAdjuster.
  3. Re-extract with both tools (parallel) and merge.
  4. Re-score.
  5. Merge retry result with previous best (respecting target region).
  6. Record the run.
  7. Stop if score ≥ threshold, plateau (Δ < 1% for 2 consecutive retries),
     or max retries reached.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..layer1.opencv_extractor import OpenCVExtractor
from ..layer1.pdfplumber_extractor import PDFPlumberExtractor
from ..layer1.result_merger import ResultMerger
from ..layer2.coverage_scorer import CoverageScorer
from ..models import ExtractionParameters, PageExtractionResult, RunRecord
from .gap_analyzer import GapAnalyzer
from .parameter_adjuster import ParameterAdjuster

logger = logging.getLogger(__name__)


class RetryController:
    """Runs the Layer 3 feedback loop for a single page."""

    def __init__(
        self,
        config: Dict[str, Any],
        pdfplumber_extractor: PDFPlumberExtractor,
        opencv_extractor: OpenCVExtractor,
        merger: ResultMerger,
        scorer: CoverageScorer,
        gap_analyzer: GapAnalyzer,
        param_adjuster: ParameterAdjuster,
    ) -> None:
        self.config = config
        self._pp = pdfplumber_extractor
        self._cv = opencv_extractor
        self._merger = merger
        self._scorer = scorer
        self._gap_analyzer = gap_analyzer
        self._adjuster = param_adjuster

        self._max_retries: int = int(config.get("max_retries", 3))
        self._threshold: float = float(config.get("accuracy_threshold", 0.95))
        self._early_stop_delta: float = float(
            config.get("early_termination_threshold", 0.01)
        )

    # ── Public ────────────────────────────────────────────────────────────────

    def process_page(
        self,
        pdf_path: str,
        page_num: int,
        initial_extraction: PageExtractionResult,
        initial_score: float,
        initial_params: ExtractionParameters,
        gap_map_dir: Path,
    ) -> Tuple[PageExtractionResult, float, List[RunRecord], List[str]]:
        """Run the retry loop.

        Returns:
            (best_extraction, best_score, run_records, gap_map_paths)
        """
        best_extraction = initial_extraction
        best_score = initial_score
        current_params = initial_params
        run_records: List[RunRecord] = []
        gap_map_paths: List[str] = []
        recent_deltas: List[float] = []

        for retry_num in range(1, self._max_retries + 1):
            logger.info(
                "Page %d – retry %d/%d (score=%.3f)",
                page_num, retry_num, self._max_retries, best_score,
            )

            # ── Gap analysis ──────────────────────────────────────────────────
            gaps, gap_map_path = self._gap_analyzer.analyze(
                pdf_path, page_num, best_extraction, gap_map_dir, retry_num
            )
            gap_map_paths.append(gap_map_path)

            # ── New parameters ────────────────────────────────────────────────
            retry_params = self._adjuster.get_params(
                retry_num, gaps, current_params
            )

            # ── Parallel extraction ───────────────────────────────────────────
            with ThreadPoolExecutor(max_workers=2) as executor:
                pp_future = executor.submit(
                    self._pp.extract_page, pdf_path, page_num, retry_params
                )
                cv_future = executor.submit(
                    self._cv.extract_page, pdf_path, page_num, retry_params
                )
                try:
                    pp_result = pp_future.result()
                except Exception as e:
                    logger.warning("pdfplumber retry %d failed: %s", retry_num, e)
                    pp_result = best_extraction  # fall back
                try:
                    cv_result = cv_future.result()
                except Exception as e:
                    logger.warning("OpenCV retry %d failed: %s", retry_num, e)
                    cv_result = best_extraction

            # ── Merge retry result then stitch with previous best ─────────────
            merged_retry = self._merger.merge(pp_result, cv_result)
            combined = self._merger.merge_with_previous(
                best_extraction, merged_retry, retry_params.target_bbox
            )

            # ── Score ─────────────────────────────────────────────────────────
            new_score = self._scorer.score_page(pdf_path, page_num, combined)
            delta = new_score - best_score

            # ── Record ────────────────────────────────────────────────────────
            run_records.append(
                RunRecord(
                    run_number=retry_num,
                    parameters=retry_params.to_dict(),
                    score_before=best_score,
                    score_after=new_score,
                    delta=delta,
                    gap_map_path=gap_map_path,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    gaps=gaps,
                )
            )

            # Update best if improved
            if new_score > best_score:
                best_score = new_score
                best_extraction = combined
                current_params = retry_params

            # ── Termination checks ────────────────────────────────────────────
            if best_score >= self._threshold:
                logger.info(
                    "Page %d – passed at retry %d (score=%.3f)",
                    page_num, retry_num, best_score,
                )
                break

            recent_deltas.append(delta)
            if len(recent_deltas) >= 2:
                if all(d < self._early_stop_delta for d in recent_deltas[-2:]):
                    logger.info(
                        "Page %d – early termination at retry %d "
                        "(Δ=%.4f, %.4f < %.3f)",
                        page_num, retry_num, recent_deltas[-2],
                        recent_deltas[-1], self._early_stop_delta,
                    )
                    break

        return best_extraction, best_score, run_records, gap_map_paths
