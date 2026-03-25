"""Digital path executor.

Handles pages with embedded text layers (digital PDFs).
Uses pdfplumber for text/table extraction and PyMuPDF Rich Extractor
for detailed visual state. OpenCV is NOT used on digital pages.

Flow:
  pdfplumber -> Custom Table Logic -> Rich Extractor (PyMuPDF)
  -> Coverage Scoring -> Visual Twin retry if < 95% -> Output
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..layer1.custom_table_logic import CustomTableLogic
from ..layer1.pdfplumber_extractor import PDFPlumberExtractor
from ..layer2.coverage_scorer import CoverageScorer
from ..layer3.gap_analyzer import GapAnalyzer
from ..layer3.parameter_adjuster import ParameterAdjuster
from ..models import (
    ExtractionParameters,
    PageClassification,
    PageExtractionResult,
    PageResult,
    PageType,
    RunRecord,
    VerificationStatus,
)
from ..rich_extractor import extract_rich_page, save_rich_page

logger = logging.getLogger(__name__)


def _extract_rich_text(rich_data: Dict[str, Any]) -> str:
    """Extract plain text from rich visual JSON data."""
    parts: List[str] = []
    for block in rich_data.get("text_blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    parts.append(text)
    return " ".join(parts)


class DigitalPathExecutor:
    """Executes the digital extraction path for a single page."""

    def __init__(
        self,
        config: Dict[str, Any],
        pdfplumber_extractor: PDFPlumberExtractor,
        custom_table_logic: CustomTableLogic,
        scorer: CoverageScorer,
        gap_analyzer: GapAnalyzer,
        param_adjuster: ParameterAdjuster,
    ) -> None:
        self.config = config
        self._pp = pdfplumber_extractor
        self._custom = custom_table_logic
        self._scorer = scorer
        self._gap_analyzer = gap_analyzer
        self._adjuster = param_adjuster

        self._threshold: float = float(config.get("accuracy_threshold", 0.95))
        self._max_retries: int = int(config.get("max_retries", 3))
        self._early_stop_delta: float = float(
            config.get("early_termination_threshold", 0.01)
        )

    def execute(
        self,
        pdf_path: str,
        page_num: int,
        classification: PageClassification,
        params: ExtractionParameters,
        output_base: Path,
    ) -> PageResult:
        """Run the digital extraction path.

        1. pdfplumber extraction
        2. Custom table logic
        3. Rich extraction (PyMuPDF)
        4. Coverage scoring
        5. Retry loop if below threshold
        """
        # -- Step 1: pdfplumber extraction --
        try:
            extraction = self._pp.extract_page(pdf_path, page_num, params)
        except Exception as e:
            logger.warning("pdfplumber failed p%d: %s", page_num + 1, e)
            extraction = PageExtractionResult(
                page_num=page_num, text_blocks=[], tables=[],
                images=[], source="pdfplumber",
            )

        # -- Step 2: Custom table logic --
        extraction = self._custom.process(extraction)

        # Tag as digital
        extraction.page_type = PageType.DIGITAL
        extraction.is_scanned = False

        # -- Step 3: Rich extraction (PyMuPDF) --
        rich_text = ""
        try:
            rich_data = extract_rich_page(pdf_path, page_num)
            save_rich_page(rich_data, output_base / "rich", page_num)
            rich_text = _extract_rich_text(rich_data)
        except Exception as e:
            logger.warning("Rich extraction failed p%d: %s", page_num + 1, e)

        # -- Step 4: Coverage scoring --
        detailed = self._scorer.score_page_detailed(
            pdf_path, page_num, extraction, rich_text
        )
        combined_score = detailed["total_score"]
        layer1_score = self._scorer.score_page(pdf_path, page_num, extraction)

        contributions = {
            "pdfplumber": detailed["pdfplumber_pct"],
            "rich_extractor": detailed["rich_pct"],
            "total": round(combined_score, 4),
        }

        logger.info(
            "Page %d [DIGITAL] score: %.1f%% (pdfplumber=%.1f%% | rich=%.1f%%)",
            page_num + 1,
            combined_score * 100,
            detailed["pdfplumber_pct"] * 100,
            detailed["rich_pct"] * 100,
        )

        # Digital pages above threshold are auto-verified
        if combined_score >= self._threshold:
            extraction.verification_status = VerificationStatus.AUTO_VERIFIED
            return PageResult(
                page_num=page_num,
                final_score=combined_score,
                initial_score=layer1_score,
                passed=True,
                extraction=extraction,
                run_records=[],
                gap_map_paths=[],
                status="passed_initial",
                source_contributions=contributions,
                classification=classification,
            )

        # -- Step 5: Retry loop (pdfplumber only, no OpenCV) --
        best_extraction, best_score, run_records, gap_map_paths = \
            self._retry_loop(
                pdf_path, page_num, extraction, layer1_score, params, output_base,
            )

        # Re-score after retries
        detailed_after = self._scorer.score_page_detailed(
            pdf_path, page_num, best_extraction, rich_text
        )
        best_combined = detailed_after["total_score"]

        contributions_after = {
            "pdfplumber": detailed_after["pdfplumber_pct"],
            "rich_extractor": detailed_after["rich_pct"],
            "total": round(best_combined, 4),
        }

        passed = best_combined >= self._threshold
        status = "resolved" if passed else "unresolved"

        if passed:
            best_extraction.verification_status = VerificationStatus.AUTO_VERIFIED
        else:
            logger.warning(
                "Page %d [DIGITAL] UNRESOLVED after %d retries (best=%.1f%%)",
                page_num + 1, len(run_records), best_combined * 100,
            )

        return PageResult(
            page_num=page_num,
            final_score=best_combined,
            initial_score=layer1_score,
            passed=passed,
            extraction=best_extraction,
            run_records=run_records,
            gap_map_paths=gap_map_paths,
            status=status,
            source_contributions=contributions_after,
            classification=classification,
        )

    def _retry_loop(
        self,
        pdf_path: str,
        page_num: int,
        initial_extraction: PageExtractionResult,
        initial_score: float,
        initial_params: ExtractionParameters,
        output_base: Path,
    ) -> Tuple[PageExtractionResult, float, List[RunRecord], List[str]]:
        """Retry loop for digital pages -- pdfplumber only."""
        from datetime import datetime, timezone

        best_extraction = initial_extraction
        best_score = initial_score
        current_params = initial_params
        run_records: List[RunRecord] = []
        gap_map_paths: List[str] = []
        recent_deltas: List[float] = []
        gap_map_dir = output_base / "gap_maps"

        for retry_num in range(1, self._max_retries + 1):
            logger.info(
                "Page %d [DIGITAL] retry %d/%d (score=%.3f)",
                page_num + 1, retry_num, self._max_retries, best_score,
            )

            # Gap analysis
            gaps, gap_map_path = self._gap_analyzer.analyze(
                pdf_path, page_num, best_extraction, gap_map_dir, retry_num
            )
            gap_map_paths.append(gap_map_path)

            # Adjusted parameters
            retry_params = self._adjuster.get_params(
                retry_num, gaps, current_params
            )

            # Re-extract with pdfplumber only
            try:
                pp_result = self._pp.extract_page(pdf_path, page_num, retry_params)
            except Exception as e:
                logger.warning("pdfplumber retry %d failed: %s", retry_num, e)
                pp_result = best_extraction

            retry_extraction = self._custom.process(pp_result)
            retry_extraction.page_type = PageType.DIGITAL

            # Score
            new_score = self._scorer.score_page(pdf_path, page_num, retry_extraction)
            delta = new_score - best_score

            run_records.append(RunRecord(
                run_number=retry_num,
                parameters=retry_params.to_dict(),
                score_before=best_score,
                score_after=new_score,
                delta=delta,
                gap_map_path=gap_map_path,
                timestamp=datetime.now(timezone.utc).isoformat(),
                gaps=gaps,
            ))

            if new_score > best_score:
                best_score = new_score
                best_extraction = retry_extraction
                current_params = retry_params

            # Termination checks
            if best_score >= self._threshold:
                logger.info(
                    "Page %d [DIGITAL] passed at retry %d (score=%.3f)",
                    page_num + 1, retry_num, best_score,
                )
                break

            recent_deltas.append(delta)
            if len(recent_deltas) >= 2:
                if all(d < self._early_stop_delta for d in recent_deltas[-2:]):
                    logger.info(
                        "Page %d [DIGITAL] early stop at retry %d",
                        page_num + 1, retry_num,
                    )
                    break

        return best_extraction, best_score, run_records, gap_map_paths
