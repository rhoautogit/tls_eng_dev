"""Layer 3 – Parameter Adjuster.

Selects the next retry's ExtractionParameters based on the retry number,
the gaps found, and the current parameters.  Each retry escalates to a
more aggressive preprocessing strategy as specified:

  Retry 1 – Switch pdfplumber mode, loosen tolerances, target gap region.
  Retry 2 – Enhanced OpenCV preprocessing (CLAHE, sharpening, noise
             reduction), text-alignment pdfplumber mode.
  Retry 3 – Full combined, maximum sensitivity.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..models import BoundingBox, ExtractionParameters, Gap

logger = logging.getLogger(__name__)


class ParameterAdjuster:
    """Builds retry ExtractionParameters from gap analysis."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def get_params(
        self,
        retry_num: int,
        gaps: List[Gap],
        current_params: ExtractionParameters,
    ) -> ExtractionParameters:
        """Return new ExtractionParameters for the given retry number."""
        if retry_num == 1:
            return self._retry_1(gaps, current_params)
        if retry_num == 2:
            return self._retry_2(gaps, current_params)
        return self._retry_3(gaps, current_params)

    # ── Private builders ──────────────────────────────────────────────────────

    def _target_from_gaps(self, gaps: List[Gap]) -> Optional[BoundingBox]:
        """Return the bounding box of the most severe gap, or None."""
        if not gaps:
            return None
        return max(gaps, key=lambda g: g.area_ratio).bbox

    def _retry_1(
        self, gaps: List[Gap], current: ExtractionParameters
    ) -> ExtractionParameters:
        """Switch pdfplumber mode; loosen tolerances; target worst gap."""
        s = self.config.get("retry_strategies", {}).get("retry_1", {})
        multiplier = float(s.get("tolerance_multiplier", 1.5))

        new_mode = "stream" if current.pdfplumber_mode == "lattice" else "lattice"
        params = ExtractionParameters(
            pdfplumber_mode=new_mode,
            pdfplumber_snap_tolerance=current.pdfplumber_snap_tolerance * multiplier,
            pdfplumber_join_tolerance=current.pdfplumber_join_tolerance * multiplier,
            pdfplumber_edge_min_length=max(
                1.0, current.pdfplumber_edge_min_length / multiplier
            ),
            pdfplumber_use_text_alignment=False,
            pdfplumber_word_x_tolerance=current.pdfplumber_word_x_tolerance,
            pdfplumber_word_y_tolerance=current.pdfplumber_word_y_tolerance,
            opencv_dpi=current.opencv_dpi,
            opencv_threshold_block_size=current.opencv_threshold_block_size,
            opencv_threshold_constant=current.opencv_threshold_constant,
            opencv_kernel_h=current.opencv_kernel_h,
            opencv_kernel_v=current.opencv_kernel_v,
            opencv_iterations=current.opencv_iterations,
            target_bbox=self._target_from_gaps(gaps) if s.get("target_gap_region") else None,
        )
        logger.debug("Retry 1 params: mode=%s tol=%.1f", new_mode, multiplier)
        return params

    def _retry_2(
        self, gaps: List[Gap], current: ExtractionParameters
    ) -> ExtractionParameters:
        """Enhanced OpenCV preprocessing; text-alignment pdfplumber."""
        s = self.config.get("retry_strategies", {}).get("retry_2", {})
        clahe_cfg = s.get("clahe", {})
        grid = tuple(clahe_cfg.get("tile_grid_size", [8, 8]))

        params = ExtractionParameters(
            pdfplumber_mode=current.pdfplumber_mode,
            pdfplumber_snap_tolerance=current.pdfplumber_snap_tolerance,
            pdfplumber_join_tolerance=current.pdfplumber_join_tolerance,
            pdfplumber_edge_min_length=current.pdfplumber_edge_min_length,
            pdfplumber_use_text_alignment=bool(s.get("text_alignment_detection", True)),
            pdfplumber_word_x_tolerance=current.pdfplumber_word_x_tolerance,
            pdfplumber_word_y_tolerance=current.pdfplumber_word_y_tolerance,
            opencv_dpi=current.opencv_dpi,
            opencv_use_clahe=True,
            opencv_clahe_clip=float(clahe_cfg.get("clip_limit", 2.0)),
            opencv_clahe_grid=grid,
            opencv_use_sharpening=bool(s.get("sharpening", True)),
            opencv_use_noise_reduction=bool(s.get("noise_reduction", True)),
            target_bbox=self._target_from_gaps(gaps) if s.get("target_gap_region") else None,
        )
        logger.debug("Retry 2 params: CLAHE+sharpening+noise_reduction")
        return params

    def _retry_3(
        self, gaps: List[Gap], current: ExtractionParameters
    ) -> ExtractionParameters:
        """Full combined reprocessing with maximum sensitivity."""
        s = self.config.get("retry_strategies", {}).get("retry_3", {})
        clahe_cfg = s.get("clahe", {})
        grid = tuple(clahe_cfg.get("tile_grid_size", [4, 4]))

        params = ExtractionParameters(
            pdfplumber_mode="lattice",  # try lattice one more time
            pdfplumber_snap_tolerance=float(s.get("pdfplumber_snap_tolerance", 10.0)),
            pdfplumber_join_tolerance=float(s.get("pdfplumber_join_tolerance", 10.0)),
            pdfplumber_edge_min_length=float(s.get("pdfplumber_edge_min_length", 1.0)),
            pdfplumber_use_text_alignment=bool(s.get("text_alignment_detection", True)),
            pdfplumber_word_x_tolerance=1,
            pdfplumber_word_y_tolerance=1,
            opencv_dpi=int(s.get("opencv_dpi", 300)),
            opencv_use_clahe=True,
            opencv_clahe_clip=float(clahe_cfg.get("clip_limit", 3.0)),
            opencv_clahe_grid=grid,
            opencv_use_sharpening=True,
            opencv_use_noise_reduction=True,
            opencv_iterations=2,
            combine_all_strategies=True,
            max_sensitivity=True,
            target_bbox=None,   # full page on retry 3
        )
        logger.debug("Retry 3 params: maximum sensitivity full-page reprocessing")
        return params
