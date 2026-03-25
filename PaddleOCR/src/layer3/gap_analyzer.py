"""Layer 3 – Gap Analyzer.

Compares the original rendered page against the visual twin to identify
regions of content that were not extracted.  Each gap is classified by
estimated type (table / text / image) and severity (low / medium / high).

A gap-map PNG is saved for inclusion in the digestion report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ..models import BoundingBox, Gap, PageExtractionResult
from .visual_twin import (
    create_covered_mask,
    render_extraction_twin,
    render_page_image,
)

logger = logging.getLogger(__name__)

# BGR colours used in gap-map visualisation
_COVERED_OVERLAY = (80, 200, 80)   # green – successfully extracted
_GAP_OVERLAY = (40, 40, 220)       # red   – missed content
_BOX_HIGH = (0, 0, 255)
_BOX_MED = (0, 165, 255)
_BOX_LOW = (0, 255, 255)


# ─── Gap Type Estimation ──────────────────────────────────────────────────────

def _estimate_gap_type(region: np.ndarray) -> str:
    """Classify a gap region's content type from its appearance."""
    if region.size == 0:
        return "unknown"
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY) if region.ndim == 3 else region
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    area = gray.shape[0] * gray.shape[1]
    if area == 0:
        return "unknown"

    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, gray.shape[1] // 10), 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, gray.shape[0] // 10)))
    h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kern)
    v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern)

    h_density = np.count_nonzero(h_lines) / area
    v_density = np.count_nonzero(v_lines) / area

    if h_density > 0.03 and v_density > 0.03:
        return "table"

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small = sum(1 for c in contours if cv2.contourArea(c) < 200)
    if small > 8:
        return "text"

    return "image"


# ─── Gap Map Visualisation ────────────────────────────────────────────────────

def _draw_gap_map(
    original: np.ndarray,
    covered_mask: np.ndarray,
    gap_mask: np.ndarray,
    gaps: List[Gap],
    scale_x: float,
    scale_y: float,
) -> np.ndarray:
    vis = original.copy()

    # Semi-transparent overlay for covered regions
    green = np.zeros_like(vis)
    green[covered_mask > 0] = _COVERED_OVERLAY
    vis = cv2.addWeighted(vis, 0.75, green, 0.25, 0)

    # Semi-transparent overlay for gaps
    red = np.zeros_like(vis)
    red[gap_mask > 0] = _GAP_OVERLAY
    vis = cv2.addWeighted(vis, 0.75, red, 0.25, 0)

    # Bounding boxes + labels for each gap
    for gap in gaps:
        x0 = max(0, int(gap.bbox.x0 * scale_x))
        y0 = max(0, int(gap.bbox.y0 * scale_y))
        x1 = int(gap.bbox.x1 * scale_x)
        y1 = int(gap.bbox.y1 * scale_y)
        colour = {"high": _BOX_HIGH, "medium": _BOX_MED}.get(gap.severity, _BOX_LOW)
        cv2.rectangle(vis, (x0, y0), (x1, y1), colour, 2)
        label = f"{gap.estimated_type} [{gap.severity}] {gap.area_ratio:.1%}"
        cv2.putText(
            vis, label, (x0 + 2, max(y0 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA,
        )
    return vis


# ─── Main Analyzer ────────────────────────────────────────────────────────────

class GapAnalyzer:
    """Finds uncovered content regions and produces annotated gap-map images."""

    def __init__(self, config: Dict[str, Any]) -> None:
        vis_cfg = config.get("visual", {})
        self._dpi: int = int(vis_cfg.get("render_dpi", 150))
        self._content_thresh: int = int(vis_cfg.get("content_threshold", 240))
        self._dilate_k: int = int(vis_cfg.get("content_dilate_kernel", 5))
        self._dilate_iter: int = int(vis_cfg.get("content_dilate_iterations", 2))
        self._min_ratio: float = float(vis_cfg.get("min_gap_area_ratio", 0.005))
        sev = vis_cfg.get("gap_severity", {})
        self._sev_low: float = float(sev.get("low_max", 0.02))
        self._sev_med: float = float(sev.get("medium_max", 0.05))

    def analyze(
        self,
        pdf_path: str,
        page_num: int,
        extraction: PageExtractionResult,
        gap_map_dir: Path,
        retry_num: int = 0,
    ) -> Tuple[List[Gap], str]:
        """Run gap analysis and save a gap-map image.

        Returns:
            (gaps, gap_map_path)
        """
        original, pdf_w, pdf_h = render_page_image(pdf_path, page_num, self._dpi)
        img_h, img_w = original.shape[:2]

        scale_x = img_w / pdf_w if pdf_w else 1.0
        scale_y = img_h / pdf_h if pdf_h else 1.0

        # Build masks
        twin = render_extraction_twin(extraction, img_w, img_h, scale_x, scale_y)
        covered_mask = create_covered_mask(twin)

        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        _, content_mask = cv2.threshold(gray, self._content_thresh, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((self._dilate_k, self._dilate_k), np.uint8)
        content_mask = cv2.dilate(content_mask, kernel, iterations=self._dilate_iter)

        gap_mask = cv2.bitwise_and(content_mask, cv2.bitwise_not(covered_mask))

        # Find connected gap components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            gap_mask, connectivity=8
        )
        page_area = img_h * img_w
        gaps: List[Gap] = []

        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            ratio = area / page_area
            if ratio < self._min_ratio:
                continue

            gx = int(stats[label, cv2.CC_STAT_LEFT])
            gy = int(stats[label, cv2.CC_STAT_TOP])
            gw = int(stats[label, cv2.CC_STAT_WIDTH])
            gh = int(stats[label, cv2.CC_STAT_HEIGHT])

            region = original[gy : gy + gh, gx : gx + gw]
            gap_type = _estimate_gap_type(region)

            severity = (
                "low" if ratio < self._sev_low
                else "medium" if ratio < self._sev_med
                else "high"
            )

            gaps.append(
                Gap(
                    bbox=BoundingBox(
                        gx / scale_x, gy / scale_y,
                        (gx + gw) / scale_x, (gy + gh) / scale_y,
                    ),
                    area_ratio=ratio,
                    estimated_type=gap_type,
                    severity=severity,
                    page_num=page_num,
                )
            )

        # Sort by descending severity / area
        gaps.sort(key=lambda g: -g.area_ratio)

        # Save gap map
        gap_map_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_retry{retry_num}" if retry_num > 0 else "_initial"
        gap_map_path = str(
            gap_map_dir / f"page_{page_num + 1:03d}{suffix}_gapmap.png"
        )
        gap_map_img = _draw_gap_map(
            original, covered_mask, gap_mask, gaps, scale_x, scale_y
        )
        cv2.imwrite(gap_map_path, gap_map_img)

        logger.debug(
            "Page %d retry %d: %d gaps found, map → %s",
            page_num, retry_num, len(gaps), gap_map_path,
        )
        return gaps, gap_map_path
