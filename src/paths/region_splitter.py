"""Region splitter for hybrid PDF pages.

Analyzes a page that contains both digital text regions and scanned/image
regions, and splits it into typed regions that can be routed to the
appropriate extraction tools.

Uses PyMuPDF to identify where embedded text exists and where large images
are placed, then classifies rectangular regions of the page.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import fitz

from ..models import BoundingBox, PageType, RegionInfo

logger = logging.getLogger(__name__)


class RegionSplitter:
    """Splits a hybrid page into digital and scanned regions."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

    def split_page(
        self, pdf_path: str, page_num: int
    ) -> List[RegionInfo]:
        """Identify digital vs scanned regions on a hybrid page.

        Returns a list of RegionInfo objects, each tagged as DIGITAL or SCANNED.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pw, ph = page.rect.width, page.rect.height

        # Get text blocks with positions
        text_blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, type)
        text_regions = []
        for block in text_blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            if block_type == 0 and text.strip():
                text_regions.append(BoundingBox(x0, y0, x1, y1))

        # Get image regions
        image_regions = []
        for img_info in page.get_image_info():
            bbox = img_info.get("bbox")
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = bbox
                w, h = abs(x1 - x0), abs(y1 - y0)
                # Only consider significant images (not tiny icons)
                if w > 50 and h > 50:
                    image_regions.append(BoundingBox(x0, y0, x1, y1))

        doc.close()

        regions: List[RegionInfo] = []

        # Classify image regions as SCANNED if they don't overlap
        # significantly with text
        for img_bb in image_regions:
            text_overlap = self._text_overlap_ratio(img_bb, text_regions)
            if text_overlap < 0.3:
                # This image region has little text -- treat as scanned
                regions.append(RegionInfo(
                    bbox=img_bb,
                    region_type=PageType.SCANNED,
                    char_count=0,
                ))

        # The rest of the page (where text blocks are) is digital
        # Create a digital region covering the text areas
        if text_regions:
            # Compute bounding box of all text regions
            all_x0 = min(r.x0 for r in text_regions)
            all_y0 = min(r.y0 for r in text_regions)
            all_x1 = max(r.x1 for r in text_regions)
            all_y1 = max(r.y1 for r in text_regions)

            total_chars = sum(
                len(block[4].strip()) for block in text_blocks
                if block[6] == 0 and block[4].strip()
            )

            digital_bb = BoundingBox(all_x0, all_y0, all_x1, all_y1)
            regions.append(RegionInfo(
                bbox=digital_bb,
                region_type=PageType.DIGITAL,
                char_count=total_chars,
            ))

        logger.debug(
            "Page %d hybrid split: %d digital regions, %d scanned regions",
            page_num + 1,
            sum(1 for r in regions if r.region_type == PageType.DIGITAL),
            sum(1 for r in regions if r.region_type == PageType.SCANNED),
        )

        return regions

    def _text_overlap_ratio(
        self,
        image_bb: BoundingBox,
        text_regions: List[BoundingBox],
    ) -> float:
        """Compute what fraction of the image bbox overlaps with text regions."""
        if image_bb.area <= 0:
            return 0.0

        total_overlap = 0.0
        for text_bb in text_regions:
            # Compute intersection area
            ix0 = max(image_bb.x0, text_bb.x0)
            iy0 = max(image_bb.y0, text_bb.y0)
            ix1 = min(image_bb.x1, text_bb.x1)
            iy1 = min(image_bb.y1, text_bb.y1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            total_overlap += inter

        return min(1.0, total_overlap / image_bb.area)
