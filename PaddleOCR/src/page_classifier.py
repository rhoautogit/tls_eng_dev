"""Page type classifier for the TLS PDF Pipeline.

Classifies each PDF page as digital, scanned, or hybrid based on
embedded text content and image region analysis. This is the first
step in the pipeline -- all routing decisions depend on this result.

Detection logic:
  - Digital: embedded chars >= digital_min_chars AND image coverage < large_image_ratio
             AND text is not garbled (broken font encodings)
  - Scanned: embedded chars < scanned_max_chars, OR text is garbled
  - Hybrid:  chars between thresholds with images, OR chars above digital threshold
             but with large image regions present
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF

from .models import BoundingBox, PageClassification, PageType

logger = logging.getLogger(__name__)

# Default thresholds (overridden by pipeline_config.yaml)
DEFAULT_DIGITAL_MIN_CHARS = 100
DEFAULT_SCANNED_MAX_CHARS = 20
DEFAULT_LARGE_IMAGE_RATIO = 0.3

# Pattern for pdfplumber-style CID placeholders: (cid:XX)
_CID_PATTERN = re.compile(r"\(cid:\d+\)")


def is_garbled_text(text: str, threshold: float = 0.20) -> bool:
    """Detect if extracted text is garbled due to broken font encodings.

    Checks for two common symptoms:
      1. High ratio of control characters (PyMuPDF symptom)
      2. High ratio of CID placeholders (pdfplumber symptom)

    Returns True if more than ``threshold`` (default 20%) of the text
    is unreadable.
    """
    if not text or len(text.strip()) == 0:
        return False

    stripped = text.strip()
    total_chars = len(stripped)

    # Count control characters (ASCII 0-31 excluding common whitespace)
    control_count = sum(
        1 for c in stripped
        if ord(c) < 32 and c not in ('\n', '\r', '\t', ' ')
    )

    # Count CID placeholder characters
    cid_matches = _CID_PATTERN.findall(stripped)
    cid_char_count = sum(len(m) for m in cid_matches)

    garbled_chars = control_count + cid_char_count
    garbled_ratio = garbled_chars / total_chars if total_chars > 0 else 0.0

    return garbled_ratio >= threshold


class PageClassifier:
    """Classifies PDF pages as digital, scanned, or hybrid."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        page_cfg = cfg.get("page_classification", {})

        # Use page_classification config, fall back to merger thresholds,
        # then to hard defaults
        merger_cfg = cfg.get("extraction", {}).get("merger", {})

        self.digital_min_chars: int = page_cfg.get(
            "digital_min_chars",
            merger_cfg.get("digital_threshold", DEFAULT_DIGITAL_MIN_CHARS),
        )
        self.scanned_max_chars: int = page_cfg.get(
            "scanned_max_chars",
            merger_cfg.get("scanned_threshold", DEFAULT_SCANNED_MAX_CHARS),
        )
        self.large_image_ratio: float = page_cfg.get(
            "large_image_ratio", DEFAULT_LARGE_IMAGE_RATIO,
        )

    def classify_page(
        self, pdf_path: str, page_num: int
    ) -> PageClassification:
        """Classify a single page as digital, scanned, or hybrid.

        Opens the PDF, inspects the embedded text layer and image objects,
        then applies threshold logic to determine page type. Pages with
        broken font encodings (garbled text) are reclassified as scanned
        so they get OCR'd instead.
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        pw, ph = page.rect.width, page.rect.height
        page_area = pw * ph

        # Count embedded text characters
        text = page.get_text("text") or ""
        char_count = len(text.strip())

        # Check for garbled/broken font encodings
        garbled = is_garbled_text(text)

        # Analyze image regions
        image_list = page.get_images(full=True)
        image_bboxes = self._get_image_bboxes(page)
        image_count = len(image_list)
        image_coverage = self._compute_image_coverage(
            image_bboxes, page_area
        )

        doc.close()

        # If text is garbled, treat the page as scanned regardless of
        # char count -- the embedded text layer is useless.
        if garbled:
            page_type = PageType.SCANNED
            logger.warning(
                "Page %d has garbled text (broken font encoding), "
                "reclassifying as SCANNED for OCR",
                page_num + 1,
            )
        else:
            page_type = self._determine_type(
                char_count, image_count, image_coverage
            )

        classification = PageClassification(
            page_num=page_num,
            page_type=page_type,
            embedded_char_count=char_count,
            image_region_count=image_count,
            image_coverage_ratio=image_coverage,
            page_width=pw,
            page_height=ph,
            has_garbled_text=garbled,
        )

        logger.info(
            "Page %d classified as %s (chars=%d, images=%d, img_coverage=%.2f%s)",
            page_num + 1,
            page_type.value,
            char_count,
            image_count,
            image_coverage,
            ", GARBLED" if garbled else "",
        )

        return classification

    def classify_document(
        self, pdf_path: str
    ) -> List[PageClassification]:
        """Classify all pages in a PDF document."""
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        classifications = []
        for page_num in range(total_pages):
            classifications.append(self.classify_page(pdf_path, page_num))

        # Log summary
        type_counts = {t: 0 for t in PageType}
        for c in classifications:
            type_counts[c.page_type] += 1
        logger.info(
            "Document classification: %d digital, %d scanned, %d hybrid (total %d)",
            type_counts[PageType.DIGITAL],
            type_counts[PageType.SCANNED],
            type_counts[PageType.HYBRID],
            total_pages,
        )

        return classifications

    def _determine_type(
        self,
        char_count: int,
        image_count: int,
        image_coverage: float,
    ) -> PageType:
        """Apply threshold logic to determine page type.

        Decision matrix:
          chars >= digital_min AND image_coverage < large_ratio -> DIGITAL
          chars < scanned_max                                   -> SCANNED
          chars in between with images                          -> HYBRID
          chars >= digital_min but large image coverage          -> HYBRID
        """
        has_large_images = image_coverage >= self.large_image_ratio

        if char_count < self.scanned_max_chars:
            return PageType.SCANNED

        if char_count >= self.digital_min_chars and not has_large_images:
            return PageType.DIGITAL

        # Everything else is hybrid:
        # - chars between scanned_max and digital_min (uncertain zone)
        # - chars above digital_min but with large image regions
        return PageType.HYBRID

    def _get_image_bboxes(self, page: fitz.Page) -> List[BoundingBox]:
        """Extract bounding boxes of images on the page."""
        bboxes = []
        try:
            for img_info in page.get_image_info():
                bbox = img_info.get("bbox")
                if bbox and len(bbox) == 4:
                    x0, y0, x1, y1 = bbox
                    # Skip tiny images (icons, bullets, etc.)
                    w, h = abs(x1 - x0), abs(y1 - y0)
                    if w > 10 and h > 10:
                        bboxes.append(BoundingBox(x0, y0, x1, y1))
        except Exception as e:
            logger.warning("Failed to get image bboxes on page: %s", e)
        return bboxes

    def _compute_image_coverage(
        self,
        image_bboxes: List[BoundingBox],
        page_area: float,
    ) -> float:
        """Compute fraction of page area covered by images.

        Uses a simple union of bounding boxes (no overlap handling for now).
        """
        if page_area <= 0 or not image_bboxes:
            return 0.0

        total_image_area = sum(bb.area for bb in image_bboxes)
        # Cap at 1.0 in case of overlapping images
        return min(1.0, total_image_area / page_area)
