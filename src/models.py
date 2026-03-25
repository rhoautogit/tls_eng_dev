"""Shared data models for the TLS PDF Pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ─── Page Type Classification ────────────────────────────────────────────────

class PageType(Enum):
    """Classification of a PDF page based on its content layer."""
    DIGITAL = "digital"
    SCANNED = "scanned"
    HYBRID = "hybrid"


@dataclass
class PageClassification:
    """Result of classifying a single PDF page."""
    page_num: int
    page_type: PageType
    embedded_char_count: int
    image_region_count: int
    image_coverage_ratio: float       # fraction of page area covered by images
    page_width: float = 0.0
    page_height: float = 0.0
    has_garbled_text: bool = False     # True if embedded text uses broken font encodings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_num": self.page_num,
            "page_type": self.page_type.value,
            "embedded_char_count": self.embedded_char_count,
            "image_region_count": self.image_region_count,
            "image_coverage_ratio": round(self.image_coverage_ratio, 4),
            "has_garbled_text": self.has_garbled_text,
        }


@dataclass
class RegionInfo:
    """A rectangular region on a hybrid page, tagged as digital or scanned."""
    bbox: BoundingBox
    region_type: PageType              # DIGITAL or SCANNED
    char_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "region_type": self.region_type.value,
            "char_count": self.char_count,
        }


class ConfidenceLevel(Enum):
    """Confidence gate levels for scanned page verification."""
    HIGH = "high"           # > 95% -- auto-verified
    MEDIUM = "medium"       # 85-95% -- needs Qwen-VL review
    LOW = "low"             # < 85% -- needs Qwen-VL + human review


@dataclass
class ConfidenceGate:
    """Confidence gate result for a scanned/hybrid page."""
    ocr_confidence: float
    level: ConfidenceLevel
    needs_qwen_vl: bool
    needs_human_review: bool
    word_count: int = 0
    high_confidence_words: int = 0     # words above 95% confidence
    flagged_words: int = 0             # words below threshold

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ocr_confidence": round(self.ocr_confidence, 4),
            "level": self.level.value,
            "needs_qwen_vl": self.needs_qwen_vl,
            "needs_human_review": self.needs_human_review,
            "word_count": self.word_count,
            "high_confidence_words": self.high_confidence_words,
            "flagged_words": self.flagged_words,
        }


@dataclass
class ValidationResult:
    """5-layer validation engine output."""
    coverage_score: float              # V1: spatial coverage
    accuracy_score: float              # V2: text/table accuracy
    completeness_score: float          # V3: missing content inventory
    structural_score: float            # V4: document structure integrity
    cross_validation_score: float      # V5: cross-tool agreement
    composite_score: float = 0.0       # weighted aggregate
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage_score": round(self.coverage_score, 4),
            "accuracy_score": round(self.accuracy_score, 4),
            "completeness_score": round(self.completeness_score, 4),
            "structural_score": round(self.structural_score, 4),
            "cross_validation_score": round(self.cross_validation_score, 4),
            "composite_score": round(self.composite_score, 4),
            "passed": self.passed,
        }


# ─── Verification Status ─────────────────────────────────────────────────────

class VerificationStatus(Enum):
    """Verification status for a page after the full pipeline."""
    AUTO_VERIFIED = "auto_verified"         # high confidence, no review needed
    QWEN_VERIFIED = "qwen_verified"         # Qwen-VL reviewed and confirmed
    QWEN_CORRECTED = "qwen_corrected"       # Qwen-VL made corrections
    PENDING_HUMAN = "pending_human_review"  # flagged for human review
    NOT_VERIFIED = "not_verified"           # not yet processed


# ─── Geometry ────────────────────────────────────────────────────────────────

@dataclass
class BoundingBox:
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.0, self.y1 - self.y0)

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> Dict[str, float]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}

    def iou(self, other: "BoundingBox") -> float:
        """Intersection over union."""
        ix0 = max(self.x0, other.x0)
        iy0 = max(self.y0, other.y0)
        ix1 = min(self.x1, other.x1)
        iy1 = min(self.y1, other.y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlaps(self, other: "BoundingBox", threshold: float = 0.0) -> bool:
        return self.iou(other) > threshold

    def contains_point(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


# ─── Content Elements ─────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    text: str
    bbox: BoundingBox
    page_num: int
    font_size: float = 0.0
    confidence: float = 1.0
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "bbox": self.bbox.to_dict(),
            "page_num": self.page_num,
            "font_size": self.font_size,
            "confidence": self.confidence,
            "source": self.source,
        }


@dataclass
class Table:
    data: List[List[Optional[str]]]   # row-major: data[row][col]
    bbox: BoundingBox
    page_num: int
    confidence: float = 1.0
    source: str = ""
    headers: Optional[List[str]] = None
    continued_from_page: Optional[int] = None  # for multi-page tables
    continued_on_page: Optional[int] = None

    @property
    def num_rows(self) -> int:
        return len(self.data)

    @property
    def num_cols(self) -> int:
        return max((len(r) for r in self.data), default=0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data": self.data,
            "bbox": self.bbox.to_dict(),
            "page_num": self.page_num,
            "confidence": self.confidence,
            "source": self.source,
            "headers": self.headers,
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "continued_from_page": self.continued_from_page,
            "continued_on_page": self.continued_on_page,
        }


@dataclass
class ImageElement:
    bbox: BoundingBox
    page_num: int
    image_path: str = ""
    caption: str = ""
    ocr_text: str = ""
    embedding: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "bbox": self.bbox.to_dict(),
            "page_num": self.page_num,
            "image_path": self.image_path,
            "caption": self.caption,
            "ocr_text": self.ocr_text,
        }
        if self.embedding is not None:
            d["embedding"] = self.embedding
        return d


# ─── Extraction Results ───────────────────────────────────────────────────────

@dataclass
class PageExtractionResult:
    page_num: int
    text_blocks: List[TextBlock]
    tables: List[Table]
    images: List[ImageElement]
    source: str
    is_scanned: bool = False
    page_width: float = 0.0
    page_height: float = 0.0
    page_type: PageType = PageType.DIGITAL
    confidence_gate: Optional[ConfidenceGate] = None
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED

    def all_text(self) -> str:
        """Concatenate all extracted text for coverage calculation."""
        parts: List[str] = []
        for b in self.text_blocks:
            parts.append(b.text)
        for t in self.tables:
            for row in t.data:
                for cell in row:
                    if cell:
                        parts.append(cell)
        for img in self.images:
            if img.caption:
                parts.append(img.caption)
        return " ".join(parts)


# ─── Extraction Parameters ────────────────────────────────────────────────────

@dataclass
class ExtractionParameters:
    """All tunable parameters for a single extraction run."""
    # pdfplumber
    pdfplumber_mode: str = "lattice"           # "lattice" | "stream"
    pdfplumber_snap_tolerance: float = 3.0
    pdfplumber_join_tolerance: float = 3.0
    pdfplumber_edge_min_length: float = 3.0
    pdfplumber_use_text_alignment: bool = False
    pdfplumber_word_x_tolerance: int = 3
    pdfplumber_word_y_tolerance: int = 3

    # OpenCV
    opencv_dpi: int = 200
    opencv_threshold_block_size: int = 11
    opencv_threshold_constant: int = 2
    opencv_kernel_h: Tuple[int, int] = (30, 1)
    opencv_kernel_v: Tuple[int, int] = (1, 30)
    opencv_iterations: int = 1
    opencv_use_clahe: bool = False
    opencv_clahe_clip: float = 2.0
    opencv_clahe_grid: Tuple[int, int] = (8, 8)
    opencv_use_sharpening: bool = False
    opencv_use_noise_reduction: bool = False

    # Targeting
    target_bbox: Optional[BoundingBox] = None  # crop to this region if set
    combine_all_strategies: bool = False
    max_sensitivity: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdfplumber_mode": self.pdfplumber_mode,
            "pdfplumber_snap_tolerance": self.pdfplumber_snap_tolerance,
            "pdfplumber_join_tolerance": self.pdfplumber_join_tolerance,
            "pdfplumber_edge_min_length": self.pdfplumber_edge_min_length,
            "pdfplumber_use_text_alignment": self.pdfplumber_use_text_alignment,
            "opencv_dpi": self.opencv_dpi,
            "opencv_threshold_block_size": self.opencv_threshold_block_size,
            "opencv_threshold_constant": self.opencv_threshold_constant,
            "opencv_kernel_h": list(self.opencv_kernel_h),
            "opencv_kernel_v": list(self.opencv_kernel_v),
            "opencv_iterations": self.opencv_iterations,
            "opencv_use_clahe": self.opencv_use_clahe,
            "opencv_clahe_clip": self.opencv_clahe_clip,
            "opencv_clahe_grid": list(self.opencv_clahe_grid),
            "opencv_use_sharpening": self.opencv_use_sharpening,
            "opencv_use_noise_reduction": self.opencv_use_noise_reduction,
            "target_bbox": self.target_bbox.to_dict() if self.target_bbox else None,
            "combine_all_strategies": self.combine_all_strategies,
            "max_sensitivity": self.max_sensitivity,
        }


# ─── Gap Analysis ─────────────────────────────────────────────────────────────

@dataclass
class Gap:
    bbox: BoundingBox
    area_ratio: float                   # fraction of page area
    estimated_type: str                 # "table" | "text" | "image" | "unknown"
    severity: str                       # "low" | "medium" | "high"
    page_num: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox.to_dict(),
            "area_ratio": round(self.area_ratio, 4),
            "estimated_type": self.estimated_type,
            "severity": self.severity,
            "page_num": self.page_num,
        }


# ─── Run Tracking ─────────────────────────────────────────────────────────────

@dataclass
class RunRecord:
    run_number: int          # 0 = initial, 1/2/3 = retries
    parameters: Dict[str, Any]
    score_before: float
    score_after: float
    delta: float
    gap_map_path: str = ""
    timestamp: str = ""
    gaps: List[Gap] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_number": self.run_number,
            "parameters": self.parameters,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "delta": round(self.delta, 4),
            "gap_map_path": self.gap_map_path,
            "timestamp": self.timestamp,
            "gaps": [g.to_dict() for g in self.gaps],
        }


# ─── Final Results ────────────────────────────────────────────────────────────

@dataclass
class PageResult:
    page_num: int
    final_score: float
    initial_score: float
    passed: bool
    extraction: PageExtractionResult
    run_records: List[RunRecord]
    gap_map_paths: List[str]
    status: str              # "passed_initial" | "resolved" | "unresolved"
    source_contributions: Dict[str, float] = field(default_factory=dict)
    classification: Optional[PageClassification] = None
    validation: Optional[ValidationResult] = None

    def confidence_label(self, high: float = 0.8, medium: float = 0.5) -> str:
        if self.final_score >= high:
            return "high"
        if self.final_score >= medium:
            return "medium"
        return "low"

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "page_num": self.page_num,
            "page_display": self.page_num + 1,
            "final_score": round(self.final_score, 4),
            "initial_score": round(self.initial_score, 4),
            "passed": self.passed,
            "status": self.status,
            "page_type": self.extraction.page_type.value,
            "verification_status": self.extraction.verification_status.value,
            "confidence": self.confidence_label(),
            "num_text_blocks": len(self.extraction.text_blocks),
            "num_tables": len(self.extraction.tables),
            "num_images": len(self.extraction.images),
            "is_scanned": self.extraction.is_scanned,
            "source_contributions": self.source_contributions,
            "run_records": [r.to_dict() for r in self.run_records],
            "gap_map_paths": self.gap_map_paths,
        }
        if self.classification:
            result["classification"] = self.classification.to_dict()
        if self.validation:
            result["validation"] = self.validation.to_dict()
        if self.extraction.confidence_gate:
            result["confidence_gate"] = self.extraction.confidence_gate.to_dict()
        return result


@dataclass
class DocumentResult:
    pdf_path: str
    total_pages: int
    pages: List[PageResult]
    timestamp: str
    overall_score: float

    def to_dict(self) -> Dict[str, Any]:
        passed = [p for p in self.pages if p.passed]
        unresolved = [p for p in self.pages if p.status == "unresolved"]
        high_conf = [p for p in self.pages if p.confidence_label() == "high"]
        med_conf = [p for p in self.pages if p.confidence_label() == "medium"]
        low_conf = [p for p in self.pages if p.confidence_label() == "low"]

        # Average per-tool contributions across all pages
        n = max(1, len(self.pages))
        avg_contributions: Dict[str, float] = {}
        if self.pages and self.pages[0].source_contributions:
            keys = self.pages[0].source_contributions.keys()
            for k in keys:
                avg_contributions[k] = round(
                    sum(p.source_contributions.get(k, 0.0) for p in self.pages) / n, 4
                )

        # Page type breakdown
        digital_pages = [p for p in self.pages
                         if p.extraction.page_type == PageType.DIGITAL]
        scanned_pages = [p for p in self.pages
                         if p.extraction.page_type == PageType.SCANNED]
        hybrid_pages = [p for p in self.pages
                        if p.extraction.page_type == PageType.HYBRID]

        # Verification status breakdown
        auto_verified = [p for p in self.pages
                         if p.extraction.verification_status == VerificationStatus.AUTO_VERIFIED]
        qwen_verified = [p for p in self.pages
                         if p.extraction.verification_status == VerificationStatus.QWEN_VERIFIED]
        qwen_corrected = [p for p in self.pages
                          if p.extraction.verification_status == VerificationStatus.QWEN_CORRECTED]
        pending_human = [p for p in self.pages
                         if p.extraction.verification_status == VerificationStatus.PENDING_HUMAN]

        return {
            "pdf_path": self.pdf_path,
            "total_pages": self.total_pages,
            "overall_score": round(self.overall_score, 4),
            "timestamp": self.timestamp,
            "summary": {
                "passed_pages": len(passed),
                "failed_pages": self.total_pages - len(passed),
                "unresolved_pages": len(unresolved),
                "pass_rate": round(len(passed) / max(1, self.total_pages), 4),
                "page_types": {
                    "digital": len(digital_pages),
                    "scanned": len(scanned_pages),
                    "hybrid": len(hybrid_pages),
                },
                "verification_status": {
                    "auto_verified": len(auto_verified),
                    "qwen_verified": len(qwen_verified),
                    "qwen_corrected": len(qwen_corrected),
                    "pending_human_review": len(pending_human),
                },
                "confidence_breakdown": {
                    "high": len(high_conf),
                    "medium": len(med_conf),
                    "low": len(low_conf),
                },
                "tool_contributions": avg_contributions,
            },
            "pages": [p.to_dict() for p in self.pages],
        }
