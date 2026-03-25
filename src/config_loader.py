"""YAML configuration loader for the TLS PDF Pipeline."""
from pathlib import Path
from typing import Any, Dict

import yaml

from .models import ExtractionParameters


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration from a YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def default_params_from_config(cfg: Dict[str, Any]) -> ExtractionParameters:
    """Build a default ExtractionParameters from the loaded config."""
    ext = cfg.get("extraction", {})
    pp = ext.get("pdfplumber", {})
    mode = pp.get("default_mode", "lattice")
    mode_cfg = pp.get(mode, {})
    oc = ext.get("opencv", {})
    thr = oc.get("adaptive_threshold", {})
    morph = oc.get("morphological", {})

    return ExtractionParameters(
        pdfplumber_mode=mode,
        pdfplumber_snap_tolerance=float(mode_cfg.get("snap_tolerance", 3.0)),
        pdfplumber_join_tolerance=float(mode_cfg.get("join_tolerance", 3.0)),
        pdfplumber_edge_min_length=float(mode_cfg.get("edge_min_length", 3.0)),
        pdfplumber_use_text_alignment=False,
        pdfplumber_word_x_tolerance=int(pp.get("word_x_tolerance", 3)),
        pdfplumber_word_y_tolerance=int(pp.get("word_y_tolerance", 3)),
        opencv_dpi=int(oc.get("dpi", 200)),
        opencv_threshold_block_size=int(thr.get("block_size", 11)),
        opencv_threshold_constant=int(thr.get("constant", 2)),
        opencv_kernel_h=tuple(morph.get("kernel_h", [30, 1])),
        opencv_kernel_v=tuple(morph.get("kernel_v", [1, 30])),
        opencv_iterations=int(morph.get("iterations", 1)),
    )
