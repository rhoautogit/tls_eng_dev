"""Path executors for the branched pipeline architecture.

Each page is classified as digital, scanned, or hybrid, then routed
to the appropriate path executor.
"""
from .digital_path import DigitalPathExecutor
from .scanned_path import ScannedPathExecutor
from .hybrid_path import HybridPathExecutor

__all__ = ["DigitalPathExecutor", "ScannedPathExecutor", "HybridPathExecutor"]
