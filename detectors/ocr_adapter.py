"""OCR Compatibility Adapter for Python 3.12 and Python 3.13.

Normalizes optical character recognition (OCR) across:
1. Legacy `rapidocr_onnxruntime` (Python < 3.13)
2. Modern `rapidocr` paired with `onnxruntime` (Python >= 3.13)
3. Mock OCR engines used in automated test suites

Guarantees the exact internal output structure expected by LabelDetector:
Tuple[Optional[List[Tuple[List[List[float]], str, float]]], float]
where each detection item is:
    (box_points, raw_text, confidence)
with:
    box_points: 4-corner coordinates [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
    raw_text: Recognized string
    confidence: Probability score as a float
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.logger import get_logger

logger = get_logger("detector.ocr_adapter")


def normalize_ocr_output(
    raw_output: Any,
) -> Tuple[Optional[List[Tuple[List[List[float]], str, float]]], float]:
    """Normalize raw OCR engine output from either engine version into standard 3-tuples.

    Returns:
        (normalized_items, elapse_time)
        where normalized_items is None or a list of (box_points, raw_text, confidence).
    """
    if raw_output is None:
        return None, 0.0

    # Case 1: Modern RapidOCROutput object (rapidocr >= 3.0.0)
    # Attributes: .boxes (np.ndarray or list), .txts (tuple/list), .scores (tuple/list), .elapse
    if hasattr(raw_output, "txts") and hasattr(raw_output, "boxes") and hasattr(raw_output, "scores"):
        txts = getattr(raw_output, "txts")
        boxes = getattr(raw_output, "boxes")
        scores = getattr(raw_output, "scores")
        elapse = float(getattr(raw_output, "elapse", 0.0) or 0.0)

        if txts is None or boxes is None or scores is None or len(txts) == 0:
            return None, elapse

        normalized: List[Tuple[List[List[float]], str, float]] = []
        for box, txt, score in zip(boxes, txts, scores):
            if box is None or txt is None or score is None:
                continue
            box_points = [[float(p[0]), float(p[1])] for p in box]
            normalized.append((box_points, str(txt), float(score)))

        return (normalized if normalized else None), elapse

    # Case 2: Tuple/List format (legacy rapidocr_onnxruntime or test mocks)
    # Expected format: (raw_results, elapse) or just raw_results
    raw_results = raw_output
    elapse = 0.0

    if isinstance(raw_output, tuple) and len(raw_output) == 2:
        raw_results, raw_elapse = raw_output
        if isinstance(raw_elapse, (int, float)):
            elapse = float(raw_elapse)

    if raw_results is None:
        return None, elapse

    # If raw_results itself is a RapidOCROutput instance inside a tuple
    if hasattr(raw_results, "txts") and hasattr(raw_results, "boxes") and hasattr(raw_results, "scores"):
        return normalize_ocr_output(raw_results)

    if not isinstance(raw_results, (list, tuple)) or len(raw_results) == 0:
        return None, elapse

    normalized = []
    for item in raw_results:
        if item is None or len(item) < 3:
            continue
        raw_box, raw_text, raw_conf = item[0], item[1], item[2]
        try:
            box_points = [[float(p[0]), float(p[1])] for p in raw_box]
            confidence = float(raw_conf)
            normalized.append((box_points, str(raw_text), confidence))
        except (TypeError, ValueError, IndexError) as err:
            logger.debug("Failed to normalize OCR candidate item %s: %s", item, err)
            continue

    return (normalized if normalized else None), elapse


class OCRAdapter:
    """Wrapper that adapts any OCR engine implementation to the uniform LabelDetector interface."""

    def __init__(self, engine: Any, engine_type: str = "unknown") -> None:
        self._engine = engine
        self._engine_type = engine_type

    @property
    def engine(self) -> Any:
        """Underlying raw OCR engine instance."""
        return self._engine

    @property
    def engine_type(self) -> str:
        """Identified engine family ('rapidocr', 'rapidocr_onnxruntime', 'mock', etc.)."""
        return self._engine_type

    def __call__(
        self, img: np.ndarray
    ) -> Tuple[Optional[List[Tuple[List[List[float]], str, float]]], float]:
        """Invoke OCR inference and return normalized (results, elapse)."""
        if self._engine is None:
            return None, 0.0

        raw_output = self._engine(img)
        return normalize_ocr_output(raw_output)

    def __repr__(self) -> str:
        return f"OCRAdapter(engine_type={self._engine_type!r}, engine={self._engine!r})"


def create_ocr_engine(use_cls: bool = False) -> Optional[OCRAdapter]:
    """Factory function to instantiate and return an OCR engine wrapped in OCRAdapter.

    Prioritizes the appropriate package for the active Python version:
    - On Python >= 3.13: uses maintained `rapidocr` package with onnxruntime backend.
    - On Python < 3.13: uses `rapidocr_onnxruntime` or falls back to `rapidocr`.
    """
    py_version = sys.version_info[:2]

    # For Python >= 3.13, prioritize modern rapidocr
    if py_version >= (3, 13):
        # 1. Try modern rapidocr
        try:
            from rapidocr import RapidOCR
            engine = RapidOCR()
            logger.info("Initialized modern RapidOCR engine for Python %s", sys.version.split()[0])
            return OCRAdapter(engine, engine_type="rapidocr")
        except ImportError:
            pass

        # 2. Fallback to rapidocr_onnxruntime if installed
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR(use_cls=use_cls)
            logger.info("Initialized legacy RapidOCR ONNX engine for Python %s", sys.version.split()[0])
            return OCRAdapter(engine, engine_type="rapidocr_onnxruntime")
        except ImportError:
            pass

    else:
        # For Python < 3.13, prioritize rapidocr_onnxruntime
        try:
            from rapidocr_onnxruntime import RapidOCR
            engine = RapidOCR(use_cls=use_cls)
            logger.info("Initialized legacy RapidOCR ONNX engine for Python %s", sys.version.split()[0])
            return OCRAdapter(engine, engine_type="rapidocr_onnxruntime")
        except ImportError:
            pass

        # Fallback to modern rapidocr if available
        try:
            from rapidocr import RapidOCR
            engine = RapidOCR()
            logger.info("Initialized modern RapidOCR engine for Python %s", sys.version.split()[0])
            return OCRAdapter(engine, engine_type="rapidocr")
        except ImportError:
            pass

    logger.warning("No OCR provider (rapidocr or rapidocr_onnxruntime) could be imported.")
    return None


def get_ocr_engine_info() -> Dict[str, str]:
    """Retrieve metadata about the installed OCR packages without exposing any credentials."""
    import importlib.metadata

    info = {
        "python_version": sys.version.split()[0],
        "package": "none",
        "version": "unknown",
        "backend": "none",
        "backend_version": "unknown",
    }

    # Check onnxruntime
    try:
        import onnxruntime
        info["backend"] = "onnxruntime"
        try:
            info["backend_version"] = importlib.metadata.version("onnxruntime")
        except Exception:
            info["backend_version"] = getattr(onnxruntime, "__version__", "unknown")
    except ImportError:
        pass

    # Check modern rapidocr
    try:
        import rapidocr
        info["package"] = "rapidocr"
        try:
            info["version"] = importlib.metadata.version("rapidocr")
        except Exception:
            info["version"] = getattr(rapidocr, "__version__", "unknown")
        return info
    except ImportError:
        pass

    # Check legacy rapidocr_onnxruntime
    try:
        import rapidocr_onnxruntime
        info["package"] = "rapidocr_onnxruntime"
        try:
            info["version"] = importlib.metadata.version("rapidocr_onnxruntime")
        except Exception:
            info["version"] = getattr(rapidocr_onnxruntime, "__version__", "unknown")
        return info
    except ImportError:
        pass

    return info
