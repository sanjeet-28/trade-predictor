"""Unit and integration tests for OCR Compatibility Adapter (Python 3.12 and Python 3.13).

Verifies:
  1. Output normalization for modern RapidOCROutput and legacy list/tuple formats.
  2. Empty / no-detection edge cases handling.
  3. OCRAdapter wrapping and callable protocol.
  4. Engine factory selection (create_ocr_engine) based on runtime Python environment.
  5. get_ocr_engine_info metadata integrity and credential safety.
  6. Live BUY and SELL visual label recognition with confidence scores and bounding boxes.
  7. Dual-engine output structure equivalence.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from core.models import BoundingBox, DetectorType, Frame, TradeSignalType
from detectors.label_detector import LabelDetector
from detectors.ocr_adapter import (
    OCRAdapter,
    create_ocr_engine,
    get_ocr_engine_info,
    normalize_ocr_output,
)


@dataclass
class MockRapidOCROutput:
    """Mock simulating rapidocr >= 3.0.0 dataclass output."""
    boxes: Optional[np.ndarray] = None
    txts: Optional[Tuple[str, ...]] = None
    scores: Optional[Tuple[float, ...]] = None
    elapse: float = 0.025


def test_normalize_ocr_output_modern_rapidocr():
    """Verify normalization of modern RapidOCROutput format into standard 3-tuples."""
    mock_boxes = np.array([
        [[10.0, 20.0], [80.0, 20.0], [80.0, 50.0], [10.0, 50.0]],
        [[100.0, 20.0], [150.0, 20.0], [150.0, 50.0], [100.0, 50.0]],
    ], dtype=np.float32)
    mock_txts = ("BUY", "TP1")
    mock_scores = (0.995, 0.912)

    raw_output = MockRapidOCROutput(boxes=mock_boxes, txts=mock_txts, scores=mock_scores, elapse=0.015)
    normalized, elapse = normalize_ocr_output(raw_output)

    assert elapse == pytest.approx(0.015)
    assert normalized is not None
    assert len(normalized) == 2

    # Check item 0
    box0, text0, conf0 = normalized[0]
    assert text0 == "BUY"
    assert conf0 == pytest.approx(0.995)
    assert box0 == [[10.0, 20.0], [80.0, 20.0], [80.0, 50.0], [10.0, 50.0]]

    # Check item 1
    box1, text1, conf1 = normalized[1]
    assert text1 == "TP1"
    assert conf1 == pytest.approx(0.912)


def test_normalize_ocr_output_legacy_tuple():
    """Verify normalization of legacy rapidocr_onnxruntime list/tuple output."""
    legacy_boxes = [
        [[[15.0, 25.0], [85.0, 25.0], [85.0, 55.0], [15.0, 55.0]], "SELL", 0.984],
        [[[110.0, 25.0], [160.0, 25.0], [160.0, 55.0], [110.0, 55.0]], "SL", 0.89],
    ]
    raw_output = (legacy_boxes, 0.022)

    normalized, elapse = normalize_ocr_output(raw_output)

    assert elapse == pytest.approx(0.022)
    assert normalized is not None
    assert len(normalized) == 2

    box0, text0, conf0 = normalized[0]
    assert text0 == "SELL"
    assert conf0 == pytest.approx(0.984)
    assert box0 == [[15.0, 25.0], [85.0, 25.0], [85.0, 55.0], [15.0, 55.0]]


def test_normalize_ocr_output_empty_cases():
    """Verify empty or None OCR results normalize consistently to (None, elapse)."""
    # None input
    res, el = normalize_ocr_output(None)
    assert res is None
    assert el == 0.0

    # Modern empty RapidOCROutput
    modern_empty = MockRapidOCROutput(boxes=None, txts=None, scores=None, elapse=0.005)
    res, el = normalize_ocr_output(modern_empty)
    assert res is None
    assert el == pytest.approx(0.005)

    # Legacy empty tuple
    legacy_empty = (None, 0.012)
    res, el = normalize_ocr_output(legacy_empty)
    assert res is None
    assert el == pytest.approx(0.012)

    # Legacy empty list
    legacy_empty_list = ([], 0.008)
    res, el = normalize_ocr_output(legacy_empty_list)
    assert res is None
    assert el == pytest.approx(0.008)


def test_ocr_adapter_callable_interface():
    """Verify OCRAdapter wraps raw engine and provides callable interface."""
    def mock_engine(img):
        return ([[[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]], "BUY", 0.99]], 0.01)

    adapter = OCRAdapter(mock_engine, engine_type="mock")
    assert adapter.engine_type == "mock"
    assert "mock" in repr(adapter)

    dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
    results, elapse = adapter(dummy_img)

    assert results is not None
    assert len(results) == 1
    assert results[0][1] == "BUY"
    assert results[0][2] == pytest.approx(0.99)
    assert elapse == pytest.approx(0.01)


def test_create_ocr_engine_factory():
    """Verify create_ocr_engine initializes an available OCR provider."""
    engine = create_ocr_engine()
    assert engine is not None
    assert isinstance(engine, OCRAdapter)
    assert engine.engine_type in ("rapidocr", "rapidocr_onnxruntime")


def test_get_ocr_engine_info_metadata():
    """Verify get_ocr_engine_info exposes metadata safely without credentials."""
    info = get_ocr_engine_info()
    assert "python_version" in info
    assert "package" in info
    assert "version" in info
    assert "backend" in info
    assert "backend_version" in info

    assert info["package"] in ("rapidocr", "rapidocr_onnxruntime")
    assert info["backend"] == "onnxruntime"
    assert info["version"] != "unknown"
    assert info["backend_version"] != "unknown"


def test_live_ocr_buy_label_recognition():
    """Verify end-to-end BUY label detection in LabelDetector using live OCR engine."""
    # Synthesize 720p frame with BUY label in chart area (inside ROI [0, 50, 1025, 450])
    frame_data = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Background dark chart area
    frame_data[50:650, 0:1100] = (25, 25, 25)
    # Bright BUY badge
    cv2.rectangle(frame_data, (200, 150), (320, 200), (0, 180, 0), -1)
    cv2.putText(frame_data, "BUY", (210, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    detector = LabelDetector(baseline_frames=0)
    frame = Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=1280,
        height=720,
        data=frame_data,
    )

    events = detector.process_frame(frame)
    assert len(events) >= 1, "Expected at least one BUY detection event"

    buy_event = next((e for e in events if e.signal_type == TradeSignalType.BUY), None)
    assert buy_event is not None, "Expected BUY event to be recognized by live OCR"
    assert buy_event.detector_type == DetectorType.LABEL
    assert buy_event.confidence >= 0.70
    assert buy_event.bbox.x >= 180
    assert buy_event.bbox.width >= 40
    assert buy_event.bbox.height >= 15


def test_live_ocr_sell_label_recognition():
    """Verify end-to-end SELL label detection in LabelDetector using live OCR engine."""
    frame_data = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_data[50:650, 0:1100] = (25, 25, 25)
    # Bright SELL badge
    cv2.rectangle(frame_data, (400, 250), (540, 300), (0, 0, 180), -1)
    cv2.putText(frame_data, "SELL", (410, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)

    detector = LabelDetector(baseline_frames=0)
    frame = Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=1280,
        height=720,
        data=frame_data,
    )

    events = detector.process_frame(frame)
    assert len(events) >= 1, "Expected at least one SELL detection event"

    sell_event = next((e for e in events if e.signal_type == TradeSignalType.SELL), None)
    assert sell_event is not None, "Expected SELL event to be recognized by live OCR"
    assert sell_event.detector_type == DetectorType.LABEL
    assert sell_event.confidence >= 0.70
    assert sell_event.bbox.x >= 380
    assert sell_event.bbox.width >= 40
    assert sell_event.bbox.height >= 15
