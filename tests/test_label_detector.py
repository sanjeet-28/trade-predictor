"""Unit tests for Detector A: BUY/SELL Label Detection."""

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock
import numpy as np
import pytest

from core.models import BoundingBox, DetectionEvent, DetectorType, Frame, TradeSignalType
from detectors.label_detector import LabelDetector


def make_dummy_frame(width: int = 1280, height: int = 720) -> Frame:
    """Create a dummy frame buffer."""
    data = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=width,
        height=height,
        data=data,
    )


class MockOCR:
    """Mock OCR engine returning predetermined text boxes."""

    def __init__(self, detections: List[Tuple[List[List[float]], str, float]]) -> None:
        self.detections = detections

    def __call__(self, img: np.ndarray):
        return self.detections, [0.01, 0.01, 0.01]


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_clear_buy_label() -> None:
    """Test 1: Clear BUY label produces a BUY detection event."""
    # Box in ROI: [100, 100] to [160, 130]
    box = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    mock_ocr = MockOCR([(box, "BUY", 0.98)])

    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    events = detector.process_frame(make_dummy_frame())

    assert len(events) == 1
    assert events[0].detector_type == DetectorType.LABEL
    assert events[0].signal_type == TradeSignalType.BUY
    assert events[0].confidence == pytest.approx(0.98)
    assert events[0].bbox.width > 0


def test_clear_sell_label() -> None:
    """Test 2: Clear SELL label produces a SELL detection event."""
    box = [[200.0, 250.0], [270.0, 250.0], [270.0, 280.0], [200.0, 280.0]]
    mock_ocr = MockOCR([(box, "SELL", 0.95)])

    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    events = detector.process_frame(make_dummy_frame())

    assert len(events) == 1
    assert events[0].detector_type == DetectorType.LABEL
    assert events[0].signal_type == TradeSignalType.SELL
    assert events[0].confidence == pytest.approx(0.95)


def test_imperfect_ocr_recognition() -> None:
    """Test 3: Common OCR typos (8UY, 5ELL, SEL1, BU¥) are correctly classified."""
    detector = LabelDetector(baseline_frames=0, ocr_engine=None)

    assert detector._classify_text("8UY") == TradeSignalType.BUY
    assert detector._classify_text("BU¥") == TradeSignalType.BUY
    assert detector._classify_text("LONG") == TradeSignalType.BUY
    assert detector._classify_text("5ELL") == TradeSignalType.SELL
    assert detector._classify_text("SEL1") == TradeSignalType.SELL
    assert detector._classify_text("SHORT") == TradeSignalType.SELL


def test_label_absent() -> None:
    """Test 4: When no label is present, zero events are emitted."""
    mock_ocr = MockOCR([])
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    events = detector.process_frame(make_dummy_frame())
    assert len(events) == 0


def test_existing_label_across_multiple_frames() -> None:
    """Test 5: Label remaining visible across multiple frames does not re-alert."""
    box = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    mock_ocr = MockOCR([(box, "BUY", 0.98)])

    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)

    # Frame 1: New label detected
    events1 = detector.process_frame(make_dummy_frame())
    assert len(events1) == 1

    # Frames 2-5: Same label continues to be visible
    for _ in range(4):
        events_subsequent = detector.process_frame(make_dummy_frame())
        assert len(events_subsequent) == 0, "Repeated label must not trigger duplicate events"


def test_newly_appearing_label_sequence() -> None:
    """Test 6: A new label appearing after an empty period triggers an event."""
    mock_ocr = MagicMock()
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)

    # Frame 1: Empty
    mock_ocr.return_value = ([], [])
    assert len(detector.process_frame(make_dummy_frame())) == 0

    # Frame 2: Newly appears
    box = [[300.0, 200.0], [360.0, 200.0], [360.0, 230.0], [300.0, 230.0]]
    mock_ocr.return_value = ([(box, "BUY", 0.96)], [])
    events2 = detector.process_frame(make_dummy_frame())
    assert len(events2) == 1
    assert events2[0].signal_type == TradeSignalType.BUY


def test_disappearing_and_reappearing_label() -> None:
    """Test 7: Label that disappears for > max_missing_frames is treated as new upon return."""
    box = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    mock_ocr = MagicMock()
    detector = LabelDetector(baseline_frames=0, max_missing_frames=3, ocr_engine=mock_ocr)

    # 1. Label appears
    mock_ocr.return_value = ([(box, "SELL", 0.95)], [])
    assert len(detector.process_frame(make_dummy_frame())) == 1

    # 2. Label disappears for 4 frames (> max_missing_frames=3)
    mock_ocr.return_value = ([], [])
    for _ in range(4):
        detector.process_frame(make_dummy_frame())

    # 3. Label reappears at same location
    mock_ocr.return_value = ([(box, "SELL", 0.95)], [])
    events_reappear = detector.process_frame(make_dummy_frame())
    assert len(events_reappear) == 1, "Reappearing label after expiration must trigger a new event"


def test_small_coordinate_jitter_tolerance() -> None:
    """Test 8: Small coordinate jitter (<= max_spatial_distance_px) does not produce duplicate events."""
    box1 = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    # Box 2 shifted slightly by 5px
    box2 = [[105.0, 103.0], [165.0, 103.0], [165.0, 133.0], [105.0, 133.0]]

    mock_ocr = MagicMock()
    detector = LabelDetector(baseline_frames=0, max_spatial_distance_px=30.0, ocr_engine=mock_ocr)

    # Frame 1
    mock_ocr.return_value = ([(box1, "BUY", 0.95)], [])
    assert len(detector.process_frame(make_dummy_frame())) == 1

    # Frame 2 with coordinate jitter
    mock_ocr.return_value = ([(box2, "BUY", 0.94)], [])
    assert len(detector.process_frame(make_dummy_frame())) == 0, "Coordinate jitter must be matched to existing label"


def test_multiple_labels_simultaneously() -> None:
    """Test 9: Multiple labels present simultaneously are tracked independently."""
    box_buy = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    box_sell = [[400.0, 300.0], [470.0, 300.0], [470.0, 330.0], [400.0, 330.0]]

    mock_ocr = MockOCR([(box_buy, "BUY", 0.96), (box_sell, "SELL", 0.94)])
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)

    events = detector.process_frame(make_dummy_frame())
    assert len(events) == 2
    types = {e.signal_type for e in events}
    assert types == {TradeSignalType.BUY, TradeSignalType.SELL}


def test_startup_baseline_suppression() -> None:
    """Test 10: Labels visible at startup are cataloged as baseline without emitting events."""
    box = [[100.0, 100.0], [160.0, 100.0], [160.0, 130.0], [100.0, 130.0]]
    mock_ocr = MagicMock()
    mock_ocr.return_value = ([(box, "BUY", 0.98)], [])

    # Set baseline calibration for 3 frames
    detector = LabelDetector(baseline_frames=3, ocr_engine=mock_ocr)

    # Frames 1-3: Baseline calibration period
    for i in range(1, 4):
        events = detector.process_frame(make_dummy_frame())
        assert len(events) == 0, f"Frame {i} should be suppressed during baseline calibration"
        assert detector.is_calibrated is (i >= 3)

    # Frame 4: Pre-existing label still visible - still NO event
    events_frame4 = detector.process_frame(make_dummy_frame())
    assert len(events_frame4) == 0, "Pre-existing baseline label must not alert after calibration"

    # Frame 5: Brand-new SELL label appears - SHOULD alert!
    new_box = [[500.0, 300.0], [570.0, 300.0], [570.0, 330.0], [500.0, 330.0]]
    mock_ocr.return_value = ([(box, "BUY", 0.98), (new_box, "SELL", 0.95)], [])
    events_frame5 = detector.process_frame(make_dummy_frame())
    assert len(events_frame5) == 1
    assert events_frame5[0].signal_type == TradeSignalType.SELL


def test_false_positive_text_rejection() -> None:
    """Test 11: Static chart zones and tickers are rejected as false positives."""
    detector = LabelDetector(baseline_frames=0, ocr_engine=None)

    # Static zone headers
    assert detector._classify_text("SellZone (Supply)") is None
    assert detector._classify_text("Buy Zone (Demand)") is None
    assert detector._classify_text("Buy Zone") is None
    assert detector._classify_text("Sell Zone") is None

    # News and market headers
    assert detector._classify_text("MARKET NEWS") is None
    assert detector._classify_text("LIVE PULSE") is None
    assert detector._classify_text("REAL-TIME UTC") is None
    assert detector._classify_text("XAUUSD") is None

    # Price levels and Take Profit
    assert detector._classify_text("TP1 (4072.2082)") is None
    assert detector._classify_text("TP2") is None
    assert detector._classify_text("TP3(4081.3446)") is None
    assert detector._classify_text("Entry (4067.64)") is None
    assert detector._classify_text("4,066.80") is None


def test_different_image_resolutions() -> None:
    """Test 12: Scaling factor adapts coordinate translation properly."""
    # Local box at 2x scale: [200, 200] to [300, 250]
    scaled_box = [[200.0, 200.0], [300.0, 200.0], [300.0, 250.0], [200.0, 250.0]]
    mock_ocr = MockOCR([(scaled_box, "BUY", 0.95)])

    # Scale factor 2.0: coordinates should be halved back to original scale
    detector = LabelDetector(baseline_frames=0, scale_factor=2.0, ocr_engine=mock_ocr)
    events = detector.process_frame(make_dummy_frame())

    assert len(events) == 1
    # 200 / 2.0 = 100; (300 - 200) / 2.0 = 50
    assert events[0].bbox.x == 100
    assert events[0].bbox.y == 150  # 100 + offset_y(50)
    assert events[0].bbox.width == 50
    assert events[0].bbox.height == 25


def test_actual_captured_frame_detection() -> None:
    """Test 13: Test detector against actual captured calibration frame.

    Verifies that the static 'Current Position: Buy' widget at [992, 401] is correctly
    suppressed by Phase 8C static HUD exclusion and contextual filtering.
    """
    calib_frame_path = Path("captures/calibration/calibration_frame_1.png")
    if not calib_frame_path.exists():
        pytest.skip("Calibration frame not present")

    import cv2
    img = cv2.imread(str(calib_frame_path))
    assert img is not None

    frame = Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=img.shape[1],
        height=img.shape[0],
        data=img,
    )

    # Run default detector: the static 'Buy' widget at [992, 401] is correctly rejected by HUD filter
    detector = LabelDetector(baseline_frames=0, min_confidence=0.75)
    events = detector.process_frame(frame)
    assert len(events) == 0, "Static Current Position widget at [992, 401] must be suppressed"

    # If HUD exclusion is explicitly disabled, the raw OCR token can be detected
    unfiltered_detector = LabelDetector(
        baseline_frames=0, min_confidence=0.75, excluded_rois=[]
    )
    # Still rejected by contextual HUD keyword check ("Current Position" adjacent)
    unfiltered_events = unfiltered_detector.process_frame(frame)
    assert len(unfiltered_events) == 0, "Contextual HUD filter must reject 'Current Position' widget"

