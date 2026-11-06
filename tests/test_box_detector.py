"""Comprehensive unit tests for Detector B: Trade-Position Box Detection.

Tests cover all 15 required scenarios:
  1. Valid BUY position box (green profit zone above red loss zone).
  2. Valid SELL position box (red loss zone above green profit zone).
  3. Green and red regions without valid combined structure.
  4. Rectangle without entry line (unpaired green or red zone).
  5. Ordinary candlesticks (narrow width/aspect ratio rejection).
  6. Ordinary horizontal price lines (thin height rejection).
  7. Existing position box across multiple frames (deduplication).
  8. Newly appearing position box sequence.
  9. Disappearing and reappearing box (re-alerting after missing frames).
  10. Small box movement and resizing (continuity tracking).
  11. Multiple colored chart elements with a valid box.
  12. Startup baseline suppression (silent cataloging).
  13. Different image resolutions and custom ROI.
  14. False-positive rejection (blank, noise, plain chart).
  15. Actual captured calibration frames from the live stream.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pytest

from core.models import BoundingBox, DetectionEvent, DetectorType, Frame, TradeSignalType
from detectors.box_detector import BoxDetector


# ---------------------------------------------------------------------------
# Synthetic Frame Helpers
# ---------------------------------------------------------------------------

def create_chart_frame(
    width: int = 1280,
    height: int = 720,
    bg_color: Tuple[int, int, int] = (24, 20, 18),  # Dark TradingView BGR
) -> Frame:
    """Create a synthetic chart frame with dark background."""
    img = np.full((height, width, 3), bg_color, dtype=np.uint8)
    return Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=width,
        height=height,
        data=img,
    )


def draw_position_box(
    img: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    signal_type: TradeSignalType,
    profit_ratio: float = 0.60,
    green_bgr: Tuple[int, int, int] = (0, 180, 0),
    red_bgr: Tuple[int, int, int] = (0, 0, 180),
) -> int:
    """Draw a trade position box with risk/reward zones on the image.

    Returns the entry line y-coordinate.
    """
    profit_h = int(height * profit_ratio)
    loss_h = height - profit_h

    if signal_type == TradeSignalType.BUY:
        # Green (profit) on top, Red (loss) below
        cv2.rectangle(img, (x, y), (x + width - 1, y + profit_h - 1), green_bgr, -1)
        entry_y = y + profit_h
        cv2.rectangle(img, (x, entry_y), (x + width - 1, y + height - 1), red_bgr, -1)
    else:
        # Red (loss) on top, Green (profit) below
        cv2.rectangle(img, (x, y), (x + width - 1, y + loss_h - 1), red_bgr, -1)
        entry_y = y + loss_h
        cv2.rectangle(img, (x, entry_y), (x + width - 1, y + height - 1), green_bgr, -1)

    return entry_y


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_valid_buy_position_box() -> None:
    """Scenario 1: Valid BUY position box (green profit zone above red loss zone)."""
    frame = create_chart_frame()
    entry_y = draw_position_box(frame.data, x=200, y=200, width=120, height=100, signal_type=TradeSignalType.BUY)

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)

    assert len(events) == 1
    event = events[0]
    assert event.detector_type == DetectorType.BOX
    assert event.signal_type == TradeSignalType.BUY
    assert event.bbox.x == pytest.approx(200, abs=2)
    assert event.bbox.y == pytest.approx(200, abs=2)
    assert event.bbox.width == pytest.approx(120, abs=2)
    assert event.bbox.height == pytest.approx(100, abs=2)
    assert event.raw_metadata["entry_y"] == pytest.approx(entry_y, abs=2)
    assert event.confidence >= 0.70


def test_valid_sell_position_box() -> None:
    """Scenario 2: Valid SELL position box (red loss zone above green profit zone)."""
    frame = create_chart_frame()
    entry_y = draw_position_box(frame.data, x=300, y=180, width=140, height=120, signal_type=TradeSignalType.SELL)

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)

    assert len(events) == 1
    event = events[0]
    assert event.detector_type == DetectorType.BOX
    assert event.signal_type == TradeSignalType.SELL
    assert event.bbox.x == pytest.approx(300, abs=2)
    assert event.bbox.y == pytest.approx(180, abs=2)
    assert event.bbox.width == pytest.approx(140, abs=2)
    assert event.bbox.height == pytest.approx(120, abs=2)
    assert event.raw_metadata["entry_y"] == pytest.approx(entry_y, abs=2)


def test_green_and_red_regions_without_valid_combined_structure() -> None:
    """Scenario 3: Separated green and red regions that do not form a trade position."""
    frame = create_chart_frame()
    # Green in top-left, red in bottom-right (no horizontal overlap)
    cv2.rectangle(frame.data, (100, 100), (200, 160), (0, 180, 0), -1)
    cv2.rectangle(frame.data, (600, 400), (700, 460), (0, 0, 180), -1)

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)
    assert len(events) == 0

    # Vertically separated by 200px gap (too far vertically to share entry)
    frame2 = create_chart_frame()
    cv2.rectangle(frame2.data, (200, 100), (300, 160), (0, 180, 0), -1)
    cv2.rectangle(frame2.data, (200, 360), (300, 420), (0, 0, 180), -1)

    events2 = detector.process_frame(frame2)
    assert len(events2) == 0


def test_rectangle_without_entry_line_unpaired() -> None:
    """Scenario 4: Lone green or red rectangle without opposing risk/reward zone."""
    detector = BoxDetector(baseline_frames=0)

    # Lone green zone (e.g. demand zone banner)
    frame_green = create_chart_frame()
    cv2.rectangle(frame_green.data, (150, 200), (350, 280), (0, 180, 0), -1)
    events_g = detector.process_frame(frame_green)
    assert len(events_g) == 0

    # Lone red zone (e.g. supply zone banner)
    frame_red = create_chart_frame()
    cv2.rectangle(frame_red.data, (150, 200), (350, 280), (0, 0, 180), -1)
    events_r = detector.process_frame(frame_red)
    assert len(events_r) == 0


def test_ordinary_candlesticks_rejected() -> None:
    """Scenario 5: Narrow candlesticks (width <= 8px) are filtered out."""
    frame = create_chart_frame()
    # Draw several green and red candles with narrow bodies and wicks
    for i, x in enumerate(range(100, 500, 25)):
        color = (0, 180, 0) if i % 2 == 0 else (0, 0, 180)
        # Candle body (width=6px, height=35px)
        cv2.rectangle(frame.data, (x, 250), (x + 6, 285), color, -1)
        # Candle wick (width=1px, height=60px)
        cv2.line(frame.data, (x + 3, 230), (x + 3, 290), color, 1)

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)
    assert len(events) == 0


def test_ordinary_horizontal_price_lines_rejected() -> None:
    """Scenario 6: Thin horizontal price/grid lines (height <= 3px) are filtered out."""
    frame = create_chart_frame()
    # Draw thin horizontal price levels across the chart
    cv2.rectangle(frame.data, (50, 200), (850, 202), (0, 180, 0), -1)  # 2px high green
    cv2.rectangle(frame.data, (50, 350), (850, 352), (0, 0, 180), -1)  # 2px high red

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)
    assert len(events) == 0


def test_existing_position_box_across_multiple_frames_no_duplicate() -> None:
    """Scenario 7: Continuous presence of position box triggers only 1 initial alert."""
    frame = create_chart_frame()
    draw_position_box(frame.data, x=250, y=200, width=100, height=80, signal_type=TradeSignalType.BUY)

    detector = BoxDetector(baseline_frames=0)

    # Frame 1: Initial appearance emits event
    events_1 = detector.process_frame(frame)
    assert len(events_1) == 1

    # Frames 2-6: Same box remains visible -> deduplicated, 0 new events
    for _ in range(5):
        events_n = detector.process_frame(frame)
        assert len(events_n) == 0


def test_newly_appearing_position_box_sequence() -> None:
    """Scenario 8: Sequence of empty frames followed by a new box produces exactly 1 event."""
    empty_frame = create_chart_frame()
    detector = BoxDetector(baseline_frames=0)

    # Initial empty frames
    for _ in range(3):
        events = detector.process_frame(empty_frame)
        assert len(events) == 0

    # New trade position box appears
    active_frame = create_chart_frame()
    draw_position_box(active_frame.data, x=400, y=250, width=130, height=90, signal_type=TradeSignalType.SELL)

    events_active = detector.process_frame(active_frame)
    assert len(events_active) == 1
    assert events_active[0].signal_type == TradeSignalType.SELL

    # Subsequent frame: no duplicate
    events_next = detector.process_frame(active_frame)
    assert len(events_next) == 0


def test_disappearing_and_reappearing_box() -> None:
    """Scenario 9: Disappearing box pruned after max_missing_frames; reappearance re-alerts."""
    trade_frame = create_chart_frame()
    draw_position_box(trade_frame.data, x=250, y=200, width=100, height=80, signal_type=TradeSignalType.BUY)
    empty_frame = create_chart_frame()

    detector = BoxDetector(baseline_frames=0, max_missing_frames=5)

    # Appearance emits 1 event
    events_1 = detector.process_frame(trade_frame)
    assert len(events_1) == 1

    # Missing for 6 frames (> max_missing_frames=5)
    for _ in range(6):
        detector.process_frame(empty_frame)

    # Reappearance triggers a fresh new-trade alert
    events_reappear = detector.process_frame(trade_frame)
    assert len(events_reappear) == 1
    assert events_reappear[0].signal_type == TradeSignalType.BUY


def test_small_box_movement_and_resizing_continuity() -> None:
    """Scenario 10: Box expanding horizontally with new candles is tracked without duplicate alerts."""
    detector = BoxDetector(baseline_frames=0)

    # Frame 1: Initial position box (width = 80)
    frame1 = create_chart_frame()
    draw_position_box(frame1.data, x=200, y=200, width=80, height=90, signal_type=TradeSignalType.BUY)
    events1 = detector.process_frame(frame1)
    assert len(events1) == 1

    # Frame 2: Box extended horizontally to width = 110 (as candle advances right)
    frame2 = create_chart_frame()
    draw_position_box(frame2.data, x=200, y=200, width=110, height=90, signal_type=TradeSignalType.BUY)
    events2 = detector.process_frame(frame2)
    assert len(events2) == 0, "Horizontal expansion of active trade must not trigger duplicate event"

    # Frame 3: Box extended further to width = 140
    frame3 = create_chart_frame()
    draw_position_box(frame3.data, x=200, y=200, width=140, height=90, signal_type=TradeSignalType.BUY)
    events3 = detector.process_frame(frame3)
    assert len(events3) == 0


def test_multiple_colored_chart_elements_with_valid_box() -> None:
    """Scenario 11: Background grid, candlesticks, and indicators present alongside valid box."""
    frame = create_chart_frame()

    # Noise 1: Grid lines
    for gy in range(100, 600, 100):
        cv2.line(frame.data, (0, gy), (1050, gy), (35, 30, 28), 1)

    # Noise 2: Candlesticks
    for cx in range(80, 180, 15):
        cv2.rectangle(frame.data, (cx, 300), (cx + 5, 330), (0, 180, 0), -1)

    # Noise 3: Red supply zone banner at the very top (unpaired)
    cv2.rectangle(frame.data, (50, 70), (400, 95), (0, 0, 180), -1)

    # Valid BUY position box
    draw_position_box(frame.data, x=300, y=220, width=120, height=100, signal_type=TradeSignalType.BUY)

    detector = BoxDetector(baseline_frames=0)
    events = detector.process_frame(frame)

    assert len(events) == 1
    assert events[0].signal_type == TradeSignalType.BUY
    assert events[0].bbox.x == 300
    assert events[0].bbox.y == 220


def test_startup_baseline_suppression() -> None:
    """Scenario 12: Trades present at startup are cataloged as baseline and suppressed."""
    detector = BoxDetector(baseline_frames=3)

    # Startup frame with an existing BUY position box
    baseline_frame = create_chart_frame()
    draw_position_box(baseline_frame.data, x=200, y=200, width=100, height=80, signal_type=TradeSignalType.BUY)

    # Frames 1-3: Baseline calibration
    for i in range(1, 4):
        events = detector.process_frame(baseline_frame)
        assert len(events) == 0, f"Baseline frame {i} must not emit trade notification"

    assert detector.is_calibrated is True

    # Frame 4: Baseline trade still present -> still suppressed
    events_4 = detector.process_frame(baseline_frame)
    assert len(events_4) == 0

    # Frame 5: Brand-new second trade (SELL) appears
    new_trade_frame = create_chart_frame()
    draw_position_box(new_trade_frame.data, x=200, y=200, width=100, height=80, signal_type=TradeSignalType.BUY)
    draw_position_box(new_trade_frame.data, x=500, y=180, width=120, height=90, signal_type=TradeSignalType.SELL)

    events_5 = detector.process_frame(new_trade_frame)
    assert len(events_5) == 1
    assert events_5[0].signal_type == TradeSignalType.SELL
    assert events_5[0].bbox.x == 500


def test_different_image_resolutions() -> None:
    """Scenario 13: Full HD (1920x1080) frame with adjusted custom ROI."""
    frame_fhd = create_chart_frame(width=1920, height=1080)
    entry_y = draw_position_box(
        frame_fhd.data,
        x=350,
        y=300,
        width=180,
        height=140,
        signal_type=TradeSignalType.SELL,
    )

    detector = BoxDetector(
        roi=(0, 90, 1500, 950),
        baseline_frames=0,
    )
    events = detector.process_frame(frame_fhd)

    assert len(events) == 1
    assert events[0].signal_type == TradeSignalType.SELL
    assert events[0].bbox.x == pytest.approx(350, abs=2)
    assert events[0].bbox.y == pytest.approx(300, abs=2)
    assert events[0].bbox.width == pytest.approx(180, abs=2)
    assert events[0].bbox.height == pytest.approx(140, abs=2)


def test_false_positive_rejection() -> None:
    """Scenario 14: Blank, all-black, all-white, and random noise frames produce 0 events."""
    detector = BoxDetector(baseline_frames=0)

    # 1. All black
    black_frame = create_chart_frame(bg_color=(0, 0, 0))
    assert len(detector.process_frame(black_frame)) == 0

    # 2. All white
    white_frame = create_chart_frame(bg_color=(255, 255, 255))
    assert len(detector.process_frame(white_frame)) == 0

    # 3. Random noise
    noise_data = np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)
    noise_frame = Frame(1, datetime.now(timezone.utc), 1280, 720, noise_data)
    assert len(detector.process_frame(noise_frame)) == 0

    # 4. None data
    none_frame = Frame(1, datetime.now(timezone.utc), 1280, 720, None)
    assert len(detector.process_frame(none_frame)) == 0


def test_actual_captured_calibration_frames() -> None:
    """Scenario 15: Run Detector B on the actual captured calibration screenshots."""
    calibration_path = Path("captures/calibration/calibration_frame_1.png")
    if not calibration_path.exists():
        pytest.skip("calibration_frame_1.png not present in calibration directory")

    img = cv2.imread(str(calibration_path))
    assert img is not None, "Failed to load calibration screenshot"

    frame = Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=img.shape[1],
        height=img.shape[0],
        data=img,
    )

    # Verify single-shot detection
    detector = BoxDetector(roi=(0, 60, 920, 600), baseline_frames=0)
    events = detector.process_frame(frame)

    assert len(events) >= 1
    primary_event = events[0]
    assert primary_event.detector_type == DetectorType.BOX
    assert primary_event.signal_type == TradeSignalType.BUY
    assert primary_event.confidence >= 0.70
    assert primary_event.bbox.x > 0
    assert primary_event.bbox.width >= 35

    # Verify that sending the same frame again does not produce duplicate alerts
    duplicate_events = detector.process_frame(frame)
    assert len(duplicate_events) == 0, "Subsequent identical frame must be deduplicated"
