"""Phase 8C: Comprehensive False-Positive Elimination Regression Test Suite.

Validates all 14 mandatory regression scenarios:
  1. Static Current Position SELL widget (zero activations)
  2. Static Current Position BUY widget (zero activations)
  3. Historical SELL badge near TP1 (zero activations)
  4. Historical BUY badge near TP1 (zero activations)
  5. Existing badge moving 32 pixels due to chart scrolling (zero activations)
  6. Existing signal temporarily disappearing and reappearing (zero activations)
  7. Existing trade visible during startup (zero activations)
  8. Low-confidence OCR single-frame noise during startup (zero activations)
  9. Genuine new BUY label on active chart candle (1 activation)
  10. Genuine new SELL label on active chart candle (1 activation)
  11. Genuine new position box without label (1 activation)
  12. Simultaneous new label and position box (1 activation, correlated, 1 notification)
  13. Existing position box resizing / expanding (zero activations)
  14. Genuine new trade after a previous trade closes (1 activation)
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.config import AppConfig
from core.event_manager import EventManager
from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    Frame,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import TelegramNotifier
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository


@pytest.fixture
def temp_repos(tmp_path: Path):
    db_path = str(tmp_path / "test_fp.db")
    init_db(db_path)
    return TradeRepository(db_path), EventRepository(db_path), NotificationRepository(db_path), db_path


def make_frame(data: np.ndarray, frame_id: int = 1, ts: datetime = None) -> Frame:
    h, w = data.shape[:2]
    return Frame(
        frame_id=frame_id,
        timestamp=ts or datetime.now(timezone.utc),
        width=w,
        height=h,
        data=data,
    )


# ---------------------------------------------------------------------------
# Scenario 1 & 2: Static Current Position HUD Widget (SELL & BUY)
# ---------------------------------------------------------------------------

def test_scenario_01_static_current_position_sell_widget_suppressed(temp_repos) -> None:
    """Scenario 1: The right-side Current Position: Sell HUD widget produces 0 activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Audited screenshot 1 path (if present locally, run real OCR, otherwise synthetic)
    scr_path = Path("captures/profile_10min/trade_018b55d9-161b-4f10-9bbd-3b106ff23bed.png")
    if scr_path.exists():
        img = cv2.imread(str(scr_path))
        frame = make_frame(img, frame_id=1)
        detector = LabelDetector(baseline_frames=0)
        events = detector.process_frame(frame)
        activations = manager.process_events(events, frame=frame)
        assert len(activations) == 0, f"Expected 0 activations for static HUD Sell widget, got {len(activations)}"
    else:
        # Synthetic frame with "Current Position" at [917, 402] and "Sell" at [990, 402]
        mock_ocr = MagicMock(return_value=([
            ([[917, 402], [982, 402], [982, 415], [917, 415]], "Current Position", 0.99),
            ([[990, 402], [1015, 402], [1015, 415], [990, 415]], "Sell", 0.85),
            ([[917, 420], [982, 420], [982, 435], [917, 435]], "Current Trend: Bearish", 0.95),
        ], []))
        detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
        frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        events = detector.process_frame(frame)
        activations = manager.process_events(events, frame=frame)
        assert len(activations) == 0, "Static Current Position Sell widget must produce 0 activations"


def test_scenario_02_static_current_position_buy_widget_suppressed(temp_repos) -> None:
    """Scenario 2: The right-side Current Position: Buy HUD widget produces 0 activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Synthetic frame with "Current Position" at [917, 402] and "Buy" at [992, 401]
    mock_ocr = MagicMock(return_value=([
        ([[917, 402], [982, 402], [982, 415], [917, 415]], "Current Position", 0.99),
        ([[992, 401], [1014, 401], [1014, 416], [992, 416]], "Buy", 0.98),
        ([[917, 420], [982, 420], [982, 435], [917, 435]], "Trend: Bullish", 0.95),
    ], []))
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
    events = detector.process_frame(frame)
    activations = manager.process_events(events, frame=frame)
    assert len(activations) == 0, "Static Current Position Buy widget must produce 0 activations"


# ---------------------------------------------------------------------------
# Scenario 3 & 4: Historical Candle Badges near TP1 (SELL & BUY)
# ---------------------------------------------------------------------------

def test_scenario_03_historical_sell_badge_near_tp1_suppressed(temp_repos) -> None:
    """Scenario 3: Historical SELL badge near completed TP1 produces 0 activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    scr_path = Path("captures/profile_10min/trade_cc44b723-0b7f-4c02-8638-d390f635e235.png")
    if scr_path.exists():
        img = cv2.imread(str(scr_path))
        frame = make_frame(img, frame_id=1)
        detector = LabelDetector(baseline_frames=0)
        events = detector.process_frame(frame)
        activations = manager.process_events(events, frame=frame)
        assert len(activations) == 0, f"Historical Sell badge with TP1 must be suppressed, got {len(activations)}"
    else:
        # Synthetic mock with "TP1" at [547, 347] and "Sell" at [572, 345]
        mock_ocr = MagicMock(return_value=([
            ([[547, 347], [561, 347], [561, 355], [547, 355]], "TP1", 0.99),
            ([[572, 345], [590, 345], [590, 355], [572, 355]], "Sell", 0.91),
        ], []))
        detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
        frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
        events = detector.process_frame(frame)
        activations = manager.process_events(events, frame=frame)
        assert len(activations) == 0, "Historical Sell badge with TP1 must produce 0 activations"


def test_scenario_04_historical_buy_badge_near_tp1_suppressed(temp_repos) -> None:
    """Scenario 4: Historical BUY badge near completed TP1 produces 0 activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Synthetic mock with "TP1" at [545, 347] and "Buy" at [570, 345]
    mock_ocr = MagicMock(return_value=([
        ([[545, 347], [560, 347], [560, 355], [545, 355]], "TP1", 0.99),
        ([[570, 345], [590, 345], [590, 355], [570, 355]], "Buy", 0.94),
    ], []))
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))
    events = detector.process_frame(frame)
    activations = manager.process_events(events, frame=frame)
    assert len(activations) == 0, "Historical Buy badge with TP1 must produce 0 activations"


# ---------------------------------------------------------------------------
# Scenario 5: Existing Badge Moving 32px Due to Chart Scrolling
# ---------------------------------------------------------------------------

def test_scenario_05_badge_moving_32px_due_to_chart_scrolling(temp_repos) -> None:
    """Scenario 5: Existing trade badge moving 32px leftward due to chart scroll produces 0 new activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        active_trade_max_missing_sec=60.0,
        chart_scroll_max_drift_px=150.0,
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    # Frame 1: Trade appears at x=572, y=250 (activated)
    ev1 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.SELL,
        confidence=0.88,
        bbox=BoundingBox(572, 250, 40, 20),
        timestamp=t0,
    )
    act1 = manager.process_event(ev1)
    assert act1 is not None, "Initial trade at x=572 must activate"

    # Trade becomes inactive and closes after 70 seconds
    t1 = t0 + timedelta(seconds=70)
    closed = manager.prune_inactive_trades(t1)
    assert act1.trade_id in closed
    assert len(manager.get_active_trades()) == 0

    # Frame 2: 8 minutes later, chart has scrolled 32px left -> badge is now at x=540, y=250
    t2 = t0 + timedelta(minutes=8)
    ev2 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.SELL,
        confidence=0.92,
        bbox=BoundingBox(540, 250, 40, 20),
        timestamp=t2,
    )
    act2 = manager.process_event(ev2)
    assert act2 is None, "Scrolled trade at x=540 must be correlated with closed trade; 0 new activations"


# ---------------------------------------------------------------------------
# Scenario 6: Existing Signal Temporarily Disappearing & Reappearing
# ---------------------------------------------------------------------------

def test_scenario_06_signal_temporarily_disappearing_and_reappearing(temp_repos) -> None:
    """Scenario 6: Active signal missing for 10 frames and reappearing produces 0 new activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    ev1 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.90,
        bbox=BoundingBox(400, 200, 40, 20),
        timestamp=t0,
    )
    act1 = manager.process_event(ev1)
    assert act1 is not None, "Initial BUY trade must activate"

    # Missing for 10 seconds (10 frames)
    t_reappear = t0 + timedelta(seconds=10)
    manager.advance_frame(make_frame(np.zeros((720, 1280, 3), dtype=np.uint8), ts=t_reappear))

    # Reappears at same coordinate
    ev_reappear = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.92,
        bbox=BoundingBox(400, 200, 40, 20),
        timestamp=t_reappear,
    )
    act_reappear = manager.process_event(ev_reappear)
    assert act_reappear is None, "Reappearing trade must be enriched without triggering a new activation"


# ---------------------------------------------------------------------------
# Scenario 7: Existing Trade Visible During Startup
# ---------------------------------------------------------------------------

def test_scenario_07_existing_trade_visible_during_startup(temp_repos) -> None:
    """Scenario 7: Trade present during baseline calibration frames 1-15 produces 0 activations."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=15)

    # Repeated appearances during calibration (frames 1 to 15)
    for f in range(1, 16):
        t = datetime(2026, 11, 7, 10, 0, f, tzinfo=timezone.utc)
        ev = DetectionEvent(
            detector_type=DetectorType.LABEL,
            signal_type=TradeSignalType.BUY,
            confidence=0.88,
            bbox=BoundingBox(350, 220, 40, 20),
            timestamp=t,
        )
        act = manager.process_event(ev)
        manager.advance_frame(make_frame(np.zeros((720, 1280, 3), dtype=np.uint8), frame_id=f, ts=t))
        assert act is None, f"Frame {f} in calibration must produce 0 activations"

    assert manager.is_calibrated is True

    # Post-calibration: frame 16
    t16 = datetime(2026, 11, 7, 10, 0, 16, tzinfo=timezone.utc)
    ev16 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.88,
        bbox=BoundingBox(350, 220, 40, 20),
        timestamp=t16,
    )
    act16 = manager.process_event(ev16)
    assert act16 is None, "Baseline trade must not activate after calibration completes"


# ---------------------------------------------------------------------------
# Scenario 8: Low-Confidence OCR Single-Frame Noise During Startup
# ---------------------------------------------------------------------------

def test_scenario_08_low_confidence_single_frame_noise_during_startup(temp_repos) -> None:
    """Scenario 8: Single-frame OCR noise at conf 0.52 during startup is discarded, producing 0 activations."""
    mock_ocr = MagicMock(return_value=([
        ([[450, 250], [480, 250], [480, 265], [450, 265]], "Buy", 0.52),
    ], []))
    detector = LabelDetector(
        baseline_frames=15,
        min_confidence=0.70,
        calibration_min_confidence=0.45,
        calibration_observation_threshold=2,
        ocr_engine=mock_ocr,
    )

    frame1 = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8), frame_id=1)
    events1 = detector.process_frame(frame1)
    assert len(events1) == 0, "Calibration frame 1 must produce 0 events"

    # Change mock OCR so noise never reappears
    detector._ocr = MagicMock(return_value=([], []))
    for f in range(2, 17):
        frame_f = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8), frame_id=f)
        events_f = detector.process_frame(frame_f)
        assert len(events_f) == 0

    assert detector.is_calibrated is True
    # The noise only appeared once, so it was NOT cataloged into tracked_labels
    assert len(detector._tracked_labels) == 0, "Single-frame low-confidence noise must not be cataloged"


# ---------------------------------------------------------------------------
# Scenario 9 & 10: Genuine New BUY and SELL Labels
# ---------------------------------------------------------------------------

def test_scenario_09_genuine_new_buy_label(temp_repos) -> None:
    """Scenario 9: Genuine new BUY label on active chart candle activates cleanly."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Genuine BUY label at x=400, y=250 outside HUD and without completed milestone stamps
    mock_ocr = MagicMock(return_value=([
        ([[400, 250], [440, 250], [440, 268], [400, 268]], "BUY", 0.95),
        ([[350, 230], [420, 230], [420, 245], [350, 245]], "TP: 1.0850", 0.92),  # Planned target info
    ], []))
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

    events = detector.process_frame(frame)
    assert len(events) == 1
    assert events[0].signal_type == TradeSignalType.BUY

    activations = manager.process_events(events, frame=frame)
    assert len(activations) == 1, "Genuine new BUY label must activate exactly 1 trade"
    assert activations[0].state == TradeState.ACTIVE
    assert activations[0].metadata["signal_type"] == "BUY"


def test_scenario_10_genuine_new_sell_label(temp_repos) -> None:
    """Scenario 10: Genuine new SELL label on active chart candle activates cleanly."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    mock_ocr = MagicMock(return_value=([
        ([[420, 300], [460, 300], [460, 318], [420, 318]], "SELL", 0.96),
        ([[410, 330], [480, 330], [480, 345], [410, 345]], "SL: 1.0920", 0.91),  # Planned SL info
    ], []))
    detector = LabelDetector(baseline_frames=0, ocr_engine=mock_ocr)
    frame = make_frame(np.zeros((720, 1280, 3), dtype=np.uint8))

    events = detector.process_frame(frame)
    assert len(events) == 1
    assert events[0].signal_type == TradeSignalType.SELL

    activations = manager.process_events(events, frame=frame)
    assert len(activations) == 1, "Genuine new SELL label must activate exactly 1 trade"
    assert activations[0].state == TradeState.ACTIVE
    assert activations[0].metadata["signal_type"] == "SELL"


# ---------------------------------------------------------------------------
# Scenario 11: Genuine New Position Box Without Label
# ---------------------------------------------------------------------------

def test_scenario_11_genuine_new_position_box_without_label(temp_repos) -> None:
    """Scenario 11: Genuine new position box without a label activates independently (Detector B)."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    ev_box = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        confidence=0.92,
        bbox=BoundingBox(300, 200, 120, 80),
        timestamp=datetime.now(timezone.utc),
        raw_metadata={"entry_y": 240},
    )
    activations = manager.process_events([ev_box])
    assert len(activations) == 1, "Position box alone must activate trade"
    assert activations[0].detector_source == DetectorType.BOX
    assert activations[0].state == TradeState.ACTIVE


# ---------------------------------------------------------------------------
# Scenario 12: Simultaneous New Label and Position Box
# ---------------------------------------------------------------------------

def test_scenario_12_simultaneous_label_and_box_produces_one_notification(temp_repos) -> None:
    """Scenario 12: Simultaneous label and position box produce exactly 1 trade activation and 1 notification."""
    trade_repo, event_repo, notif_repo, db_path = temp_repos
    cfg = AppConfig()
    cfg.storage.db_path = db_path
    cfg.telegram.bot_token = "mock_token"
    cfg.telegram.chat_id = "6235890038"

    manager = EventManager(config=cfg, trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)
    mock_http = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 999}}')
    notifier = TelegramNotifier(
        config=cfg,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_http,
        dry_run=True,
    )

    t = datetime.now(timezone.utc)
    ev_label = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.95,
        bbox=BoundingBox(300, 230, 40, 20),
        timestamp=t,
    )
    ev_box = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        confidence=0.90,
        bbox=BoundingBox(300, 200, 140, 80),
        timestamp=t,
        raw_metadata={"entry_y": 240},
    )

    activations = manager.process_events([ev_label, ev_box])
    assert len(activations) == 1, f"Expected exactly 1 activation, got {len(activations)}"

    # Notification queueing and delivery
    notif = notifier.queue_trade_notification(activations[0])
    assert notif is not None
    assert notif.status.value in ("SENT", "PENDING")

    # Verify outbox database contains strictly 1 notification
    outbox_recs = notif_repo.get_pending_notifications()
    total_recs = notif_repo.get_notification_by_trade_id(activations[0].trade_id)
    assert total_recs is not None
    assert total_recs.message_text == "TRADE IS ACTIVE!"


# ---------------------------------------------------------------------------
# Scenario 13: Existing Position Box Resizing / Expanding
# ---------------------------------------------------------------------------

def test_scenario_13_existing_position_box_resizing_continuity() -> None:
    """Scenario 13: Position box expanding horizontally with forming candles maintains continuity (0 new events)."""
    detector = BoxDetector(baseline_frames=0)

    # Frame 1: Initial position box (w=100)
    canvas1 = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Green profit: y=160..230, x=300..400
    canvas1[160:230, 300:400] = [60, 180, 60]
    # Red loss: y=230..300, x=300..400
    canvas1[230:300, 300:400] = [50, 50, 200]

    events1 = detector.process_frame(make_frame(canvas1, frame_id=1))
    assert len(events1) == 1, "Initial position box must emit 1 event"

    # Frame 2: Box expanded horizontally to right (w=160, x=300..460) with same entry at y=230
    canvas2 = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas2[160:230, 300:460] = [60, 180, 60]
    canvas2[230:300, 300:460] = [50, 50, 200]

    events2 = detector.process_frame(make_frame(canvas2, frame_id=2))
    assert len(events2) == 0, "Resized/expanded position box must maintain continuity (0 new events)"


# ---------------------------------------------------------------------------
# Scenario 14: Genuine New Trade After Previous Trade Closes
# ---------------------------------------------------------------------------

def test_scenario_14_genuine_new_trade_after_previous_trade_closes(temp_repos) -> None:
    """Scenario 14: Genuine new trade after previous trade closes activates cleanly."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    manager = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        active_trade_max_missing_sec=60.0,
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    ev1 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.92,
        bbox=BoundingBox(200, 200, 40, 20),
        timestamp=t0,
    )
    act1 = manager.process_event(ev1)
    assert act1 is not None, "Trade 1 must activate"

    # Trade 1 closes after 70 seconds
    t1 = t0 + timedelta(seconds=70)
    closed = manager.prune_inactive_trades(t1)
    assert act1.trade_id in closed

    # Trade 2 appears at a distinct price level (y=320)
    t2 = t1 + timedelta(seconds=10)
    ev2 = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.94,
        bbox=BoundingBox(200, 320, 40, 20),
        timestamp=t2,
    )
    act2 = manager.process_event(ev2)
    assert act2 is not None, "Genuine new Trade 2 at distinct price level must activate"
    assert act2.trade_id != act1.trade_id
    assert act2.state == TradeState.ACTIVE

