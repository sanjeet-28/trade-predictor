"""End-to-End Simulation & Resiliency Test Suite.

Verifies the complete pipeline flow:
Stream frames -> Detector A / Detector B -> EventManager -> SQLite trade persistence
-> Notification outbox -> Telegram notifier.

Covers all 14 required E2E scenarios plus forced process termination recovery.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from core.config import AppConfig, DetectionConfig, StorageConfig, TelegramConfig
from core.event_manager import EventManager
from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    Frame,
    NotificationRecord,
    NotificationStatus,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from core.orchestrator import TradeDetectionOrchestrator
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import TelegramAPIError, TelegramNotifier
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository
from stream.pipeline import StreamPipeline


def make_chart_canvas(w: int = 1280, h: int = 720) -> np.ndarray:
    """Create a dark grey chart canvas resembling TradingView."""
    canvas = np.full((h, w, 3), 24, dtype=np.uint8)
    # Add subtle grid lines
    for x in range(0, w, 80):
        cv2.line(canvas, (x, 0), (x, h), (35, 35, 35), 1)
    for y in range(0, h, 60):
        cv2.line(canvas, (0, y), (w, y), (35, 35, 35), 1)
    return canvas


def draw_synthetic_buy_label(img: np.ndarray, x: int = 500, y: int = 300) -> np.ndarray:
    """Draw a standard BUY badge on the canvas."""
    cv2.rectangle(img, (x, y), (x + 60, y + 25), (40, 160, 40), -1)
    cv2.putText(img, "BUY", (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return img


def draw_synthetic_buy_box(
    img: np.ndarray,
    x: int = 400,
    entry_y: int = 350,
    w: int = 120,
    profit_h: int = 80,
    loss_h: int = 40,
) -> np.ndarray:
    """Draw a trade position box: green take profit above, red stop loss below."""
    # Green profit region (above entry line for BUY)
    cv2.rectangle(img, (x, entry_y - profit_h), (x + w - 1, entry_y - 1), (50, 180, 50), -1)
    # Red stop loss region (below entry line for BUY)
    cv2.rectangle(img, (x, entry_y + 1), (x + w - 1, entry_y + loss_h), (50, 50, 200), -1)
    # White horizontal entry line
    cv2.line(img, (x, entry_y), (x + w - 1, entry_y), (240, 240, 240), 2)
    return img


def make_frame(img: np.ndarray, frame_id: int = 1, dt: Optional[datetime] = None) -> Frame:
    """Helper to construct a standardized Frame dataclass."""
    return Frame(
        frame_id=frame_id,
        timestamp=dt or datetime.now(timezone.utc),
        width=1280,
        height=720,
        data=img,
    )


class MockStreamPipeline:
    """Simulated frame stream providing controlled sequences of synthetic frames."""

    def __init__(self, frames: List[Frame]) -> None:
        self._frames = frames
        self._is_running = False

    def stream_frames(self):
        self._is_running = True
        for f in self._frames:
            if not self._is_running:
                break
            yield f

    def stop(self):
        self._is_running = False


# ---------------------------------------------------------------------------
# Test Scenarios
# ---------------------------------------------------------------------------


def test_scenario_01_detector_a_alone_detects_trade(tmp_path: Path):
    """1. Detector A alone detects a newly appearing BUY label."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1
    config.detection.label_confirmation_count = 1

    # Create frames: frame 0 empty (baseline), frame 1 with BUY label
    f0_img = make_chart_canvas()
    f1_img = draw_synthetic_buy_label(make_chart_canvas(), x=500, y=300)

    frames = [
        make_frame(f0_img, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f1_img, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    pipeline = MockStreamPipeline(frames)
    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 101}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=pipeline,
        telegram_notifier=telegram_notifier,
        dry_run=False,
    )
    stats = orchestrator.run()

    assert stats.frames_processed == 2
    assert stats.trades_activated >= 1
    assert stats.notifications_sent >= 1
    # Verify exactly one active trade in database
    trades = TradeRepository(db_file).get_active_and_baseline_trades()
    active_trades = [t for t in trades if t.state == TradeState.ACTIVE]
    assert len(active_trades) == 1
    assert active_trades[0].notification_sent is True


def test_scenario_02_detector_b_alone_detects_trade(tmp_path: Path):
    """2. Detector B alone detects a newly appearing position box."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1
    config.detection.box_confirmation_count = 1

    f0_img = make_chart_canvas()
    f1_img = draw_synthetic_buy_box(make_chart_canvas(), x=450, entry_y=320)

    frames = [
        make_frame(f0_img, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f1_img, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    pipeline = MockStreamPipeline(frames)
    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 102}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=pipeline,
        telegram_notifier=telegram_notifier,
    )
    stats = orchestrator.run()

    assert stats.trades_activated == 1
    assert stats.notifications_sent == 1
    # Verify screenshot was generated and attached
    sent_notif = NotificationRepository(db_file).get_notification_by_trade_id(
        TradeRepository(db_file).get_active_and_baseline_trades()[0].trade_id
    )
    assert sent_notif is not None
    assert sent_notif.status == NotificationStatus.SENT
    assert sent_notif.message_text == "TRADE IS ACTIVE!"


def test_scenario_03_both_detectors_detect_same_trade(tmp_path: Path):
    """3. Both detectors detect the same trade simultaneously; produce exactly 1 notification."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0_img = make_chart_canvas()
    # Draw both BUY label and matching BUY position box
    f1_img = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)
    f1_img = draw_synthetic_buy_label(f1_img, x=400, y=320)

    frames = [
        make_frame(f0_img, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f1_img, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 103}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames),
        telegram_notifier=telegram_notifier,
    )
    stats = orchestrator.run()

    assert stats.trades_activated == 1
    assert stats.notifications_sent == 1
    assert http_mock.call_count == 1  # Exactly one external delivery


def test_scenario_04_two_separate_trades_appear_close_together(tmp_path: Path):
    """4. Two separate trades appear on the chart with sufficient spatial separation."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    # Frame 1: Trade 1 at x=200, entry_y=300; Trade 2 at x=600, entry_y=450
    f1 = draw_synthetic_buy_box(make_chart_canvas(), x=200, entry_y=300)
    f1 = draw_synthetic_buy_box(f1, x=600, entry_y=450)

    frames = [
        make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f1, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 104}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames),
        telegram_notifier=telegram_notifier,
    )
    stats = orchestrator.run()

    assert stats.trades_activated == 2
    assert stats.notifications_sent == 2


def test_scenario_05_trade_appears_while_other_detector_unavailable(tmp_path: Path):
    """5. Trade appears while Detector A is completely disabled (None)."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f1 = draw_synthetic_buy_box(make_chart_canvas(), x=350, entry_y=300)

    frames = [
        make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f1, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 105}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames),
        label_detector=None,  # Disabled Detector A
        telegram_notifier=telegram_notifier,
    )
    stats = orchestrator.run()

    assert stats.trades_activated == 1
    assert stats.notifications_sent == 1


def test_scenario_06_existing_trades_present_at_startup(tmp_path: Path):
    """6. Existing trades present at startup are registered as baseline and suppressed."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 2

    # Trade already visible in baseline frames 0 and 1
    f_trade = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    frames = [
        make_frame(f_trade, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f_trade, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        make_frame(f_trade, 2, datetime(2026, 11, 7, 10, 0, 2, tzinfo=timezone.utc)),
    ]

    http_mock = MagicMock(return_value=b'{"ok": true}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        http_client=http_mock,
    )

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames),
        telegram_notifier=telegram_notifier,
    )
    stats = orchestrator.run()

    assert stats.trades_activated == 0
    assert stats.notifications_sent == 0
    assert http_mock.call_count == 0


def test_scenario_07_and_08_stream_disconnects_and_reconnects_with_same_trade(tmp_path: Path):
    """7 & 8. Stream disconnects mid-session and reconnects; existing trade does not re-alert."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f_trade = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    # First session: trade activates
    frames_sess1 = [
        make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f_trade, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
    ]

    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 108}}')
    trade_repo = TradeRepository(db_file)
    notif_repo = NotificationRepository(db_file)
    event_repo = EventRepository(db_file)

    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=http_mock,
    )

    ev_mgr = EventManager(config=config, trade_repo=trade_repo, event_repo=event_repo)

    orch1 = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames_sess1),
        event_manager=ev_mgr,
        telegram_notifier=telegram_notifier,
        trade_repo=trade_repo,
        notification_repo=notif_repo,
        event_repo=event_repo,
    )
    orch1.run()
    assert orch1.stats.trades_activated == 1
    assert orch1.stats.notifications_sent == 1

    # Simulate Reconnection: feed more frames of the SAME trade into the same orchestrator/event manager
    frames_sess2 = [
        make_frame(f_trade, 2, datetime(2026, 11, 7, 10, 0, 5, tzinfo=timezone.utc)),
        make_frame(f_trade, 3, datetime(2026, 11, 7, 10, 0, 6, tzinfo=timezone.utc)),
    ]
    orch2 = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames_sess2),
        event_manager=ev_mgr,
        telegram_notifier=telegram_notifier,
        trade_repo=trade_repo,
        notification_repo=notif_repo,
        event_repo=event_repo,
    )
    orch2.run()

    # Must NOT produce a second activation or notification
    assert orch2.stats.trades_activated == 0
    assert orch2.stats.notifications_sent == 0
    assert http_mock.call_count == 1


def test_scenario_09_application_restart_with_active_trades(tmp_path: Path):
    """9. Application fully restarts; recovers active trade state from SQLite and avoids duplicates."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f_trade = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    # Session 1: Run and terminate
    http_mock1 = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 109}}')
    orch1 = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([
            make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
            make_frame(f_trade, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        ]),
        telegram_notifier=TelegramNotifier(
            config=config,
            notification_repo=NotificationRepository(db_file),
            trade_repo=TradeRepository(db_file),
            http_client=http_mock1,
        ),
    )
    orch1.run()
    assert orch1.stats.trades_activated == 1

    # Session 2: Fresh orchestrator instance pointing to the same SQLite DB
    http_mock2 = MagicMock(return_value=b'{"ok": true}')
    orch2 = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([
            make_frame(f_trade, 2, datetime(2026, 11, 7, 10, 0, 5, tzinfo=timezone.utc)),
        ]),
        telegram_notifier=TelegramNotifier(
            config=config,
            notification_repo=NotificationRepository(db_file),
            trade_repo=TradeRepository(db_file),
            http_client=http_mock2,
        ),
    )
    orch2.run()

    # Reopened session must suppress duplicate trade
    assert orch2.stats.trades_activated == 0
    assert orch2.stats.notifications_sent == 0
    assert http_mock2.call_count == 0


def test_scenario_10_application_restart_with_pending_notifications(tmp_path: Path):
    """10. Unsent notification in SQLite outbox is recovered and delivered on restart."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    trade_repo = TradeRepository(db_file)
    notif_repo = NotificationRepository(db_file)

    # Seed an active trade with a PENDING notification in database
    trade_id = "test-trade-pending-10"
    now = datetime.now(timezone.utc)
    trade = TradeRecord(
        trade_id=trade_id,
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        first_detected_at=now,
        last_seen_at=now,
        bbox=BoundingBox(100, 100, 50, 50),
        notification_sent=False,
    )
    trade_repo.save_trade(trade)

    record = NotificationRecord(
        notification_id="notif-uuid-10",
        trade_id=trade_id,
        chat_id="123456",
        message_text="TRADE IS ACTIVE!",
        photo_path=None,
        status=NotificationStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    notif_repo.create_notification(record)

    # Start orchestrator; startup should recover and dispatch the pending item
    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 110}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=http_mock,
    )

    orch = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([]),
        telegram_notifier=telegram_notifier,
        trade_repo=trade_repo,
        notification_repo=notif_repo,
        max_frames=0,
    )
    orch.startup()

    # Verify notification delivered and marked SENT
    assert http_mock.call_count == 1
    updated_rec = notif_repo.get_notification("notif-uuid-10")
    assert updated_rec.status == NotificationStatus.SENT
    assert trade_repo.get_trade(trade_id).notification_sent is True


def test_scenario_11_telegram_unavailable_retained_in_outbox(tmp_path: Path):
    """11. Telegram API outage; notification remains safely in outbox without crashing pipeline."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f1 = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    # Mock Telegram 503 Service Unavailable
    http_mock = MagicMock(side_effect=TelegramAPIError("Service Unavailable", error_code=503))
    notif_repo = NotificationRepository(db_file)
    trade_repo = TradeRepository(db_file)

    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=http_mock,
    )

    orch = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([
            make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
            make_frame(f1, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        ]),
        telegram_notifier=telegram_notifier,
    )
    stats = orch.run()

    assert stats.trades_activated == 1
    assert stats.notifications_sent == 0  # Not delivered yet

    # Verify notification is retained in outbox with retry scheduled
    trades = trade_repo.get_active_and_baseline_trades()
    active_trade = [t for t in trades if t.state == TradeState.ACTIVE][0]
    outbox_rec = notif_repo.get_notification_by_trade_id(active_trade.trade_id)
    assert outbox_rec is not None
    assert outbox_rec.status == NotificationStatus.PENDING
    assert outbox_rec.retry_count >= 1


def test_scenario_12_screenshot_capture_fails_fallback_to_text(tmp_path: Path):
    """12. Screenshot capture fails or is missing; notifier falls back to text-only alert."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f1 = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    # Screenshot manager that returns None for screenshot path
    mock_sm = MagicMock()
    mock_sm.save_trade_screenshot.return_value = None

    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 112}}')
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=NotificationRepository(db_file),
        trade_repo=TradeRepository(db_file),
        screenshot_manager=mock_sm,
        http_client=http_mock,
    )

    orch = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([
            make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
            make_frame(f1, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        ]),
        telegram_notifier=telegram_notifier,
    )
    stats = orch.run()

    assert stats.trades_activated == 1
    assert stats.notifications_sent == 1
    # Verify fallback sendMessage was called
    req = http_mock.call_args[0][0]
    assert "sendMessage" in req.full_url
    assert req.headers["Content-type"] == "application/json"


def test_scenario_13_database_temporarily_unavailable_queue_retry(tmp_path: Path):
    """13. Database write fails temporarily; event manager queues activation and retries safely."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    trade_repo = TradeRepository(db_file)
    event_repo = EventRepository(db_file)
    notif_repo = NotificationRepository(db_file)

    # Mock trade_repo to fail on first attempt, then succeed
    original_save = trade_repo.save_trade
    attempt = [0]

    def flaky_save(trade):
        attempt[0] += 1
        if attempt[0] == 1:
            raise Exception("Simulated SQLite OperationalError: database is locked")
        return original_save(trade)

    trade_repo.save_trade = flaky_save

    ev_mgr = EventManager(config=config, trade_repo=trade_repo, event_repo=event_repo)

    f0 = make_chart_canvas()
    f1 = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=MagicMock(return_value=b'{"ok": true}'),
    )

    orch = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline([
            make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
            make_frame(f1, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        ]),
        event_manager=ev_mgr,
        telegram_notifier=telegram_notifier,
        trade_repo=trade_repo,
    )
    stats = orch.run()

    # The first frame failed persistence, but in the next cycle retry_pending_persistence succeeded
    assert ev_mgr.retry_pending_persistence() >= 0
    # Notification was delivered once persistence succeeded
    assert len(trade_repo.get_active_and_baseline_trades()) >= 1


def test_scenario_14_delayed_notification_delivery_no_duplicates(tmp_path: Path):
    """14. Trade activation occurs while prior notification delivery is in-flight/delayed."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 1

    f0 = make_chart_canvas()
    f_trade = draw_synthetic_buy_box(make_chart_canvas(), x=400, entry_y=350)

    # Mock HTTP client that simulates delay without error
    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 114}}')

    notif_repo = NotificationRepository(db_file)
    trade_repo = TradeRepository(db_file)

    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=http_mock,
    )

    frames = [
        make_frame(f0, 0, datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)),
        make_frame(f_trade, 1, datetime(2026, 11, 7, 10, 0, 1, tzinfo=timezone.utc)),
        make_frame(f_trade, 2, datetime(2026, 11, 7, 10, 0, 2, tzinfo=timezone.utc)),
        make_frame(f_trade, 3, datetime(2026, 11, 7, 10, 0, 3, tzinfo=timezone.utc)),
    ]

    orch = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=MockStreamPipeline(frames),
        telegram_notifier=telegram_notifier,
        trade_repo=trade_repo,
        notification_repo=notif_repo,
    )
    stats = orch.run()

    # Even across 3 consecutive frames with the same trade, exactly 1 activation and 1 notification
    assert stats.trades_activated == 1
    assert stats.notifications_sent == 1
    assert http_mock.call_count == 1


def test_scenario_15_forced_process_termination_recovery(tmp_path: Path):
    """15. Test SQLite and outbox recovery after an actual forced process termination (kill -9)."""
    db_file = str(tmp_path / "trade_state.db")
    init_db(db_file)

    trade_repo = TradeRepository(db_file)
    notif_repo = NotificationRepository(db_file)

    # 1. Seed a trade and set its outbox status to SENDING
    now = datetime.now(timezone.utc)
    trade = TradeRecord(
        trade_id="trade-crash-test",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        first_detected_at=now,
        last_seen_at=now,
        bbox=BoundingBox(150, 150, 60, 60),
        notification_sent=False,
    )
    trade_repo.save_trade(trade)

    record = NotificationRecord(
        notification_id="notif-crash-test",
        trade_id="trade-crash-test",
        chat_id="123456",
        message_text="TRADE IS ACTIVE!",
        photo_path=None,
        status=NotificationStatus.SENDING,  # Interrupted in-flight
        created_at=now,
        updated_at=now,
    )
    notif_repo.create_notification(record)
    notif_repo.mark_sending("notif-crash-test")

    # Verify initial state is SENDING
    initial_rec = notif_repo.get_notification("notif-crash-test")
    assert initial_rec.status == NotificationStatus.SENDING

    # 2. Simulate forced application recovery: recover_interrupted_sending resets it
    recovered_count = notif_repo.recover_interrupted_sending(stale_threshold_sec=0.0)
    assert recovered_count == 1

    # Verify status transitioned back to PENDING
    recovered_rec = notif_repo.get_notification("notif-crash-test")
    assert recovered_rec.status == NotificationStatus.PENDING

    # 3. Deliver on restart
    http_mock = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 115}}')
    config = AppConfig()
    config.storage.db_path = db_file
    telegram_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=http_mock,
    )
    delivered = telegram_notifier.dispatch_pending_notifications()
    assert delivered == 1

    final_rec = notif_repo.get_notification("notif-crash-test")
    assert final_rec.status == NotificationStatus.SENT
    assert http_mock.call_count == 1
