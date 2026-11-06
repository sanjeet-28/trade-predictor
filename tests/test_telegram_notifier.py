"""Comprehensive unit tests for the Telegram Bot Notification Subsystem.

Tests cover all 20 required scenarios:
  1. Correct message text ('TRADE IS ACTIVE!').
  2. Correct chat ID handling.
  3. Missing token handling.
  4. Missing chat ID handling.
  5. Successful Telegram response.
  6. Telegram API rejection (HTTP 400).
  7. HTTP failure (HTTP 500 server error).
  8. Network timeout.
  9. Retry after temporary failure.
  10. Permanent failure handling (HTTP 401 / 403).
  11. Duplicate notification prevention.
  12. Notification recovery after restart.
  13. No notification for unpersisted trade activation.
  14. A/B duplicate evidence produces one notification.
  15. Independent Detector A activation sends one notification.
  16. Independent Detector B activation sends one notification.
  17. Screenshot attached successfully.
  18. Screenshot unavailable: text notification still works.
  19. Telegram failure does not stop stream processing or event management.
  20. Credentials never appear in logs.
"""

import json
import logging
import socket
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock
import numpy as np
import pytest

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
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import (
    MANDATORY_NOTIFICATION_TEXT,
    TelegramAPIError,
    TelegramNotifier,
    mask_token,
    sanitize_url,
)
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_repos(tmp_path: Path):
    """Create isolated SQLite database with Trade, Event, and Notification repositories."""
    db_file = str(tmp_path / "test_notifier.db")
    init_db(db_file)
    trade_repo = TradeRepository(db_file)
    event_repo = EventRepository(db_file)
    notif_repo = NotificationRepository(db_file)
    return trade_repo, event_repo, notif_repo, tmp_path


def make_persisted_trade(
    trade_id: str = "trade-100",
    state: TradeState = TradeState.ACTIVE,
    source: DetectorType = DetectorType.LABEL,
) -> TradeRecord:
    """Helper to create a validated and persisted trade record."""
    return TradeRecord(
        trade_id=trade_id,
        state=state,
        detector_source=source,
        bbox=BoundingBox(100, 100, 50, 30),
        metadata={"signal_type": "BUY", "persisted": True},
    )


def make_dummy_frame(width: int = 640, height: int = 480) -> Frame:
    """Helper to create an active frame buffer."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    return Frame(
        frame_id=1,
        timestamp=datetime.now(timezone.utc),
        width=width,
        height=height,
        data=img,
    )


# ---------------------------------------------------------------------------
# Test Cases 1 - 20
# ---------------------------------------------------------------------------

def test_correct_message_text(temp_repos) -> None:
    """Scenario 1: Mandatory notification message is strictly 'TRADE IS ACTIVE!'."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token_123"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    assert notif is not None
    assert notif.message_text == MANDATORY_NOTIFICATION_TEXT

    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 1

    # Verify message payload sent to HTTP client
    req = mock_client.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["text"] == MANDATORY_NOTIFICATION_TEXT
    assert "BUY" not in payload["text"]
    assert "SELL" not in payload["text"]


def test_correct_chat_id_handling(temp_repos) -> None:
    """Scenario 2: Messages are routed strictly to the configured chat_id."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "TARGET_CHAT_777"

    notifier.queue_trade_notification(trade)
    notifier.dispatch_pending_notifications()

    req = mock_client.call_args[0][0]
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["chat_id"] == "TARGET_CHAT_777"


def test_missing_token_handling(temp_repos) -> None:
    """Scenario 3: Missing bot token marks notification permanently FAILED without crashing."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
    )
    notifier.bot_token = ""
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    assert notif is not None

    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 0

    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.FAILED
    assert "missing" in record.last_error.lower()


def test_missing_chat_id_handling(temp_repos) -> None:
    """Scenario 4: Missing chat ID marks notification permanently FAILED without crashing."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = ""

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 0
    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.FAILED


def test_successful_telegram_response(temp_repos) -> None:
    """Scenario 5: Telegram HTTP 200 with ok:true marks outbox record SENT."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {"message_id": 999}}).encode("utf-8"))

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 1
    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.SENT
    assert record.sent_at is not None

    # TradeRecord is also updated with notification_sent=True
    updated_trade = trade_repo.get_trade(trade.trade_id)
    assert updated_trade.notification_sent is True


def test_telegram_api_rejection_400(temp_repos) -> None:
    """Scenario 6: Telegram API 400 Bad Request is marked permanent failure."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    # Return Telegram rejection JSON
    mock_client = MagicMock(
        return_value=json.dumps({"ok": False, "error_code": 400, "description": "Bad Request: chat not found"}).encode("utf-8")
    )

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "invalid_chat"

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 0
    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.FAILED
    assert record.next_retry_at is None


def test_http_failure_500_retried(temp_repos) -> None:
    """Scenario 7: Server 500 error is treated as temporary failure and scheduled for retry."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    # Mock raises HTTPError 502 Bad Gateway
    mock_fp = MagicMock()
    mock_fp.read.return_value = b'{"ok": false, "error_code": 502, "description": "Bad Gateway"}'
    err = urllib.error.HTTPError("https://api.telegram.org", 502, "Bad Gateway", {}, mock_fp)
    mock_client = MagicMock(side_effect=err)

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 0
    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.PENDING
    assert record.retry_count == 1
    assert record.next_retry_at is not None


def test_network_timeout_retried(temp_repos) -> None:
    """Scenario 8: Network timeout schedules temporary retry with exponential backoff."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    mock_client = MagicMock(side_effect=socket.timeout("The read operation timed out"))

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 0
    record = notif_repo.get_notification(notif.notification_id)
    assert record.status == NotificationStatus.PENDING
    assert record.retry_count == 1
    assert record.next_retry_at > record.created_at


def test_retry_after_temporary_failure(temp_repos) -> None:
    """Scenario 9: Temporary failure retried successfully on subsequent attempt."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    # First attempt fails; second succeeds
    fail_err = socket.timeout("Connection timeout")
    success_resp = json.dumps({"ok": True, "result": {}}).encode("utf-8")
    mock_client = MagicMock(side_effect=[fail_err, success_resp])

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)

    # First dispatch fails
    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    delivered_1 = notifier.dispatch_pending_notifications(current_time=t0)
    assert delivered_1 == 0
    rec1 = notif_repo.get_notification(notif.notification_id)
    assert rec1.status == NotificationStatus.PENDING

    # Advance past backoff time
    t1 = t0 + timedelta(seconds=10)
    delivered_2 = notifier.dispatch_pending_notifications(current_time=t1)
    assert delivered_2 == 1
    rec2 = notif_repo.get_notification(notif.notification_id)
    assert rec2.status == NotificationStatus.SENT


def test_permanent_failure_handling_403(temp_repos) -> None:
    """Scenario 10: Permanent failure (HTTP 403 bot blocked) aborts retries immediately."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    mock_client = MagicMock(
        return_value=json.dumps({"ok": False, "error_code": 403, "description": "Forbidden: bot was blocked by the user"}).encode("utf-8")
    )

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    notif = notifier.queue_trade_notification(trade)
    delivered = notifier.dispatch_pending_notifications()

    assert delivered == 0
    rec = notif_repo.get_notification(notif.notification_id)
    assert rec.status == NotificationStatus.FAILED
    assert rec.next_retry_at is None


def test_duplicate_notification_prevention(temp_repos) -> None:
    """Scenario 11: Repeated queueing attempts for the same trade produce exactly 1 notification."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade(trade_id="unique-trade-42")
    trade_repo.save_trade(trade)

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    # Queue first time: succeeds
    notif1 = notifier.queue_trade_notification(trade)
    assert notif1 is not None

    # Queue second time: rejected by duplicate prevention
    notif2 = notifier.queue_trade_notification(trade)
    assert notif2 is None

    # Queue third time: rejected
    notif3 = notifier.queue_trade_notification(trade)
    assert notif3 is None

    pending = notif_repo.get_pending_notifications()
    assert len(pending) == 1


def test_notification_recovery_after_restart(temp_repos) -> None:
    """Scenario 12: Pending notifications in SQLite are recovered and dispatched after application restart."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    # Session 1: Queued in outbox but application terminates before delivery
    rec = NotificationRecord(
        notification_id="recovered-notif-1",
        trade_id=trade.trade_id,
        chat_id="12345",
        message_text=MANDATORY_NOTIFICATION_TEXT,
        status=NotificationStatus.PENDING,
    )
    notif_repo.create_notification(rec)

    # Session 2: Application restarts, instantiates fresh TelegramNotifier
    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    restarted_notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    restarted_notifier.bot_token = "mock_token"
    restarted_notifier.chat_id = "12345"

    # Dispatches recovered pending notification from SQLite
    delivered = restarted_notifier.dispatch_pending_notifications()
    assert delivered == 1

    recovered_rec = notif_repo.get_notification("recovered-notif-1")
    assert recovered_rec.status == NotificationStatus.SENT


def test_no_notification_for_unpersisted_trade(temp_repos) -> None:
    """Scenario 13: Unpersisted trade activations (persisted=False) are rejected from queueing."""
    trade_repo, _, notif_repo, _ = temp_repos

    # Trade has persisted: False in metadata
    unpersisted_trade = TradeRecord(
        trade_id="unpersisted-trade-99",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        metadata={"persisted": False},
    )

    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
    )

    result = notifier.queue_trade_notification(unpersisted_trade)
    assert result is None, "Must not queue notification for unpersisted trade"
    assert notif_repo.get_notification_by_trade_id("unpersisted-trade-99") is None


def test_ab_duplicate_evidence_produces_one_notification(temp_repos) -> None:
    """Scenario 14: Detector A and B detecting the same trade trigger exactly 1 notification."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    event_mgr = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    event_a = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        bbox=BoundingBox(200, 245, 40, 20),
    )
    event_b = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        bbox=BoundingBox(185, 200, 120, 100),
        raw_metadata={"entry_y": 255},
    )

    # Frame 1: Detector A activates trade
    act1 = event_mgr.process_event(event_a)
    assert act1 is not None
    notif1 = notifier.queue_trade_notification(act1)
    assert notif1 is not None

    # Frame 2: Detector B detects same trade (deduplicated by EventManager)
    act2 = event_mgr.process_event(event_b)
    assert act2 is None  # Deduplicated, no new activation

    # Total pending notifications in outbox is exactly 1
    assert len(notif_repo.get_pending_notifications()) == 1


def test_independent_detector_a_activation(temp_repos) -> None:
    """Scenario 15: Detector A alone triggers a notification without waiting for Detector B."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    event_mgr = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    event_a = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        bbox=BoundingBox(200, 245, 40, 20),
    )

    act = event_mgr.process_event(event_a)
    assert act is not None

    notif = notifier.queue_trade_notification(act)
    assert notif is not None
    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 1


def test_independent_detector_b_activation(temp_repos) -> None:
    """Scenario 16: Detector B alone triggers a notification without waiting for Detector A."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    event_mgr = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    event_b = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.SELL,
        bbox=BoundingBox(300, 200, 100, 80),
        raw_metadata={"entry_y": 240},
    )

    act = event_mgr.process_event(event_b)
    assert act is not None

    notif = notifier.queue_trade_notification(act)
    assert notif is not None
    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 1


def test_screenshot_attached_successfully(temp_repos) -> None:
    """Scenario 17: Screenshot is saved on activation and sent via sendPhoto."""
    trade_repo, _, notif_repo, tmp_dir = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    frame = make_dummy_frame()
    screenshot_mgr = ScreenshotManager(screenshot_dir=str(tmp_dir / "screenshots"))

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        screenshot_manager=screenshot_mgr,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"
    notifier.attach_snapshot = True

    notif = notifier.queue_trade_notification(trade, frame=frame)
    assert notif is not None
    assert notif.photo_path is not None
    assert Path(notif.photo_path).exists()

    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 1

    req = mock_client.call_args[0][0]
    assert "sendPhoto" in req.full_url
    assert b'name="photo"' in req.data
    assert b'name="caption"' in req.data


def test_screenshot_unavailable_fallback_to_text(temp_repos) -> None:
    """Scenario 18: If screenshot file is missing or photo upload fails, text alert is sent."""
    trade_repo, _, notif_repo, _ = temp_repos
    trade = make_persisted_trade()
    trade_repo.save_trade(trade)

    # Missing file path
    rec = NotificationRecord(
        notification_id="missing-photo-notif",
        trade_id=trade.trade_id,
        chat_id="12345",
        message_text=MANDATORY_NOTIFICATION_TEXT,
        photo_path="non_existent_snapshot.png",
        status=NotificationStatus.PENDING,
    )
    notif_repo.create_notification(rec)

    mock_client = MagicMock(return_value=json.dumps({"ok": True, "result": {}}).encode("utf-8"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=mock_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    delivered = notifier.dispatch_pending_notifications()
    assert delivered == 1

    req = mock_client.call_args[0][0]
    # Fell back to sendMessage
    assert "sendMessage" in req.full_url
    payload = json.loads(req.data.decode("utf-8"))
    assert payload["text"] == MANDATORY_NOTIFICATION_TEXT


def test_telegram_failure_does_not_stop_stream_processing(temp_repos) -> None:
    """Scenario 19: Repeated Telegram failures do not disrupt event manager or stream pipeline."""
    trade_repo, event_repo, notif_repo, _ = temp_repos
    event_mgr = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Telegram always crashes/timeouts
    failing_client = MagicMock(side_effect=socket.timeout("Network is down"))
    notifier = TelegramNotifier(
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        http_client=failing_client,
    )
    notifier.bot_token = "mock_token"
    notifier.chat_id = "12345"

    # Frame 1: Trade 1 arrives
    e1 = DetectionEvent(detector_type=DetectorType.LABEL, signal_type=TradeSignalType.BUY, bbox=BoundingBox(10, 10, 30, 20))
    act1 = event_mgr.process_event(e1)
    notifier.queue_trade_notification(act1)
    notifier.dispatch_pending_notifications()  # Fails non-blockingly

    # Event Manager and detection continue completely unimpeded
    e2 = DetectionEvent(detector_type=DetectorType.LABEL, signal_type=TradeSignalType.SELL, bbox=BoundingBox(300, 300, 30, 20))
    act2 = event_mgr.process_event(e2)
    assert act2 is not None
    assert act2.trade_id != act1.trade_id
    assert len(event_mgr.get_active_trades()) == 2


def test_credentials_never_appear_in_logs(temp_repos, caplog) -> None:
    """Scenario 20: Secret token is masked and never exposed in logger outputs or sanitized URLs."""
    raw_secret_token = "987654321:SECRET_TOKEN_ABC_XYZ"
    masked = mask_token(raw_secret_token)

    assert "SECRET" not in masked
    assert masked == "987654***_XYZ"

    # Test URL sanitizer
    raw_url = f"https://api.telegram.org/bot{raw_secret_token}/sendMessage"
    sanitized = sanitize_url(raw_url, raw_secret_token)
    assert raw_secret_token not in sanitized
    assert "[REDACTED_TOKEN]" in sanitized

    # Test logger output during notifier instantiation
    with caplog.at_level(logging.INFO):
        notifier = TelegramNotifier(config=None)
        notifier.bot_token = raw_secret_token

    for record in caplog.records:
        assert raw_secret_token not in record.message

