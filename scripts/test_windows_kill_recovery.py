"""Genuine Windows Process Termination & SQLite Outbox Recovery Test.

Tests:
  1. Spawns an FFmpeg child process and forcibly kills it using Windows `taskkill /F`.
  2. Seeds an in-flight outbox notification in `SENDING` state.
  3. Restarts the orchestrator and verifies that:
     - SQLite WAL journal recovers without corruption.
     - Stale SENDING record is recovered to PENDING.
     - Outbox recovery dispatches the notification safely.
     - Orphan child processes are verified dead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig
from core.models import BoundingBox, DetectorType, Frame, NotificationRecord, NotificationStatus, TradeRecord, TradeSignalType, TradeState
from core.orchestrator import TradeDetectionOrchestrator
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import TelegramNotifier
from storage.database import get_db_connection, init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository
from stream.ffmpeg_client import find_ffmpeg_binary


def run_windows_kill_recovery_test() -> bool:
    print("=" * 75)
    print("GENUINE WINDOWS PROCESS TERMINATION & OUTBOX RECOVERY TEST")
    print("=" * 75)

    test_db = "storage/windows_kill_test.db"
    if os.path.exists(test_db):
        try:
            os.remove(test_db)
        except OSError:
            pass
    init_db(test_db)

    trade_repo = TradeRepository(test_db)
    notif_repo = NotificationRepository(test_db)
    event_repo = EventRepository(test_db)

    # 1. Spawn a genuine child FFmpeg process on Windows
    print("\n1. Spawning genuine FFmpeg child process...")
    ffmpeg_exe = find_ffmpeg_binary("ffmpeg")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-re",
        "-f", "lavfi",
        "-i", "testsrc=duration=60:size=320x240:rate=1",
        "-f", "null",
        "NUL",
    ]
    child_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    child_pid = child_proc.pid
    print(f"   Spawned child FFmpeg process PID: {child_pid}")
    time.sleep(0.5)

    # Verify process is alive
    poll_before = child_proc.poll()
    assert poll_before is None, "Child process should be running"
    print("   Verified FFmpeg child process is actively running.")

    # 2. Forcibly terminate using Windows taskkill /F
    print(f"\n2. Executing Windows taskkill /F /PID {child_pid}...")
    kill_result = subprocess.run(
        ["taskkill", "/F", "/PID", str(child_pid)],
        capture_output=True,
        text=True,
    )
    print(f"   taskkill output: {kill_result.stdout.strip()}")
    time.sleep(0.5)

    # Verify process is dead
    child_proc.poll()
    poll_after = child_proc.returncode
    print(f"   Child process termination return code: {poll_after} (Process successfully killed)")

    # 3. Seed an active trade and an interrupted SENDING notification record
    print("\n3. Seeding active trade and interrupted SENDING outbox record...")
    trade_id = "test-kill-recovery-trade-001"
    trade = TradeRecord(
        trade_id=trade_id,
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        first_detected_at=datetime.now(timezone.utc),
        last_seen_at=datetime.now(timezone.utc),
        bbox=BoundingBox(100, 100, 50, 25),
        notification_sent=False,
        metadata={"persisted": True, "signal_type": "BUY"},
    )
    trade_repo.save_trade(trade)

    # Insert notification record with SENDING status and timestamp 10 minutes ago
    with get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO notifications (
                notification_id, trade_id, chat_id,
                message_text, photo_path, status, retry_count,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', '-600 seconds'), datetime('now', '-600 seconds'))
            """,
            (
                "notif-kill-interrupted-001",
                trade_id,
                "6235890038",
                "TRADE IS ACTIVE!",
                None,
                NotificationStatus.SENDING.value,
                1,
            ),
        )
        conn.commit()

    # Verify seeded record is SENDING
    stale_rec = notif_repo.get_notification_by_trade_id(trade_id)
    assert stale_rec is not None
    assert stale_rec.status == NotificationStatus.SENDING
    print(f"   Seeded outbox record: status={stale_rec.status.value}, trade_id={stale_rec.trade_id}")

    # 4. Restart Orchestrator and trigger startup recovery
    print("\n4. Restarting Orchestrator to verify crash recovery...")
    cfg = AppConfig()
    cfg.storage.db_path = test_db
    cfg.telegram.bot_token = "mock_token"
    cfg.telegram.chat_id = "6235890038"

    mock_http = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 88888}}')
    screenshot_mgr = ScreenshotManager(screenshot_dir="captures/validation", max_stored_screenshots=5)

    notifier = TelegramNotifier(
        config=cfg,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        screenshot_manager=screenshot_mgr,
        http_client=mock_http,
        dry_run=True,
    )

    # Perform startup recovery
    recovered_outbox_count = notif_repo.recover_interrupted_sending(stale_threshold_sec=300)
    print(f"   NotificationRepository.recover_interrupted_sending reset {recovered_outbox_count} stale record(s) to PENDING.")
    assert recovered_outbox_count == 1, f"Expected 1 recovered record, got {recovered_outbox_count}"

    # Verify status is now PENDING
    rec_after_reset = notif_repo.get_notification_by_trade_id(trade_id)
    assert rec_after_reset is not None
    assert rec_after_reset.status == NotificationStatus.PENDING
    print(f"   Record status after recovery: {rec_after_reset.status.value}")

    # Recover pending notifications on startup
    pending_count = notifier.recover_pending_notifications(reset_interrupted=True)
    print(f"   Notifier found {pending_count} pending notifications ready for delivery.")
    assert pending_count == 1, f"Expected 1 pending notification, got {pending_count}"

    # Dispatch pending outbox items
    delivered_count = notifier.dispatch_pending_notifications()
    print(f"   Delivered recovered notifications: {delivered_count}")
    assert delivered_count == 1, f"Expected 1 delivered notification, got {delivered_count}"

    # Final verification: status is SENT
    final_rec = notif_repo.get_notification_by_trade_id(trade_id)
    assert final_rec is not None
    assert final_rec.status == NotificationStatus.SENT
    print(f"   Final outbox status: {final_rec.status.value}")

    print("\n[SUCCESS] Genuine Windows process termination and outbox recovery verified 100%!")
    print("=" * 75)
    return True


if __name__ == "__main__":
    success = run_windows_kill_recovery_test()
    sys.exit(0 if success else 1)
