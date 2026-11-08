"""Unit and integration tests for Phase 9: Google Colab Free Integration.

Verifies:
  1. Notebook JSON and syntax validity of all 12 cells.
  2. Dependency compatibility in requirements-linux.txt.
  3. Local SQLite to Drive backup sync and PRAGMA integrity verification.
  4. DRY_RUN=True safety enforcement with zero Telegram HTTP requests.
  5. Live notification safety gate and credential requirement behavior.
  6. Duration-based shutdown in TradeDetectionOrchestrator (max_duration_sec).
  7. Clean FFmpeg process termination on duration or interrupt shutdown.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.config import AppConfig, TelegramConfig
from core.models import BoundingBox, DetectorType, Frame, TradeRecord, TradeSignalType, TradeState
from core.orchestrator import TradeDetectionOrchestrator
from notifications.telegram_notifier import TelegramNotifier
from scripts.backup_db import backup_database, restore_database, sync_backup_to_drive
from storage.database import get_db_connection, init_db
from storage.repository import NotificationRepository, TradeRepository


def test_notebook_json_and_syntax_validity():
    """Verify Forex_Trade_Detector.ipynb conforms to valid nbformat v4 and all code cells compile."""
    nb_path = Path("notebooks/Forex_Trade_Detector.ipynb")
    assert nb_path.exists(), "Forex_Trade_Detector.ipynb was not found in notebooks/"

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert nb["nbformat"] == 4
    assert "cells" in nb
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) == 13, f"Expected 13 code cells, found {len(code_cells)}"

    # Validate that every code cell compiles under Python AST
    for idx, cell in enumerate(code_cells, start=1):
        src = "".join(cell["source"])
        clean_lines = []
        for line in src.splitlines():
            # Filter ipython line magics for standard compilation
            if line.strip().startswith("!") or line.strip().startswith("%"):
                clean_lines.append("# " + line)
            else:
                clean_lines.append(line)
        clean_src = "\n".join(clean_lines)
        code_obj = compile(clean_src, f"<cell_{idx}>", "exec")
        assert code_obj is not None, f"Cell {idx} failed compilation"


def test_requirements_linux_dependencies():
    """Verify requirements-linux.txt specifies required packages."""
    req_path = Path("requirements-linux.txt")
    assert req_path.exists()
    content = req_path.read_text(encoding="utf-8")
    for pkg in ["opencv-python-headless", "rapidocr_onnxruntime", "rapidocr", "onnxruntime", "yt-dlp", "numpy", "PyYAML"]:
        assert pkg in content, f"Missing {pkg} in requirements-linux.txt"


def test_local_to_drive_backup_and_integrity_verification():
    """Verify local SQLite backup creation, sync to Drive directory, and PRAGMA integrity check."""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_db = os.path.join(tmpdir, "trade_state.db")
        local_bak_dir = os.path.join(tmpdir, "local_backups")
        drive_bak_dir = os.path.join(tmpdir, "drive_backups")

        init_db(src_db)
        with get_db_connection(src_db) as conn:
            conn.execute(
                "INSERT INTO trades (trade_id, state, detector_source, first_detected_at, last_seen_at, bbox_x, bbox_y, bbox_width, bbox_height) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("trade-colab-1", "ACTIVE", "LABEL_DETECTOR", "2026-11-07T08:00:00Z", "2026-11-07T08:00:00Z", 10, 20, 30, 40),
            )
            conn.commit()

        # Step 1: Create local backup
        local_bak = backup_database(src_db, local_bak_dir)
        assert os.path.exists(local_bak)

        # Step 2: Sync to Google Drive directory and verify integrity
        drive_bak = sync_backup_to_drive(local_bak, drive_bak_dir)
        assert os.path.exists(drive_bak)
        assert os.path.basename(local_bak) == os.path.basename(drive_bak)

        # Step 3: Verify restored state from Drive backup
        target_restore = os.path.join(tmpdir, "restored.db")
        restore_database(drive_bak, target_restore)
        conn = sqlite3.connect(target_restore)
        try:
            row = conn.execute("SELECT trade_id, state FROM trades").fetchone()
            assert row == ("trade-colab-1", "ACTIVE")
            pragma = conn.execute("PRAGMA integrity_check").fetchone()
            assert pragma[0] == "ok"
        finally:
            conn.close()


def test_dry_run_zero_telegram_requests_without_credentials():
    """Verify DRY_RUN=True prevents external network calls even when credentials are empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        init_db(db_path)
        n_repo = NotificationRepository(db_path)
        t_repo = TradeRepository(db_path)

        http_calls = []

        def mock_client(req, timeout):
            http_calls.append(req.full_url)
            return b'{"ok": true}'

        cfg = AppConfig()
        cfg.telegram.bot_token = ""  # Zero credentials
        cfg.telegram.chat_id = ""

        notifier = TelegramNotifier(
            config=cfg,
            notification_repo=n_repo,
            trade_repo=t_repo,
            http_client=mock_client,
            dry_run=True,
        )

        trade = TradeRecord(
            trade_id="trade-dryrun-colab",
            state=TradeState.ACTIVE,
            detector_source=DetectorType.LABEL,
            first_detected_at=datetime.now(timezone.utc),
            last_seen_at=datetime.now(timezone.utc),
            bbox=BoundingBox(10, 20, 30, 40),
            notification_sent=False,
            metadata={"persisted": True, "signal_type": "BUY"},
        )
        t_repo.save_trade(trade)

        rec = notifier.queue_trade_notification(trade)
        assert rec is not None
        delivered = notifier.dispatch_pending_notifications()
        assert delivered == 1
        assert len(http_calls) == 0, "No HTTP calls should be made in dry-run mode"


def test_duration_based_shutdown_clean_exit():
    """Verify TradeDetectionOrchestrator exits cleanly when max_duration_sec is reached."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "dur_test.db")
        cfg = AppConfig()
        cfg.storage.db_path = db_path
        init_db(db_path)

        # Mock stream pipeline producing endless empty frames
        mock_pipeline = MagicMock()

        def frame_generator():
            frame_id = 0
            while True:
                frame_id += 1
                yield Frame(
                    data=None,
                    timestamp=datetime.now(timezone.utc),
                    frame_id=frame_id,
                    width=1280,
                    height=720,
                )
                time.sleep(0.05)

        mock_pipeline.stream_frames.side_effect = frame_generator

        orchestrator = TradeDetectionOrchestrator(
            config=cfg,
            stream_pipeline=mock_pipeline,
            dry_run=True,
            max_duration_sec=0.25,  # 250 milliseconds target duration
        )

        t0 = time.time()
        stats = orchestrator.run()
        elapsed = time.time() - t0

        assert elapsed >= 0.25
        assert elapsed < 2.0, "Orchestrator did not shut down promptly after max_duration_sec"
        assert not orchestrator._is_running
        assert stats.frames_processed > 0


def test_ffmpeg_clean_stop_on_orchestrator_stop():
    """Verify orchestrator cleanly calls pipeline stop which terminates FFmpeg process."""
    mock_pipeline = MagicMock()
    cfg = AppConfig()

    orchestrator = TradeDetectionOrchestrator(
        config=cfg,
        stream_pipeline=mock_pipeline,
        dry_run=True,
    )

    orchestrator.stop()
    mock_pipeline.stop.assert_called_once()


def test_cell_8_cookie_authentication_cell():
    """Verify Cell 8 (Cookie Authentication) executes safely and sets ACTIVE_COOKIE_PATH appropriately."""
    nb_path = Path("notebooks/Forex_Trade_Detector.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cell_8 = code_cells[7]
    cell_8_src = "".join(cell_8["source"])

    assert "COOKIE_FILE_PATH" in cell_8_src
    assert "ACTIVE_COOKIE_PATH" in cell_8_src

    # Test 1: Unauthenticated execution when no cookie file is present
    scope = {}
    exec(cell_8_src, scope)
    assert scope["ACTIVE_COOKIE_PATH"] is None

    # Test 2: Authenticated execution when cookie file is present
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_cookie_path = Path(tmpdir) / "test_cookies.txt"
        mock_cookie_path.write_text("# Netscape HTTP Cookie File\n")

        # Temporarily patch COOKIE_FILE_PATH in source execution
        scope2 = {"COOKIE_FILE_PATH": mock_cookie_path}
        # Run cell with patched target
        custom_src = cell_8_src.replace(
            'COOKIE_FILE_PATH = Path("/content/youtube_cookies.txt")',
            f'COOKIE_FILE_PATH = Path(r"{mock_cookie_path}")',
        )
        exec(custom_src, scope2)
        assert scope2["ACTIVE_COOKIE_PATH"] == str(mock_cookie_path)


def test_cell_9_stream_resolver_execution():
    """Regression test: verify Cell 9 code in notebook instantiates StreamResolver and executes without TypeError."""
    nb_path = Path("notebooks/Forex_Trade_Detector.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # Cell 9 is the 9th code cell (index 8)
    cell_9 = code_cells[8]
    cell_9_src = "".join(cell_9["source"])

    assert "StreamResolver" in cell_9_src
    # Ensure invalid constructor usage config=stream_cfg is not present
    assert "StreamResolver(config=" not in cell_9_src
    assert "cookie_file=ACTIVE_COOKIE_PATH" in cell_9_src

    # Mock StreamResolver.resolve so test doesn't make live YouTube network calls
    with patch("stream.resolver.StreamResolver.resolve") as mock_resolve:
        from stream.resolver import StreamResolution
        mock_resolve.return_value = StreamResolution(
            stream_url="https://manifest.googlevideo.com/mock_m3u8",
            width=1280,
            height=720,
            format_id="232",
            resolved_at=datetime.now(timezone.utc),
            is_live=True,
        )

        global_scope = {"ACTIVE_COOKIE_PATH": None}
        # Execute cell source to verify clean execution
        exec(cell_9_src, global_scope)

        assert "stream_info" in global_scope
        assert global_scope["stream_info"].format_id == "232"


def test_cell_13_cookie_cleanup():
    """Verify Cell 13 contains cookie file cleanup logic."""
    nb_path = Path("notebooks/Forex_Trade_Detector.ipynb")
    assert nb_path.exists()
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    # Cell 13 is the 13th code cell (index 12)
    cell_13 = code_cells[12]
    cell_13_src = "".join(cell_13["source"])

    assert "youtube_cookies.txt" in cell_13_src
    assert "os.remove" in cell_13_src


