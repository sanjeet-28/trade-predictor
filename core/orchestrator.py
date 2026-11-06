"""System Orchestrator: Coordinates end-to-end processing across all subsystems.

Integrates:
  StreamPipeline -> Detector A (Label) / Detector B (Box) -> EventManager
  -> SQLite Database (Trades & Events) -> Persistent Notification Outbox
  -> ScreenshotManager -> TelegramNotifier

Supports live streaming, simulated replay, dry-run safety, and graceful shutdown.
Runs seamlessly across Windows 11 and Linux.
"""

from __future__ import annotations

import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.config import AppConfig, load_config
from core.event_manager import EventManager
from core.logger import get_logger
from core.models import Frame, TradeRecord
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import TelegramNotifier
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository
from stream.pipeline import StreamPipeline

logger = get_logger("core.orchestrator")


@dataclass
class OrchestratorStats:
    """Metrics and counters collected during orchestrator execution."""
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frames_processed: int = 0
    detector_a_events: int = 0
    detector_b_events: int = 0
    trades_activated: int = 0
    notifications_queued: int = 0
    notifications_sent: int = 0
    reconnect_events: int = 0


class TradeDetectionOrchestrator:
    """Unified application runner coordinating stream ingestion, detection, and notification."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        stream_pipeline: Optional[StreamPipeline] = None,
        label_detector: Optional[LabelDetector] = None,
        box_detector: Optional[BoxDetector] = None,
        event_manager: Optional[EventManager] = None,
        telegram_notifier: Optional[TelegramNotifier] = None,
        screenshot_manager: Optional[ScreenshotManager] = None,
        trade_repo: Optional[TradeRepository] = None,
        event_repo: Optional[EventRepository] = None,
        notification_repo: Optional[NotificationRepository] = None,
        local_video_path: Optional[str] = None,
        dry_run: bool = False,
        simulate_mode: bool = False,
        max_frames: Optional[int] = None,
    ) -> None:
        self.config = config or load_config()
        self.dry_run = dry_run
        self.simulate_mode = simulate_mode
        self.max_frames = max_frames
        self.local_video_path = local_video_path

        # 1. Database & Repositories
        db_path = getattr(self.config.storage, "db_path", "storage/trade_state.db") if hasattr(self.config, "storage") else "storage/trade_state.db"
        init_db(db_path)
        self.trade_repo = trade_repo or TradeRepository(db_path)
        self.event_repo = event_repo or EventRepository(db_path)
        self.notification_repo = notification_repo or NotificationRepository(db_path)

        # 2. Screenshot Manager
        self.screenshot_manager = screenshot_manager or ScreenshotManager(
            screenshot_dir=self.config.telegram.screenshot_dir,
            max_stored_screenshots=self.config.telegram.max_stored_screenshots,
        )

        # 3. Detectors
        self.label_detector = label_detector or LabelDetector(
            baseline_frames=self.config.detection.baseline_calibration_frames,
        )
        self.box_detector = box_detector or BoxDetector(
            baseline_frames=self.config.detection.baseline_calibration_frames,
        )

        # 4. Central Event Manager
        self.event_manager = event_manager or EventManager(
            config=self.config,
            trade_repo=self.trade_repo,
            event_repo=self.event_repo,
        )

        # 5. Telegram Notifier
        self.telegram_notifier = telegram_notifier or TelegramNotifier(
            config=self.config,
            notification_repo=self.notification_repo,
            trade_repo=self.trade_repo,
            screenshot_manager=self.screenshot_manager,
            dry_run=self.dry_run,
        )

        # 6. Stream Pipeline
        self.pipeline = stream_pipeline or StreamPipeline(
            config=self.config,
            local_video_path=self.local_video_path,
        )

        # Execution State & Metrics
        self.stats = OrchestratorStats()
        self._is_running: bool = False
        self._shutdown_requested: bool = False

        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Register graceful shutdown handlers for POSIX (Linux) and Windows."""
        def handle_signal(sig: int, frame: Any) -> None:
            logger.info("Received termination signal (%s). Initiating graceful shutdown...", sig)
            self.stop()

        try:
            signal.signal(signal.SIGINT, handle_signal)
        except (ValueError, AttributeError):
            pass

        if hasattr(signal, "SIGTERM"):
            try:
                signal.signal(signal.SIGTERM, handle_signal)
            except (ValueError, AttributeError):
                pass

    @property
    def is_running(self) -> bool:
        """Return True if orchestrator loop is actively processing frames."""
        return self._is_running

    def startup(self) -> None:
        """Perform pre-run health checks and recover stale outbox states."""
        logger.info(
            "Initializing TradeDetectionOrchestrator (dry_run=%s, simulate=%s, max_frames=%s)",
            self.dry_run,
            self.simulate_mode,
            self.max_frames,
        )

        # Recover any interrupted 'SENDING' or 'PENDING' notifications from previous crash
        if self.telegram_notifier:
            recovered = self.telegram_notifier.recover_pending_notifications(reset_interrupted=True)
            if recovered > 0:
                logger.info("Recovered %d pending notifications from previous session.", recovered)
                # Attempt to dispatch recovered outbox items
                self.telegram_notifier.dispatch_pending_notifications()

    def run(self) -> OrchestratorStats:
        """Run the main processing loop until stream ends, max_frames reached, or stopped."""
        self.startup()

        if self.max_frames is not None and self.max_frames <= 0:
            logger.info("max_frames <= 0 specified (%d); exiting without starting stream.", self.max_frames)
            self.stop()
            return self.stats

        self._is_running = True
        logger.info("Orchestrator processing loop started.")

        try:
            for frame in self.pipeline.stream_frames():
                if not self._is_running or self._shutdown_requested:
                    break

                self.process_single_frame(frame)

                if self.max_frames is not None and self.stats.frames_processed >= self.max_frames:
                    logger.info("Reached maximum target frames (%d). Exiting loop.", self.max_frames)
                    break

        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt caught in run loop. Stopping orchestrator...")
        except Exception as e:
            logger.error("Unhandled exception in orchestrator run loop: %s", e, exc_info=True)
        finally:
            self.stop()

        return self.stats

    def process_single_frame(self, frame: Frame) -> List[TradeRecord]:
        """Execute end-to-end processing pipeline for a single video frame.

        Returns:
            List of TradeRecords activated by this frame.
        """
        self.stats.frames_processed += 1

        # 1. Detector A: BUY / SELL Label Recognition
        events_a = []
        if self.label_detector:
            try:
                events_a = self.label_detector.process_frame(frame)
                self.stats.detector_a_events += len(events_a)
            except Exception as e:
                logger.error("Detector A error on frame %d: %s", self.stats.frames_processed, e)

        # 2. Detector B: Trade-Position Box Recognition
        events_b = []
        if self.box_detector:
            try:
                events_b = self.box_detector.process_frame(frame)
                self.stats.detector_b_events += len(events_b)
            except Exception as e:
                logger.error("Detector B error on frame %d: %s", self.stats.frames_processed, e)

        # 3. Central Event Manager Deduplication & Multi-Evidence Correlation
        all_events = events_a + events_b
        new_trades: List[TradeRecord] = []
        if self.event_manager:
            try:
                new_trades = self.event_manager.process_events(all_events, frame=frame)
                self.stats.trades_activated += len(new_trades)
            except Exception as e:
                logger.error("EventManager error on frame %d: %s", self.stats.frames_processed, e)

        # 4. Enqueue Notifications for Newly Confirmed Activations
        for trade in new_trades:
            if self.telegram_notifier:
                rec = self.telegram_notifier.queue_trade_notification(trade, frame=frame)
                if rec:
                    self.stats.notifications_queued += 1

        # 5. Outbox Dispatch & Persistence Retry Pass
        if self.telegram_notifier:
            delivered = self.telegram_notifier.dispatch_pending_notifications()
            self.stats.notifications_sent += delivered

        if self.event_manager:
            self.event_manager.retry_pending_persistence()

        return new_trades

    def stop(self) -> None:
        """Signal orchestrator and stream pipeline to shut down cleanly."""
        if not self._is_running and self._shutdown_requested:
            return

        self._shutdown_requested = True
        self._is_running = False
        logger.info("Stopping orchestrator and dependent subsystems...")

        if self.pipeline:
            self.pipeline.stop()

        elapsed = (datetime.now(timezone.utc) - self.stats.start_time).total_seconds()
        logger.info(
            "Orchestrator stopped cleanly. Summary: %d frames processed, %d A-events, %d B-events, "
            "%d trades activated, %d notifications sent in %.1fs.",
            self.stats.frames_processed,
            self.stats.detector_a_events,
            self.stats.detector_b_events,
            self.stats.trades_activated,
            self.stats.notifications_sent,
            elapsed,
        )
