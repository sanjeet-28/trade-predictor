"""Livestream health and frame integrity watchdog.

Monitors frame arrival timing, detects black frames, catches frozen video streams,
and flags unexpected resolution changes without mistaking normal chart inactivity
for an encoder freeze.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.logger import get_logger
from core.models import Frame
from storage.repository import HealthLogRepository

logger = get_logger("stream.health")


@dataclass
class StreamHealthStatus:
    """Current diagnostic metrics and health state of the video stream."""
    is_healthy: bool = True
    total_frames_received: int = 0
    consecutive_black_frames: int = 0
    consecutive_frozen_frames: int = 0
    last_frame_timestamp: Optional[datetime] = None
    last_anomaly_description: Optional[str] = None
    last_mse: float = 0.0


class StreamHealthMonitor:
    """Evaluates video frames for freeze, black screen, delays, and geometry issues."""

    def __init__(
        self,
        expected_width: int = 1280,
        expected_height: int = 720,
        black_frame_threshold: float = 2.0,
        freeze_mse_threshold: float = 0.001,
        freeze_duration_threshold_sec: float = 30.0,
        max_frame_delay_sec: float = 10.0,
        health_repo: Optional[HealthLogRepository] = None,
    ) -> None:
        self.expected_width = expected_width
        self.expected_height = expected_height
        self.black_frame_threshold = black_frame_threshold
        self.freeze_mse_threshold = freeze_mse_threshold
        self.freeze_duration_threshold_sec = freeze_duration_threshold_sec
        self.max_frame_delay_sec = max_frame_delay_sec
        self.health_repo = health_repo

        self.status = StreamHealthStatus()
        self._previous_frame_data: Optional[np.ndarray] = None
        self._freeze_start_time: Optional[datetime] = None

    def inspect_frame(self, frame: Frame) -> bool:
        """Inspect a newly arrived frame for health anomalies.

        Returns True if the frame is healthy, or False if a critical health anomaly is detected.
        """
        now = datetime.now(timezone.utc)
        self.status.total_frames_received += 1
        is_healthy = True

        # 1. Check Frame Delays
        if self.status.last_frame_timestamp is not None:
            delay = (now - self.status.last_frame_timestamp).total_seconds()
            if delay > self.max_frame_delay_sec:
                msg = f"Excessive frame arrival delay: {delay:.2f}s (max: {self.max_frame_delay_sec}s)"
                self._record_anomaly("WARNING", msg)
                is_healthy = False

        self.status.last_frame_timestamp = now

        # 2. Check Frame Dimensions
        if frame.width != self.expected_width or frame.height != self.expected_height:
            msg = f"Unexpected frame dimensions: {frame.width}x{frame.height} (expected: {self.expected_width}x{self.expected_height})"
            self._record_anomaly("ERROR", msg)
            return False

        if frame.data is None:
            self._record_anomaly("ERROR", "Received empty frame data buffer")
            return False

        # 3. Check for Black Frames
        mean_intensity = float(np.mean(frame.data))
        if mean_intensity < self.black_frame_threshold:
            self.status.consecutive_black_frames += 1
            if self.status.consecutive_black_frames >= 5:
                msg = f"Persistent black frame detected: mean intensity={mean_intensity:.2f}"
                self._record_anomaly("ERROR", msg)
                is_healthy = False
        else:
            self.status.consecutive_black_frames = 0

        # 4. Check for Frozen Frames vs Normal Chart Inactivity
        # Normal chart inactivity still exhibits minor encoder compression noise or subpixel changes (MSE > 0).
        # A true stream freeze or encoder deadlock exhibits absolute identity (MSE <= freeze_mse_threshold).
        if self._previous_frame_data is not None:
            mse = float(np.mean((frame.data.astype(np.float32) - self._previous_frame_data.astype(np.float32)) ** 2))
            self.status.last_mse = mse

            if mse <= self.freeze_mse_threshold:
                self.status.consecutive_frozen_frames += 1
                if self._freeze_start_time is None:
                    self._freeze_start_time = now
                else:
                    freeze_duration = (now - self._freeze_start_time).total_seconds()
                    if freeze_duration >= self.freeze_duration_threshold_sec:
                        msg = f"Stream freeze confirmed: identical frames for {freeze_duration:.1f}s (MSE={mse:.6f})"
                        self._record_anomaly("CRITICAL", msg)
                        is_healthy = False
            else:
                self.status.consecutive_frozen_frames = 0
                self._freeze_start_time = None

        self._previous_frame_data = frame.data.copy()
        self.status.is_healthy = is_healthy
        return is_healthy

    def record_process_failure(self, error_message: str) -> None:
        """Log and track external process failures such as FFmpeg crashes or disconnects."""
        self.status.is_healthy = False
        self._record_anomaly("ERROR", f"Stream process failure: {error_message}")

    def _record_anomaly(self, level: str, message: str) -> None:
        """Log anomaly and persist diagnostic record into database repository."""
        self.status.last_anomaly_description = message
        logger.warning("[%s] Stream health anomaly: %s", level, message)

        if self.health_repo is not None:
            try:
                self.health_repo.log_health(
                    level=level,
                    component="stream.health_monitor",
                    message=message,
                )
            except Exception as e:
                logger.debug("Could not record health anomaly in repository: %s", e)

