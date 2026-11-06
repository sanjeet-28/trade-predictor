"""Centralized stream ingestion pipeline.

Orchestrates URL resolution, FFmpeg raw frame streaming, health monitoring,
and resilient automatic reconnection into a single clean frame iterator.
"""

from __future__ import annotations

import time
from typing import Generator, Optional

from core.config import AppConfig
from core.logger import get_logger
from core.models import Frame
from stream.ffmpeg_client import FFmpegFrameReader, FFmpegProcessError
from stream.health_monitor import StreamHealthMonitor
from stream.resolver import StreamOfflineError, StreamResolver, StreamResolverError, StreamUnavailableError

logger = get_logger("stream.pipeline")


class StreamPipeline:
    """Manages continuous livestream frame ingestion with automatic reconnect."""

    def __init__(
        self,
        config: AppConfig,
        stream_resolver: Optional[StreamResolver] = None,
        health_monitor: Optional[StreamHealthMonitor] = None,
        local_video_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.local_video_path = local_video_path

        # Subsystems
        self.resolver = stream_resolver or StreamResolver(
            primary_url=config.stream.primary_url,
            channel_url=config.stream.channel_url,
            preferred_height=int(getattr(config.stream, "frame_height", 720)),
            initial_backoff_sec=config.stream.reconnect_backoff_initial_sec,
        )

        self.health_monitor = health_monitor or StreamHealthMonitor(
            expected_width=int(getattr(config.stream, "frame_width", 1280)),
            expected_height=int(getattr(config.stream, "frame_height", 720)),
            freeze_duration_threshold_sec=config.stream.freeze_threshold_sec,
        )

        self._active_reader: Optional[FFmpegFrameReader] = None
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        """Return True if pipeline is actively consuming frames."""
        return self._is_running

    def stream_frames(self) -> Generator[Frame, None, None]:
        """Continuously yield captured video frames with automatic recovery."""
        self._is_running = True
        backoff = self.config.stream.reconnect_backoff_initial_sec
        max_backoff = self.config.stream.reconnect_backoff_max_sec

        logger.info(
            "Starting stream pipeline (mode=%s)",
            "LOCAL_FILE" if self.local_video_path else "YOUTUBE_LIVESTREAM",
        )

        while self._is_running:
            # 1. Determine Source
            if self.local_video_path:
                source_url = self.local_video_path
                is_local = True
            else:
                try:
                    resolution = self.resolver.resolve(force_refresh=False)
                    source_url = resolution.stream_url
                    is_local = False
                except Exception as e:
                    logger.warning("Stream resolution failed: %s (%s). Retrying in %.1fs...", type(e).__name__, e, backoff)
                    self.health_monitor.record_process_failure(str(e))
                    if not self._sleep_with_cancellation(backoff):
                        break
                    backoff = min(backoff * 2.0, max_backoff)
                    continue

            # 2. Launch FFmpeg Frame Reader
            target_w = int(getattr(self.config.stream, "frame_width", 1280))
            target_h = int(getattr(self.config.stream, "frame_height", 720))
            fps = float(self.config.stream.frame_sample_rate_fps)

            self._active_reader = FFmpegFrameReader(
                source=source_url,
                width=target_w,
                height=target_h,
                fps=fps,
                is_local_file=is_local,
            )

            try:
                self._active_reader.start()
                backoff = self.config.stream.reconnect_backoff_initial_sec  # Reset backoff on successful launch

                for frame in self._active_reader.frames():
                    if not self._is_running:
                        break

                    # Health check on each incoming frame
                    self.health_monitor.inspect_frame(frame)

                    # Hand off frame to consumer
                    yield frame

            except FFmpegProcessError as e:
                logger.error("FFmpeg frame reader encountered process error: %s", e)
                self.health_monitor.record_process_failure(str(e))
            except Exception as e:
                logger.error("Unexpected error in frame reader: %s", e)
                self.health_monitor.record_process_failure(str(e))
            finally:
                if self._active_reader:
                    self._active_reader.stop()
                    self._active_reader = None

            # If playing a local test video file and reached EOF, terminate cleanly
            if self.local_video_path and self._is_running:
                logger.info("Local video playback completed.")
                break

            # If live stream disconnected or failed, force URL refresh and backoff
            if self._is_running:
                logger.warning("Stream disconnected or frame reader stopped. Reconnecting in %.1fs...", backoff)
                # Force refresh stream resolution on next reconnect attempt
                try:
                    self.resolver.resolve(force_refresh=True)
                except Exception:
                    pass

                if not self._sleep_with_cancellation(backoff):
                    break
                backoff = min(backoff * 2.0, max_backoff)

        logger.info("Stream pipeline has shut down.")

    def stop(self) -> None:
        """Signal the pipeline to stop and shut down active readers."""
        logger.info("Signaling stream pipeline to stop.")
        self._is_running = False
        if self._active_reader:
            self._active_reader.stop()

    def _sleep_with_cancellation(self, duration_sec: float) -> bool:
        """Sleep in small intervals so stop() cancels quickly."""
        interval = 0.25
        elapsed = 0.0
        while self._is_running and elapsed < duration_sec:
            time.sleep(min(interval, duration_sec - elapsed))
            elapsed += interval
        return self._is_running
