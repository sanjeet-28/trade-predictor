"""Unit tests for the integrated StreamPipeline."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from core.config import AppConfig, StreamConfig
from stream.ffmpeg_client import find_ffmpeg_binary
from stream.pipeline import StreamPipeline
from stream.resolver import StreamResolution


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a 2-second 320x240 MP4 test video using FFmpeg."""
    video_path = tmp_path / "pipeline_test.mp4"
    ffmpeg_exe = find_ffmpeg_binary("ffmpeg")

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f", "lavfi",
        "-i", "testsrc=duration=2:size=320x240:rate=1",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(video_path),
    ]

    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return video_path


def test_pipeline_local_video_mode(synthetic_video: Path) -> None:
    """Verify StreamPipeline ingests frames from a local video file to completion."""
    config = AppConfig(
        stream=StreamConfig(
            frame_width=320,
            frame_height=240,
            frame_sample_rate_fps=1.0,
            local_video_path=str(synthetic_video),
        )
    )

    pipeline = StreamPipeline(config=config, local_video_path=str(synthetic_video))
    collected_frames = []

    for frame in pipeline.stream_frames():
        collected_frames.append(frame)

    assert len(collected_frames) == 2
    assert collected_frames[0].width == 320
    assert collected_frames[0].height == 240
    assert pipeline.health_monitor.status.total_frames_received == 2
    assert pipeline.health_monitor.status.is_healthy is True


def test_pipeline_graceful_stop(synthetic_video: Path) -> None:
    """Verify calling stop() exits stream_frames() loop early."""
    config = AppConfig(
        stream=StreamConfig(
            frame_width=320,
            frame_height=240,
            frame_sample_rate_fps=1.0,
            local_video_path=str(synthetic_video),
        )
    )

    pipeline = StreamPipeline(config=config, local_video_path=str(synthetic_video))
    collected = []

    for frame in pipeline.stream_frames():
        collected.append(frame)
        pipeline.stop()  # Stop immediately after 1st frame

    assert len(collected) == 1
    assert pipeline.is_running is False


def test_pipeline_reconnect_on_resolver_failure() -> None:
    """Verify pipeline retries when resolver encounters transient failure then stops cleanly."""
    config = AppConfig(
        stream=StreamConfig(
            frame_width=320,
            frame_height=240,
            reconnect_backoff_initial_sec=0.01,
            reconnect_backoff_max_sec=0.05,
        )
    )

    mock_resolver = MagicMock()
    # Simulate a stream offline error, then pipeline stop is called
    mock_resolver.resolve.side_effect = RuntimeError("Temporary network failure")

    pipeline = StreamPipeline(config=config, stream_resolver=mock_resolver)

    # Stop after a fraction of a second
    def stop_later():
        pipeline.stop()

    import threading
    timer = threading.Timer(0.05, stop_later)
    timer.start()

    frames = list(pipeline.stream_frames())
    timer.join()

    assert len(frames) == 0
    assert mock_resolver.resolve.call_count >= 1
    assert pipeline.health_monitor.status.is_healthy is False

