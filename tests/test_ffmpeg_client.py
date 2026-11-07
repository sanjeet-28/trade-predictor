"""Unit tests for FFmpeg subprocess frame reader."""

from datetime import datetime, timezone
from pathlib import Path
import subprocess
from unittest.mock import MagicMock
import numpy as np
import pytest

from core.models import Frame
from stream.ffmpeg_client import (
    FFmpegFrameReader,
    FFmpegProcessError,
    find_ffmpeg_binary,
)


@pytest.fixture
def synthetic_video(tmp_path: Path) -> Path:
    """Generate a 2-second 320x240 MP4 test video using FFmpeg."""
    video_path = tmp_path / "synthetic_test.mp4"
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

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, f"FFmpeg video creation failed: {result.stderr.decode()}"
    assert video_path.exists()
    return video_path


def test_find_ffmpeg_binary() -> None:
    """Verify find_ffmpeg_binary locates a working executable."""
    binary_path = find_ffmpeg_binary("ffmpeg")
    assert binary_path is not None
    assert Path(binary_path).exists()


def test_ffmpeg_frame_reader_dimensions_and_sampling(synthetic_video: Path) -> None:
    """Verify FFmpegFrameReader produces frames with exact requested dimensions and FPS."""
    reader = FFmpegFrameReader(
        source=str(synthetic_video),
        width=320,
        height=240,
        fps=1.0,
        pix_fmt="bgr24",
        is_local_file=True,
    )

    frames = list(reader.frames())

    # 2 seconds at 1 FPS should produce 2 frames
    assert len(frames) == 2

    for idx, frame in enumerate(frames, start=1):
        assert isinstance(frame, Frame)
        assert frame.frame_id == idx
        assert frame.width == 320
        assert frame.height == 240
        assert frame.data is not None
        assert frame.data.shape == (240, 320, 3)
        assert frame.data.dtype.name == "uint8"

    assert reader.is_running is False


def test_ffmpeg_frame_reader_graceful_stop(synthetic_video: Path) -> None:
    """Verify stop() terminates subprocess cleanly mid-stream."""
    reader = FFmpegFrameReader(
        source=str(synthetic_video),
        width=320,
        height=240,
        fps=1.0,
        is_local_file=True,
    )

    reader.start()
    assert reader.is_running is True

    # Read 1 frame and immediately stop
    frame1 = reader.read_frame()
    assert frame1 is not None

    reader.stop()
    assert reader.is_running is False
    assert reader._process is None


def test_ffmpeg_frame_reader_invalid_source(tmp_path: Path) -> None:
    """Verify non-existent video file handles EOF or raises FFmpegProcessError."""
    missing_file = tmp_path / "non_existent.mp4"
    reader = FFmpegFrameReader(
        source=str(missing_file),
        width=320,
        height=240,
        is_local_file=True,
    )

    with pytest.raises(FFmpegProcessError):
        reader.start()
        reader.read_frame()


def test_ffmpeg_frame_reader_decoupled_queue_drop_tracking() -> None:
    """Verify decoupled live stream reader discards stale frames when consumer is busy."""
    reader = FFmpegFrameReader(
        source="http://mock.live.stream/test.m3u8",
        width=320,
        height=240,
        is_local_file=False,
    )

    frame1 = Frame(1, datetime.now(timezone.utc), 320, 240, np.zeros((240, 320, 3), dtype=np.uint8))
    frame2 = Frame(2, datetime.now(timezone.utc), 320, 240, np.ones((240, 320, 3), dtype=np.uint8))
    frame3 = Frame(3, datetime.now(timezone.utc), 320, 240, np.full((240, 320, 3), 2, dtype=np.uint8))

    reader._is_running = True
    reader._process = MagicMock()

    # Simulate reader worker buffering frames without consumer reading
    with reader._frame_lock:
        reader._latest_frame = frame1
        reader.frames_received += 1
        reader._new_frame_event.set()

    # Second frame arrives before first was read -> drop count increments
    with reader._frame_lock:
        if reader._latest_frame is not None:
            reader.frames_dropped += 1
        reader._latest_frame = frame2
        reader.frames_received += 1
        reader._new_frame_event.set()

    # Third frame arrives -> drop count increments again
    with reader._frame_lock:
        if reader._latest_frame is not None:
            reader.frames_dropped += 1
        reader._latest_frame = frame3
        reader.frames_received += 1
        reader._new_frame_event.set()

    assert reader.frames_dropped == 2
    assert reader.frames_received == 3

    # Consumer reads -> should receive the freshest frame (frame3)
    delivered = reader.read_frame()
    assert delivered is not None
    assert delivered.frame_id == 3
    assert reader.frames_delivered == 1

