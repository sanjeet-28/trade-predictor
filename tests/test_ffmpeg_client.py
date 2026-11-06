"""Unit tests for FFmpeg subprocess frame reader."""

import subprocess
from pathlib import Path
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

