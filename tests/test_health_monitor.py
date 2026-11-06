"""Unit tests for livestream health monitoring and watchdog logic."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np

from core.models import Frame
from storage.database import init_db
from storage.repository import HealthLogRepository
from stream.health_monitor import StreamHealthMonitor


def make_frame(
    frame_id: int,
    mean_val: float = 120.0,
    width: int = 1280,
    height: int = 720,
    noise_amplitude: float = 0.0,
) -> Frame:
    """Generate a synthetic test frame with controlled brightness and noise."""
    base = np.full((height, width, 3), mean_val, dtype=np.uint8)
    if noise_amplitude > 0:
        noise = np.random.randint(0, int(noise_amplitude) + 1, (height, width, 3), dtype=np.uint8)
        base = np.clip(base + noise, 0, 255).astype(np.uint8)

    return Frame(
        frame_id=frame_id,
        timestamp=datetime.now(timezone.utc),
        width=width,
        height=height,
        data=base,
    )


def test_health_monitor_healthy_stream() -> None:
    """Verify normal frames keep stream health state healthy."""
    monitor = StreamHealthMonitor(expected_width=1280, expected_height=720)

    for i in range(1, 5):
        frame = make_frame(frame_id=i, mean_val=100.0, noise_amplitude=2.0)
        assert monitor.inspect_frame(frame) is True

    assert monitor.status.is_healthy is True
    assert monitor.status.total_frames_received == 4
    assert monitor.status.consecutive_black_frames == 0
    assert monitor.status.consecutive_frozen_frames == 0


def test_health_monitor_black_frame_detection() -> None:
    """Verify persistent black frames trigger health failure."""
    monitor = StreamHealthMonitor(black_frame_threshold=2.0)

    # 4 black frames: consecutive counter increments, not yet failing
    for i in range(1, 5):
        black_frame = make_frame(frame_id=i, mean_val=0.5)
        monitor.inspect_frame(black_frame)
        assert monitor.status.consecutive_black_frames == i

    # 5th black frame triggers failure
    black_frame = make_frame(frame_id=5, mean_val=0.5)
    is_healthy = monitor.inspect_frame(black_frame)
    assert is_healthy is False
    assert monitor.status.consecutive_black_frames == 5
    assert "Persistent black frame detected" in (monitor.status.last_anomaly_description or "")


def test_health_monitor_dimension_mismatch() -> None:
    """Verify frames with unexpected dimensions fail inspection immediately."""
    monitor = StreamHealthMonitor(expected_width=1280, expected_height=720)
    bad_frame = make_frame(frame_id=1, width=640, height=480)

    assert monitor.inspect_frame(bad_frame) is False
    assert "Unexpected frame dimensions" in (monitor.status.last_anomaly_description or "")


def test_health_monitor_frozen_frame_vs_chart_inactivity() -> None:
    """Verify true freeze is detected but subtle chart inactivity does not trigger a false alarm."""
    monitor = StreamHealthMonitor(
        freeze_duration_threshold_sec=2.0,  # short threshold for testing
        freeze_mse_threshold=0.001,
    )

    # Scenario 1: Normal chart inactivity (encoder noise / clock tick produces MSE > 0.001)
    for i in range(1, 6):
        frame = make_frame(frame_id=i, mean_val=150.0, noise_amplitude=1.0)
        is_healthy = monitor.inspect_frame(frame)
        assert is_healthy is True
        assert monitor.status.consecutive_frozen_frames == 0

    # Scenario 2: Identical frames (true encoder freeze, MSE == 0.0)
    static_frame = make_frame(frame_id=10, mean_val=150.0, noise_amplitude=0.0)
    monitor.inspect_frame(static_frame)

    # Simulate arrival of identical frame past the duration threshold
    freeze_frame = make_frame(frame_id=11, mean_val=150.0, noise_amplitude=0.0)
    # Artificially set freeze start time back in past to simulate > 2s elapsed
    monitor._freeze_start_time = datetime.now(timezone.utc) - timedelta(seconds=3.0)

    is_healthy = monitor.inspect_frame(freeze_frame)
    assert is_healthy is False
    assert monitor.status.consecutive_frozen_frames >= 1
    assert "Stream freeze confirmed" in (monitor.status.last_anomaly_description or "")


def test_health_monitor_records_to_database(tmp_path: Path) -> None:
    """Verify health anomalies are written into HealthLogRepository."""
    db_path = str(tmp_path / "health_test.db")
    init_db(db_path)
    repo = HealthLogRepository(db_path)

    monitor = StreamHealthMonitor(
        expected_width=1280,
        expected_height=720,
        health_repo=repo,
    )

    bad_frame = make_frame(frame_id=1, width=800, height=600)
    monitor.inspect_frame(bad_frame)

    logs = repo.get_recent_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["level"] == "ERROR"
    assert "Unexpected frame dimensions" in logs[0]["message"]

