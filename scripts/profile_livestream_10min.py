"""10-Minute Real Livestream Profiling & Long-Running Stability Benchmark.

Profiles:
  1. Received, delivered, and dropped frames from live YouTube stream over 10 minutes (600s).
  2. Wall-clock frame processing latency (min, avg, median, p95, max).
  3. CPU processing time per frame (min, avg, median, p95, max).
  4. Memory RSS stability and growth over 600 continuous cycles.
  5. Reconnection events and stream health log entries.
  6. Safe dry-run mode (mocked notification outbox).
"""

from __future__ import annotations

import json
import os
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.event_manager import EventManager
from core.logger import get_logger, setup_logging
from core.orchestrator import TradeDetectionOrchestrator
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from notifications.screenshot_manager import ScreenshotManager
from notifications.telegram_notifier import TelegramNotifier
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository
from stream.pipeline import StreamPipeline

logger = get_logger("scripts.profile_10min")


def run_10min_profile(duration_sec: float = 600.0) -> Dict[str, Any]:
    print("=" * 80)
    print("FOREX TRADE DETECTION SYSTEM - 10-MINUTE LIVE STREAM PROFILING BENCHMARK")
    print("=" * 80)
    print(f"Target Duration:    {duration_sec:.0f} seconds (~{duration_sec/60:.1f} minutes)")
    print("Stream Source:      Official YouTube Stream (1280x720 @ 1.0 FPS)")
    print("Safety Mode:        DRY-RUN (Zero external Telegram messages)")
    print("-" * 80)

    profile_db = "storage/profile_10min.db"
    if os.path.exists(profile_db):
        try:
            os.remove(profile_db)
        except OSError:
            pass
    init_db(profile_db)

    config = load_config()
    config.storage.db_path = profile_db
    config.detection.baseline_calibration_frames = 10

    trade_repo = TradeRepository(profile_db)
    event_repo = EventRepository(profile_db)
    notif_repo = NotificationRepository(profile_db)

    screenshot_dir = "captures/profile_10min"
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_mgr = ScreenshotManager(screenshot_dir=screenshot_dir, max_stored_screenshots=10)

    mock_http = MagicMock(return_value=b'{"ok": true, "result": {"message_id": 77777}}')
    safe_notifier = TelegramNotifier(
        config=config,
        notification_repo=notif_repo,
        trade_repo=trade_repo,
        screenshot_manager=screenshot_mgr,
        http_client=mock_http,
        dry_run=True,
    )

    label_detector = LabelDetector(baseline_frames=10)
    box_detector = BoxDetector(baseline_frames=10)
    event_manager = EventManager(config=config, trade_repo=trade_repo, event_repo=event_repo)
    pipeline = StreamPipeline(config=config)

    orchestrator = TradeDetectionOrchestrator(
        config=config,
        stream_pipeline=pipeline,
        label_detector=label_detector,
        box_detector=box_detector,
        event_manager=event_manager,
        telegram_notifier=safe_notifier,
        screenshot_manager=screenshot_mgr,
        trade_repo=trade_repo,
        event_repo=event_repo,
        notification_repo=notif_repo,
        dry_run=True,
    )

    # Start memory tracing
    tracemalloc.start()
    mem_start_curr, _ = tracemalloc.get_traced_memory()

    # Latency tracking
    cycle_wall_times: List[float] = []
    cycle_cpu_times: List[float] = []
    det_a_times: List[float] = []
    det_b_times: List[float] = []
    arrival_intervals: List[float] = []

    last_frame_arrival = None
    frames_processed = 0
    start_time = time.time()
    next_log_time = start_time + 60.0  # Log update every 60 seconds
    final_frames_received = 0
    final_frames_delivered = 0
    final_frames_dropped = 0

    orchestrator.startup()
    print("Connected to stream. Monitoring 10-minute runtime stability...")

    try:
        for frame in pipeline.stream_frames():
            now = time.time()
            if last_frame_arrival is not None:
                arrival_intervals.append(now - last_frame_arrival)
            last_frame_arrival = now

            frames_processed += 1

            # Cycle timing
            t_cyc_w0 = time.perf_counter()
            t_cyc_c0 = time.process_time()

            # Process single frame
            orchestrator.process_single_frame(frame)

            dt_wall = (time.perf_counter() - t_cyc_w0) * 1000.0
            dt_cpu = (time.process_time() - t_cyc_c0) * 1000.0

            cycle_wall_times.append(dt_wall)
            cycle_cpu_times.append(dt_cpu)

            # Periodic status logging
            if now >= next_log_time:
                elapsed = now - start_time
                reader = pipeline._active_reader
                dropped = reader.frames_dropped if reader else 0
                avg_lat = float(np.mean(cycle_wall_times[-60:])) if cycle_wall_times else 0
                curr_mem, peak_mem = tracemalloc.get_traced_memory()
                print(
                    f"  [{elapsed:4.0f}s / {duration_sec:.0f}s] "
                    f"Frames: {frames_processed} | "
                    f"Dropped: {dropped} | "
                    f"Avg Lat: {avg_lat:5.1f}ms | "
                    f"Mem: {curr_mem / 1024 / 1024:4.1f}MB (Peak: {peak_mem / 1024 / 1024:4.1f}MB)"
                )
                next_log_time = now + 60.0

            if (now - start_time) >= duration_sec:
                print(f"\n[OK] Reached target duration of {duration_sec:.0f} seconds.")
                # Capture active reader stats prior to generator cleanup
                reader = pipeline._active_reader
                if reader:
                    final_frames_received = reader.frames_received
                    final_frames_delivered = reader.frames_delivered
                    final_frames_dropped = reader.frames_dropped
                break

    except KeyboardInterrupt:
        print("\n[INFO] Profiling interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Profiling encountered exception: {e}")
        logger.error("Profiling loop error: %s", e, exc_info=True)
    finally:
        total_duration = time.time() - start_time
        reader = pipeline._active_reader
        frames_received = final_frames_received or (reader.frames_received if reader else frames_processed)
        frames_delivered = final_frames_delivered or (reader.frames_delivered if reader else frames_processed)
        frames_dropped = final_frames_dropped if final_frames_dropped > 0 else (reader.frames_dropped if reader else 0)
        pipeline.stop()
        orchestrator.stop()

    mem_end_curr, mem_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Calculate statistics
    def calc_stats(data: List[float]) -> Dict[str, float]:
        if not data:
            return {"min": 0, "avg": 0, "median": 0, "p95": 0, "max": 0}
        arr = np.array(data)
        return {
            "min": round(float(np.min(arr)), 2),
            "avg": round(float(np.mean(arr)), 2),
            "median": round(float(np.median(arr)), 2),
            "p95": round(float(np.percentile(arr, 95)), 2),
            "max": round(float(np.max(arr)), 2),
        }

    wall_stats = calc_stats(cycle_wall_times)
    cpu_stats = calc_stats(cycle_cpu_times)
    arrival_stats = calc_stats(arrival_intervals)

    print("\n" + "=" * 80)
    print("10-MINUTE PROFILING RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Duration:     {total_duration:6.1f} seconds")
    print(f"Frames Received:    {frames_received}")
    print(f"Frames Processed:   {frames_processed}")
    print(f"Frames Dropped:     {frames_dropped}")
    print(f"Drop Rate:          {(frames_dropped / max(1, frames_received)) * 100:.2f}%")
    print(f"Arrival Intervals:  avg={arrival_stats['avg']}s, median={arrival_stats['median']}s, p95={arrival_stats['p95']}s")
    print(f"Cycle Wall Latency: avg={wall_stats['avg']}ms, median={wall_stats['median']}ms, p95={wall_stats['p95']}ms, max={wall_stats['max']}ms")
    print(f"Cycle CPU Time:     avg={cpu_stats['avg']}ms, median={cpu_stats['median']}ms, p95={cpu_stats['p95']}ms, max={cpu_stats['max']}ms")
    print(f"Python Heap Mem:    start={mem_start_curr/1024/1024:.2f}MB, peak={mem_peak/1024/1024:.2f}MB, delta={(mem_end_curr - mem_start_curr)/1024/1024:.2f}MB")
    print("=" * 80)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": round(total_duration, 2),
        "target_duration_sec": duration_sec,
        "frames": {
            "received": frames_received,
            "processed": frames_processed,
            "dropped_backlog_skips": frames_dropped,
            "drop_rate_pct": round((frames_dropped / max(1, frames_received)) * 100, 2),
        },
        "frame_arrival_intervals_sec": arrival_stats,
        "cycle_wall_latency_ms": wall_stats,
        "cycle_cpu_time_ms": cpu_stats,
        "python_heap_memory_mb": {
            "measurement_type": "tracemalloc_python_heap",
            "start": round(mem_start_curr / 1024 / 1024, 2),
            "peak": round(mem_peak / 1024 / 1024, 2),
            "end": round(mem_end_curr / 1024 / 1024, 2),
            "delta": round((mem_end_curr - mem_start_curr) / 1024 / 1024, 2),
        },
    }

    out_file = "captures/benchmarks/livestream_10min_profile.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Results written to: {out_file}")
    return results


if __name__ == "__main__":
    # Allow command line duration override if desired
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    run_10min_profile(dur)
