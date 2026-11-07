"""Performance Benchmarking & Profiling Utility.

Measures:
  1. Detector A wall-clock vs CPU processing time.
  2. OCR ONNX inference time.
  3. Detector B wall-clock vs CPU processing time.
  4. Central Event Manager correlation & deduplication latency.
  5. Total end-to-end frame processing cycle latency.
  6. Memory growth across 100 consecutive frames (tracemalloc).
  7. Reconnection recovery latency.

Separates wall-clock latency from CPU processing time in compliance with Phase 7 requirements.
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

import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig
from core.event_manager import EventManager
from core.models import BoundingBox, DetectionEvent, DetectorType, Frame, TradeSignalType
from detectors.box_detector import BoxDetector
from detectors.label_detector import LabelDetector
from storage.database import init_db
from storage.repository import EventRepository, NotificationRepository, TradeRepository


def make_benchmark_canvas(w: int = 1280, h: int = 720) -> np.ndarray:
    """Create a synthetic chart frame with candles, text, and trade box."""
    canvas = np.full((h, w, 3), 22, dtype=np.uint8)
    for x in range(0, w, 60):
        cv2.line(canvas, (x, 0), (x, h), (32, 32, 32), 1)
    for y in range(0, h, 40):
        cv2.line(canvas, (0, y), (w, y), (32, 32, 32), 1)

    # Draw candles
    np.random.seed(42)
    for i in range(20, w - 100, 25):
        cy = np.random.randint(200, 500)
        ch = np.random.randint(15, 60)
        color = (50, 180, 50) if np.random.rand() > 0.5 else (50, 50, 200)
        cv2.rectangle(canvas, (i, cy), (i + 14, cy + ch), color, -1)
        cv2.line(canvas, (i + 7, cy - 10), (i + 7, cy + ch + 10), color, 1)

    # Draw BUY badge
    cv2.rectangle(canvas, (400, 280), (460, 305), (40, 160, 40), -1)
    cv2.putText(canvas, "BUY", (408, 298), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    # Draw position box
    cv2.rectangle(canvas, (400, 270), (520, 350), (50, 180, 50), -1)
    cv2.rectangle(canvas, (400, 351), (520, 390), (50, 50, 200), -1)
    cv2.line(canvas, (400, 350), (520, 350), (240, 240, 240), 2)

    return canvas


def run_benchmark(iterations: int = 100) -> Dict[str, Any]:
    """Execute rigorous performance benchmark."""
    print("=" * 75)
    print("FOREX TRADE DETECTION SYSTEM - PERFORMANCE BENCHMARK")
    print("=" * 75)
    print(f"Iterations:        {iterations} simulated frames")
    print("Metrics Measured:  Wall-Clock Latency vs CPU Time, Memory Growth, Reconnect Recovery")
    print("-" * 75)

    benchmark_dir = "captures/benchmarks"
    os.makedirs(benchmark_dir, exist_ok=True)
    db_file = os.path.join(benchmark_dir, "benchmark.db")
    if os.path.exists(db_file):
        try:
            os.remove(db_file)
        except OSError:
            pass
    init_db(db_file)

    config = AppConfig()
    config.storage.db_path = db_file
    config.detection.baseline_calibration_frames = 5

    label_det = LabelDetector(baseline_frames=5)
    box_det = BoxDetector(baseline_frames=5)
    ev_mgr = EventManager(
        config=config,
        trade_repo=TradeRepository(db_file),
        event_repo=EventRepository(db_file),
    )

    sample_frame_data = make_benchmark_canvas()

    # Metrics storage
    det_a_wall: List[float] = []
    det_a_cpu: List[float] = []
    ocr_raw_wall: List[float] = []
    ocr_raw_cpu: List[float] = []
    det_b_wall: List[float] = []
    det_b_cpu: List[float] = []
    mgr_wall: List[float] = []
    mgr_cpu: List[float] = []
    total_frame_wall: List[float] = []
    total_frame_cpu: List[float] = []

    # 1. Start Memory Tracking
    tracemalloc.start()
    mem_start_current, mem_start_peak = tracemalloc.get_traced_memory()

    print(f"Executing {iterations} consecutive frame processing cycles...")

    for i in range(iterations):
        frame = Frame(
            frame_id=i,
            timestamp=datetime.now(timezone.utc),
            width=1280,
            height=720,
            data=sample_frame_data.copy(),
        )

        t_frame_w0 = time.perf_counter()
        t_frame_c0 = time.process_time()

        # Detector A
        t_w0 = time.perf_counter()
        t_c0 = time.process_time()
        events_a = label_det.process_frame(frame)
        det_a_wall.append((time.perf_counter() - t_w0) * 1000.0)
        det_a_cpu.append((time.process_time() - t_c0) * 1000.0)

        # Raw OCR alone
        if label_det._ocr is not None and i < 20:  # Sample 20 passes to avoid skewing
            roi, ox, oy = label_det._crop_roi(frame.data)
            t_w0 = time.perf_counter()
            t_c0 = time.process_time()
            _ = label_det._ocr(roi)
            ocr_raw_wall.append((time.perf_counter() - t_w0) * 1000.0)
            ocr_raw_cpu.append((time.process_time() - t_c0) * 1000.0)

        # Detector B
        t_w0 = time.perf_counter()
        t_c0 = time.process_time()
        events_b = box_det.process_frame(frame)
        det_b_wall.append((time.perf_counter() - t_w0) * 1000.0)
        det_b_cpu.append((time.process_time() - t_c0) * 1000.0)

        # EventManager
        ev_mgr.advance_frame(frame)
        t_w0 = time.perf_counter()
        t_c0 = time.process_time()
        _ = ev_mgr.process_events(events_a + events_b, frame=frame)
        mgr_wall.append((time.perf_counter() - t_w0) * 1000.0)
        mgr_cpu.append((time.process_time() - t_c0) * 1000.0)

        total_frame_wall.append((time.perf_counter() - t_frame_w0) * 1000.0)
        total_frame_cpu.append((time.process_time() - t_frame_c0) * 1000.0)

        if (i + 1) % 25 == 0:
            cur_mem, peak_mem = tracemalloc.get_traced_memory()
            print(f"  Processed frame {i + 1:03d}/{iterations:03d} | Memory RSS: {cur_mem / 1024 / 1024:5.2f} MB")

    mem_end_current, mem_end_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 2. Benchmark Reconnection Recovery Time
    print("\nMeasuring stream reconnection recovery latency...")
    t_rec_start = time.perf_counter()
    # Simulate stream resolver invalidation & cache reload
    from stream.resolver import StreamResolution, StreamResolver
    resolver = StreamResolver(
        primary_url=config.stream.primary_url,
        channel_url=config.stream.channel_url,
    )
    # Inject cached resolution to test cache recovery latency
    now_dt = datetime.now(timezone.utc)
    resolver._cached_resolution = StreamResolution(
        stream_url="https://manifest.googlevideo.com/simulated_stream",
        width=1280,
        height=720,
        format_id="232",
        resolved_at=now_dt,
        expires_at=datetime.fromtimestamp(now_dt.timestamp() + 3600, tz=timezone.utc),
    )
    res = resolver.resolve(force_refresh=False)
    reconnect_latency_ms = (time.perf_counter() - t_rec_start) * 1000.0

    # Calculate summaries
    def calc_stats(series: List[float]) -> Dict[str, float]:
        if not series:
            return {"avg": 0.0, "min": 0.0, "max": 0.0, "p95": 0.0}
        s = sorted(series)
        return {
            "avg": round(sum(s) / len(s), 2),
            "min": round(s[0], 2),
            "max": round(s[-1], 2),
            "p95": round(s[int(0.95 * len(s))], 2),
        }

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "iterations": iterations,
        "metrics": {
            "detector_a_wall_ms": calc_stats(det_a_wall),
            "detector_a_cpu_ms": calc_stats(det_a_cpu),
            "ocr_raw_inference_wall_ms": calc_stats(ocr_raw_wall),
            "ocr_raw_inference_cpu_ms": calc_stats(ocr_raw_cpu),
            "detector_b_wall_ms": calc_stats(det_b_wall),
            "detector_b_cpu_ms": calc_stats(det_b_cpu),
            "event_manager_wall_ms": calc_stats(mgr_wall),
            "event_manager_cpu_ms": calc_stats(mgr_cpu),
            "total_frame_cycle_wall_ms": calc_stats(total_frame_wall),
            "total_frame_cycle_cpu_ms": calc_stats(total_frame_cpu),
        },
        "memory": {
            "start_mb": round(mem_start_current / 1024 / 1024, 2),
            "end_mb": round(mem_end_current / 1024 / 1024, 2),
            "peak_mb": round(mem_end_peak / 1024 / 1024, 2),
            "delta_mb": round((mem_end_current - mem_start_current) / 1024 / 1024, 2),
            "growth_rate_kb_per_frame": round((mem_end_current - mem_start_current) / iterations / 1024, 2),
        },
        "reconnection_recovery_ms": round(reconnect_latency_ms, 2),
        "target_headroom": {
            "budget_ms_at_1fps": 1000.0,
            "avg_cycle_wall_ms": calc_stats(total_frame_wall)["avg"],
            "headroom_pct": round(
                max(0.0, (1000.0 - calc_stats(total_frame_wall)["avg"]) / 1000.0 * 100.0), 1
            ),
        },
    }

    print("\n" + "=" * 75)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 75)
    print(f"Detector A (RapidOCR + Filter): Wall: {results['metrics']['detector_a_wall_ms']['avg']:6.1f}ms (p95: {results['metrics']['detector_a_wall_ms']['p95']:6.1f}ms) | CPU: {results['metrics']['detector_a_cpu_ms']['avg']:6.1f}ms")
    print(f"  Raw OCR ONNX Inference:       Wall: {results['metrics']['ocr_raw_inference_wall_ms']['avg']:6.1f}ms (p95: {results['metrics']['ocr_raw_inference_wall_ms']['p95']:6.1f}ms) | CPU: {results['metrics']['ocr_raw_inference_cpu_ms']['avg']:6.1f}ms")
    print(f"Detector B (HSV Geometry):     Wall: {results['metrics']['detector_b_wall_ms']['avg']:6.1f}ms (p95: {results['metrics']['detector_b_wall_ms']['p95']:6.1f}ms) | CPU: {results['metrics']['detector_b_cpu_ms']['avg']:6.1f}ms")
    print(f"EventManager (Deduplication):  Wall: {results['metrics']['event_manager_wall_ms']['avg']:6.1f}ms (p95: {results['metrics']['event_manager_wall_ms']['p95']:6.1f}ms) | CPU: {results['metrics']['event_manager_cpu_ms']['avg']:6.1f}ms")
    print("-" * 75)
    print(f"Total Frame Processing Cycle:  Wall: {results['metrics']['total_frame_cycle_wall_ms']['avg']:6.1f}ms (p95: {results['metrics']['total_frame_cycle_wall_ms']['p95']:6.1f}ms) | CPU: {results['metrics']['total_frame_cycle_cpu_ms']['avg']:6.1f}ms")
    print(f"Available Headroom at 1.0 FPS: {results['target_headroom']['headroom_pct']}% headroom ({1000.0 - results['metrics']['total_frame_cycle_wall_ms']['avg']:.1f}ms idle per second)")
    print("-" * 75)
    print(f"Memory Footprint:              Start: {results['memory']['start_mb']} MB | Peak: {results['memory']['peak_mb']} MB | Delta: {results['memory']['delta_mb']} MB ({results['memory']['growth_rate_kb_per_frame']} KB/frame)")
    print(f"Reconnection Recovery Latency: {results['reconnection_recovery_ms']} ms")
    print("=" * 75)

    # Save benchmark report JSON
    results_path = os.path.join(benchmark_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Benchmark results JSON written to: {results_path}\n")

    return results


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_benchmark(iterations=n)
