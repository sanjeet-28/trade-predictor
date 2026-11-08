"""Configuration manager for the Forex Trade Detection System.

Loads hierarchical configuration from YAML files and merges environment
variables from .env without hardcoding stream URLs or sensitive credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml
from dotenv import load_dotenv

from core.logger import get_logger

logger = get_logger("core.config")

# Mandatory notification string enforced by project requirements
MANDATORY_NOTIFICATION_TEXT = "TRADE IS ACTIVE!"


@dataclass
class StreamConfig:
    """Stream ingestion parameters and YouTube targets."""
    primary_url: str = "https://www.youtube.com/watch?v=3H4IVQejlDE"
    channel_url: str = "https://www.youtube.com/@DTradingTips"
    frame_width: int = 1280
    frame_height: int = 720
    pixel_format: str = "bgr24"
    frame_sample_rate_fps: float = 1.0
    reconnect_backoff_initial_sec: float = 5.0
    reconnect_backoff_max_sec: float = 60.0
    freeze_threshold_sec: float = 30.0
    freeze_mse_threshold: float = 0.001
    black_frame_threshold: float = 2.0
    max_frame_delay_sec: float = 10.0
    local_video_path: Optional[str] = None
    cookie_file: Optional[str] = None


@dataclass
class DetectionConfig:
    """Detector thresholds, geometry tolerances, and calibration settings."""
    baseline_calibration_frames: int = 15
    calibration_min_confidence: float = 0.45
    min_box_area: int = 400
    iou_match_threshold: float = 0.50
    max_centroid_distance_px: float = 50.0
    min_confidence: float = 0.70
    label_roi: Optional[Tuple[int, int, int, int]] = (0, 50, 1025, 450)
    label_max_spatial_distance_px: float = 45.0
    label_max_missing_frames: int = 10
    excluded_rois: List[Tuple[int, int, int, int]] = field(
        default_factory=lambda: [(940, 370, 1280, 480)]
    )
    # Detector B (Trade-Position Box) settings
    box_roi: Optional[Tuple[int, int, int, int]] = (0, 60, 920, 600)
    box_min_area: int = 400
    box_min_width: int = 35
    box_min_height: int = 25
    box_min_aspect_ratio: float = 0.25
    box_max_aspect_ratio: float = 8.0
    box_iou_threshold: float = 0.40
    box_max_entry_delta_y: int = 12
    box_confirmation_frames: int = 1
    box_max_missing_frames: int = 10
    # EventManager & Cross-Detector Correlation settings
    correlation_window_sec: float = 60.0
    label_confirmation_count: int = 1
    box_confirmation_count: int = 1
    cross_detector_spatial_threshold_px: float = 60.0
    cross_detector_entry_delta_y: int = 25
    candidate_expiry_sec: float = 30.0
    active_trade_max_missing_sec: float = 120.0
    chart_scroll_max_drift_px: float = 150.0
    closed_trades_history_ttl_sec: float = 86400.0
    recover_on_startup: bool = True


@dataclass
class TelegramConfig:
    """Telegram Bot credentials and messaging options."""
    bot_token: str = ""
    chat_id: str = ""
    health_chat_id: Optional[str] = None
    trade_notification_text: str = MANDATORY_NOTIFICATION_TEXT
    attach_snapshot: bool = True
    enable_health_alerts: bool = True
    screenshot_dir: str = "captures/trades"
    request_timeout_sec: float = 10.0
    initial_retry_backoff_sec: float = 5.0
    max_retry_backoff_sec: float = 60.0
    max_retries: int = 5
    max_stored_screenshots: int = 100

    def __post_init__(self) -> None:
        # Enforce that the trade notification message can only be "TRADE IS ACTIVE!"
        if self.trade_notification_text.strip() != MANDATORY_NOTIFICATION_TEXT:
            logger.warning(
                "Trade notification text '%s' is not allowed by requirements; resetting to '%s'",
                self.trade_notification_text,
                MANDATORY_NOTIFICATION_TEXT,
            )
            self.trade_notification_text = MANDATORY_NOTIFICATION_TEXT


@dataclass
class StorageConfig:
    """Database and file persistence configuration."""
    db_path: str = "storage/trade_state.db"


@dataclass
class LoggingConfig:
    """Structured logging configuration."""
    level: str = "INFO"
    file_path: str = "logs/app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10 MB
    backup_count: int = 5


@dataclass
class AppConfig:
    """Root configuration tree combining all system components."""
    stream: StreamConfig = field(default_factory=StreamConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def _find_default_config_path() -> Optional[Path]:
    """Locate config.yaml or config/config.example.yaml."""
    candidates = [
        Path("config/config.yaml"),
        Path("config.yaml"),
        Path("config/config.example.yaml"),
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def load_config(
    config_path: Optional[str | Path] = None,
    env_path: Optional[str | Path] = None,
) -> AppConfig:
    """Load, merge, and validate configuration from YAML and environment variables.

    Loading precedence (highest to lowest):
      1. Environment variables (.env)
      2. YAML file (explicit path or default candidate)
      3. Built-in dataclass defaults
    """
    # 1. Load .env secrets if present
    if env_path is not None:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    # 2. Parse YAML configuration
    resolved_yaml_path: Optional[Path] = None
    if config_path:
        resolved_yaml_path = Path(config_path)
    else:
        resolved_yaml_path = _find_default_config_path()

    raw_yaml: Dict[str, Any] = {}
    if resolved_yaml_path and resolved_yaml_path.exists():
        try:
            with open(resolved_yaml_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_yaml = loaded
            logger.info("Loaded configuration file: %s", resolved_yaml_path)
        except Exception as e:
            logger.error("Failed to parse configuration file %s: %s", resolved_yaml_path, e)
            raise

    # 3. Build component configurations
    stream_data = raw_yaml.get("stream", {})
    detection_data = raw_yaml.get("detection", {})
    telegram_data = raw_yaml.get("telegram", {})
    storage_data = raw_yaml.get("storage", {})
    logging_data = raw_yaml.get("logging", {})

    stream_cfg = StreamConfig(
        primary_url=os.getenv("STREAM_PRIMARY_URL") or stream_data.get("primary_url", StreamConfig.primary_url),
        channel_url=os.getenv("STREAM_CHANNEL_URL") or stream_data.get("channel_url", StreamConfig.channel_url),
        frame_width=int(stream_data.get("frame_width", StreamConfig.frame_width)),
        frame_height=int(stream_data.get("frame_height", StreamConfig.frame_height)),
        pixel_format=str(stream_data.get("pixel_format", StreamConfig.pixel_format)),
        frame_sample_rate_fps=float(stream_data.get("frame_sample_rate_fps", StreamConfig.frame_sample_rate_fps)),
        reconnect_backoff_initial_sec=float(stream_data.get("reconnect_backoff_initial_sec", StreamConfig.reconnect_backoff_initial_sec)),
        reconnect_backoff_max_sec=float(stream_data.get("reconnect_backoff_max_sec", StreamConfig.reconnect_backoff_max_sec)),
        freeze_threshold_sec=float(stream_data.get("freeze_threshold_sec", StreamConfig.freeze_threshold_sec)),
        freeze_mse_threshold=float(stream_data.get("freeze_mse_threshold", StreamConfig.freeze_mse_threshold)),
        black_frame_threshold=float(stream_data.get("black_frame_threshold", StreamConfig.black_frame_threshold)),
        max_frame_delay_sec=float(stream_data.get("max_frame_delay_sec", StreamConfig.max_frame_delay_sec)),
        local_video_path=os.getenv("LOCAL_VIDEO_PATH") or stream_data.get("local_video_path"),
        cookie_file=os.getenv("YOUTUBE_COOKIE_FILE") or os.getenv("COOKIE_FILE") or stream_data.get("cookie_file"),
    )

    roi_val = detection_data.get("label_roi", DetectionConfig.label_roi)
    if isinstance(roi_val, (list, tuple)) and len(roi_val) == 4:
        label_roi_tuple = (int(roi_val[0]), int(roi_val[1]), int(roi_val[2]), int(roi_val[3]))
    else:
        label_roi_tuple = DetectionConfig.label_roi

    box_roi_val = detection_data.get("box_roi", DetectionConfig.box_roi)
    if isinstance(box_roi_val, (list, tuple)) and len(box_roi_val) == 4:
        box_roi_tuple = (int(box_roi_val[0]), int(box_roi_val[1]), int(box_roi_val[2]), int(box_roi_val[3]))
    else:
        box_roi_tuple = DetectionConfig.box_roi

    excluded_rois_raw = detection_data.get("excluded_rois", None)
    excluded_rois_list: List[Tuple[int, int, int, int]] = []
    if isinstance(excluded_rois_raw, list):
        for item in excluded_rois_raw:
            if isinstance(item, (list, tuple)) and len(item) == 4:
                excluded_rois_list.append((int(item[0]), int(item[1]), int(item[2]), int(item[3])))
    if not excluded_rois_list:
        excluded_rois_list = [(940, 370, 1280, 480)]

    detection_cfg = DetectionConfig(
        baseline_calibration_frames=int(detection_data.get("baseline_calibration_frames", 15)),
        calibration_min_confidence=float(detection_data.get("calibration_min_confidence", 0.45)),
        min_box_area=int(detection_data.get("min_box_area", 400)),
        iou_match_threshold=float(detection_data.get("iou_match_threshold", 0.50)),
        max_centroid_distance_px=float(detection_data.get("max_centroid_distance_px", 50.0)),
        min_confidence=float(detection_data.get("min_confidence", 0.70)),
        label_roi=label_roi_tuple,
        label_max_spatial_distance_px=float(detection_data.get("label_max_spatial_distance_px", 45.0)),
        label_max_missing_frames=int(detection_data.get("label_max_missing_frames", 10)),
        excluded_rois=excluded_rois_list,
        box_roi=box_roi_tuple,
        box_min_area=int(detection_data.get("box_min_area", 400)),
        box_min_width=int(detection_data.get("box_min_width", 35)),
        box_min_height=int(detection_data.get("box_min_height", 25)),
        box_min_aspect_ratio=float(detection_data.get("box_min_aspect_ratio", 0.25)),
        box_max_aspect_ratio=float(detection_data.get("box_max_aspect_ratio", 8.0)),
        box_iou_threshold=float(detection_data.get("box_iou_threshold", 0.40)),
        box_max_entry_delta_y=int(detection_data.get("box_max_entry_delta_y", 12)),
        box_confirmation_frames=int(detection_data.get("box_confirmation_frames", 1)),
        box_max_missing_frames=int(detection_data.get("box_max_missing_frames", 10)),
        correlation_window_sec=float(detection_data.get("correlation_window_sec", 60.0)),
        label_confirmation_count=int(detection_data.get("label_confirmation_count", 1)),
        box_confirmation_count=int(detection_data.get("box_confirmation_count", 1)),
        cross_detector_spatial_threshold_px=float(detection_data.get("cross_detector_spatial_threshold_px", 60.0)),
        cross_detector_entry_delta_y=int(detection_data.get("cross_detector_entry_delta_y", 25)),
        candidate_expiry_sec=float(detection_data.get("candidate_expiry_sec", 30.0)),
        active_trade_max_missing_sec=float(detection_data.get("active_trade_max_missing_sec", 120.0)),
        chart_scroll_max_drift_px=float(detection_data.get("chart_scroll_max_drift_px", 150.0)),
        closed_trades_history_ttl_sec=float(detection_data.get("closed_trades_history_ttl_sec", 86400.0)),
        recover_on_startup=bool(detection_data.get("recover_on_startup", True)),
    )

    telegram_cfg = TelegramConfig(
        bot_token=os.getenv("TELEGRAM_BOT_TOKEN") or telegram_data.get("bot_token", ""),
        chat_id=os.getenv("TELEGRAM_CHAT_ID") or telegram_data.get("chat_id", ""),
        health_chat_id=os.getenv("TELEGRAM_HEALTH_CHAT_ID") or telegram_data.get("health_chat_id"),
        trade_notification_text=telegram_data.get("trade_notification_text", MANDATORY_NOTIFICATION_TEXT),
        attach_snapshot=bool(telegram_data.get("attach_snapshot", TelegramConfig.attach_snapshot)),
        enable_health_alerts=bool(telegram_data.get("enable_health_alerts", TelegramConfig.enable_health_alerts)),
        screenshot_dir=str(telegram_data.get("screenshot_dir", TelegramConfig.screenshot_dir)),
        request_timeout_sec=float(telegram_data.get("request_timeout_sec", TelegramConfig.request_timeout_sec)),
        initial_retry_backoff_sec=float(telegram_data.get("initial_retry_backoff_sec", TelegramConfig.initial_retry_backoff_sec)),
        max_retry_backoff_sec=float(telegram_data.get("max_retry_backoff_sec", TelegramConfig.max_retry_backoff_sec)),
        max_retries=int(telegram_data.get("max_retries", TelegramConfig.max_retries)),
        max_stored_screenshots=int(telegram_data.get("max_stored_screenshots", TelegramConfig.max_stored_screenshots)),
    )

    storage_cfg = StorageConfig(
        db_path=os.getenv("DB_PATH") or storage_data.get("db_path", StorageConfig.db_path),
    )

    logging_cfg = LoggingConfig(
        level=os.getenv("LOG_LEVEL") or logging_data.get("level", LoggingConfig.level),
        file_path=os.getenv("LOG_FILE") or logging_data.get("file_path", LoggingConfig.file_path),
        max_bytes=int(logging_data.get("max_bytes", LoggingConfig.max_bytes)),
        backup_count=int(logging_data.get("backup_count", LoggingConfig.backup_count)),
    )

    return AppConfig(
        stream=stream_cfg,
        detection=detection_cfg,
        telegram=telegram_cfg,
        storage=storage_cfg,
        logging=logging_cfg,
    )
