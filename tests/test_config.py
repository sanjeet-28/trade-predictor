"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path

from core.config import (
    MANDATORY_NOTIFICATION_TEXT,
    AppConfig,
    StreamConfig,
    TelegramConfig,
    load_config,
)


def test_default_config_urls() -> None:
    """Verify official primary and channel URLs match project specifications."""
    cfg = load_config()
    assert cfg.stream.primary_url == "https://www.youtube.com/watch?v=3H4IVQejlDE"
    assert cfg.stream.channel_url == "https://www.youtube.com/@DTradingTips"
    assert cfg.stream.frame_sample_rate_fps == 1.0


def test_telegram_mandatory_notification_text() -> None:
    """Verify notification text is strictly 'TRADE IS ACTIVE!'."""
    cfg = load_config()
    assert cfg.telegram.trade_notification_text == MANDATORY_NOTIFICATION_TEXT


def test_telegram_notification_text_enforcement() -> None:
    """Verify attempts to change notification text to leak trade info are rejected."""
    bad_config = TelegramConfig(trade_notification_text="BUY EURUSD 1.0850 SL 1.0800")
    assert bad_config.trade_notification_text == MANDATORY_NOTIFICATION_TEXT


def test_env_var_override(monkeypatch) -> None:
    """Verify environment variables take precedence over config files."""
    monkeypatch.setenv("STREAM_PRIMARY_URL", "https://www.youtube.com/watch?v=CustomOverride123")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "mock_bot_token_test")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "123456789")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    cfg = load_config()
    assert cfg.stream.primary_url == "https://www.youtube.com/watch?v=CustomOverride123"
    assert cfg.telegram.bot_token == "mock_bot_token_test"
    assert cfg.telegram.chat_id == "123456789"
    assert cfg.logging.level == "DEBUG"


def test_example_yaml_exists_and_loads() -> None:
    """Verify config/config.example.yaml can be parsed successfully."""
    example_path = Path("config/config.example.yaml")
    assert example_path.exists(), "config.example.yaml must exist"
    cfg = load_config(config_path=example_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.detection.baseline_calibration_frames in (10, 15)
    assert cfg.detection.box_roi == (0, 60, 920, 600)
    assert cfg.detection.box_min_area == 400
    assert cfg.detection.box_min_width == 35
    assert cfg.detection.box_min_height == 25
    assert cfg.detection.box_iou_threshold == 0.40
    assert cfg.detection.box_max_entry_delta_y == 12
    assert cfg.detection.box_confirmation_frames == 1
    assert cfg.detection.box_max_missing_frames == 10
    assert cfg.detection.correlation_window_sec == 60.0
    assert cfg.detection.label_confirmation_count == 1
    assert cfg.detection.box_confirmation_count == 1
    assert cfg.detection.cross_detector_spatial_threshold_px == 60.0
    assert cfg.detection.cross_detector_entry_delta_y == 25
    assert cfg.detection.candidate_expiry_sec == 30.0
    assert cfg.detection.active_trade_max_missing_sec == 120.0
    assert cfg.detection.recover_on_startup is True
    assert cfg.telegram.screenshot_dir == "captures/trades"
    assert cfg.telegram.request_timeout_sec == 10.0
    assert cfg.telegram.initial_retry_backoff_sec == 5.0
    assert cfg.telegram.max_retry_backoff_sec == 60.0
    assert cfg.telegram.max_retries == 5
    assert cfg.telegram.max_stored_screenshots == 100

