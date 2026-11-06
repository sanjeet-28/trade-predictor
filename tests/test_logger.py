"""Unit tests for structured logging."""

import logging
from pathlib import Path

from core.logger import get_logger, setup_logging


def test_logger_setup_creates_file(tmp_path: Path) -> None:
    """Verify setup_logging creates the directory and writes log entries to disk."""
    log_file = tmp_path / "test_logs" / "test.log"
    logger = setup_logging(log_level="DEBUG", log_file=str(log_file))

    assert isinstance(logger, logging.Logger)
    logger.debug("Test debug message")
    logger.info("Test info message")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test debug message" in content
    assert "Test info message" in content


def test_get_logger_namespacing() -> None:
    """Verify get_logger scopes namespaces appropriately."""
    sub_logger = get_logger("detector.box")
    assert sub_logger.name == "trade_detector.detector.box"

