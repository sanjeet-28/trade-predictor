"""Structured logging module for the Forex Trade Detection System.

Configures formatted console output and rotating file logging with
automatic directory creation and thread-safe handling.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Optional

# Base logger namespace
LOGGER_PREFIX = "trade_detector"

_is_configured = False


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = "logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
) -> logging.Logger:
    """Configure root application logging with console and rotating file handlers."""
    global _is_configured

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_app_logger = logging.getLogger(LOGGER_PREFIX)
    root_app_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    if root_app_logger.handlers:
        root_app_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_app_logger.addHandler(console_handler)

    # 2. Rotating File Handler (if log_file specified)
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_app_logger.addHandler(file_handler)

    _is_configured = True
    root_app_logger.info("Structured logging initialized (level=%s, file=%s)", log_level, log_file)
    return root_app_logger


def get_logger(module_name: str) -> logging.Logger:
    """Return a scoped logger under the trade_detector namespace.

    Example: get_logger("stream.resolver") -> trade_detector.stream.resolver
    """
    if not _is_configured:
        setup_logging()

    if module_name.startswith(LOGGER_PREFIX):
        return logging.getLogger(module_name)
    return logging.getLogger(f"{LOGGER_PREFIX}.{module_name}")

