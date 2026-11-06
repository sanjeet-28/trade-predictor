"""Forex Trade Detection & Telegram Notification System - Main Entry Point.

Usage:
    python main.py [OPTIONS]

Options:
    --config PATH         Path to custom YAML configuration file.
    --dry-run             Run in dry-run mode (suppresses all Telegram network requests).
    --simulate            Run in offline simulation mode without requiring credentials.
    --local-video PATH    Process a local video file instead of live YouTube stream.
    --max-frames N        Exit cleanly after processing N frames.
    --test-connection     Verify Telegram bot credentials and exit.
    --version             Display system version.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.config import load_config
from core.logger import get_logger, setup_logging
from core.orchestrator import TradeDetectionOrchestrator

VERSION = "1.0.0"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line flags and options."""
    parser = argparse.ArgumentParser(
        description="Forex Trade Detection & Telegram Notification System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom YAML configuration file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Run without dispatching external Telegram HTTP requests.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        default=False,
        help="Run in offline simulation mode without requiring Telegram credentials.",
    )
    parser.add_argument(
        "--local-video",
        type=str,
        default=None,
        help="Process a local video file instead of connecting to YouTube livestream.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Exit cleanly after processing N video frames.",
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        default=False,
        help="Test Telegram Bot credentials using getMe and send a test message.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
    )

    return parser.parse_args()


def main() -> int:
    """Application entrypoint."""
    args = parse_arguments()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"[FATAL] Failed to load configuration: {e}", file=sys.stderr)
        return 1

    # Initialize logging
    log_file = getattr(config.logging, "file_path", "logs/app.log") if hasattr(config, "logging") else "logs/app.log"
    setup_logging(
        log_level=getattr(config.logging, "level", "INFO") if hasattr(config, "logging") else "INFO",
        log_file=log_file,
        max_bytes=getattr(config.logging, "max_bytes", 10485760) if hasattr(config, "logging") else 10485760,
        backup_count=getattr(config.logging, "backup_count", 5) if hasattr(config, "logging") else 5,
    )
    logger = get_logger("main")
    logger.info("Initializing Forex Trade Detection System v%s (PID=%d)", VERSION, os.getpid())

    # Standalone Connection Test
    if args.test_connection:
        from notifications.test_connection import run_connection_test
        logger.info("Running standalone Telegram connection test...")
        success = run_connection_test()
        return 0 if success else 1

    # Check credentials if running in live mode (not dry-run and not simulate)
    is_dry_run = args.dry_run or args.simulate
    if not is_dry_run:
        if not config.telegram.bot_token or not config.telegram.chat_id:
            logger.warning(
                "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID is missing from .env. "
                "Running in safe DRY-RUN mode to prevent unhandled network errors."
            )
            is_dry_run = True

    # Build and launch orchestrator
    try:
        orchestrator = TradeDetectionOrchestrator(
            config=config,
            local_video_path=args.local_video,
            dry_run=is_dry_run,
            simulate_mode=args.simulate,
            max_frames=args.max_frames,
        )
        stats = orchestrator.run()

        logger.info(
            "Application run finished. Frames: %d, Activations: %d, Notifications Sent: %d",
            stats.frames_processed,
            stats.trades_activated,
            stats.notifications_sent,
        )
        return 0

    except KeyboardInterrupt:
        logger.info("Application interrupted by user. Exiting cleanly.")
        return 0
    except Exception as e:
        logger.critical("Fatal error in application execution: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
