"""Safe Non-Blocking SQLite Online Backup Utility.

Uses SQLite's native Online Backup API (`sqlite3.Connection.backup`) to create
point-in-time database backups without write locks or interrupting active processing.

Usage:
    python -m scripts.backup_db [--source PATH] [--dest PATH]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.logger import get_logger, setup_logging

logger = get_logger("scripts.backup_db")


def backup_database(
    source_db: Optional[str] = None,
    backup_dir: str = "backups",
    max_backups_to_retain: int = 14,
) -> str:
    """Execute safe non-blocking SQLite online backup."""
    cfg = load_config()
    src = source_db or getattr(cfg.storage, "db_path", "storage/trade_state.db")

    if not os.path.exists(src):
        raise FileNotFoundError(f"Source database does not exist: {src}")

    os.makedirs(backup_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = os.path.join(backup_dir, f"trade_state_backup_{timestamp}.db")

    logger.info("Initiating SQLite online backup: %s -> %s", src, dest)
    t0 = time.perf_counter()

    # Use SQLite native online backup API
    with sqlite3.connect(src) as src_conn:
        with sqlite3.connect(dest) as dest_conn:
            # Performs atomic backup in 100-page chunks without locking readers/writers
            src_conn.backup(dest_conn, pages=100)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    backup_size = os.path.getsize(dest)
    logger.info(
        "SQLite online backup completed in %.2fms (size: %.1f KB). File: %s",
        elapsed_ms, backup_size / 1024.0, dest,
    )

    # Retention Pruning: Keep latest N backups
    existing_backups = sorted(
        [
            os.path.join(backup_dir, f)
            for f in os.listdir(backup_dir)
            if f.startswith("trade_state_backup_") and f.endswith(".db")
        ]
    )
    if len(existing_backups) > max_backups_to_retain:
        for old_file in existing_backups[:-max_backups_to_retain]:
            try:
                os.remove(old_file)
                logger.info("Pruned old database backup: %s", old_file)
            except OSError as e:
                logger.warning("Failed to remove old backup %s: %s", old_file, e)

    return dest


def restore_database(backup_file: str, target_db: Optional[str] = None) -> None:
    """Restore database from a point-in-time backup file."""
    cfg = load_config()
    target = target_db or getattr(cfg.storage, "db_path", "storage/trade_state.db")

    if not os.path.exists(backup_file):
        raise FileNotFoundError(f"Backup file not found: {backup_file}")

    logger.warning("Restoring database from %s to %s...", backup_file, target)
    target_dir = os.path.dirname(target)
    if target_dir:
        os.makedirs(target_dir, exist_ok=True)

    # Create safety copy of current target if exists
    if os.path.exists(target):
        safety_path = f"{target}.pre_restore_bak"
        shutil.copy2(target, safety_path)
        logger.info("Created pre-restore safety copy at %s", safety_path)

    # Use SQLite online backup API in reverse
    with sqlite3.connect(backup_file) as src_conn:
        with sqlite3.connect(target) as dest_conn:
            src_conn.backup(dest_conn, pages=100)

    logger.info("Database restoration complete. Verifying schema integrity...")
    with sqlite3.connect(target) as conn:
        res = conn.execute("PRAGMA integrity_check;").fetchone()
        if res and res[0] == "ok":
            logger.info("Database integrity verified: OK")
        else:
            raise RuntimeError(f"Database integrity check failed: {res}")


def main() -> int:
    """CLI handler for backup and restore."""
    parser = argparse.ArgumentParser(description="SQLite Online Backup & Restore Utility")
    parser.add_argument("--source", type=str, default=None, help="Source database path")
    parser.add_argument("--dest-dir", type=str, default="backups", help="Backup output directory")
    parser.add_argument("--restore", type=str, default=None, help="Restore from specified backup file")
    args = parser.parse_args()

    setup_logging(log_level="INFO")

    try:
        if args.restore:
            restore_database(args.restore, target_db=args.source)
            print(f"[SUCCESS] Restored database from {args.restore}")
        else:
            dest = backup_database(source_db=args.source, backup_dir=args.dest_dir)
            print(f"[SUCCESS] Backup created at: {dest}")
        return 0
    except Exception as e:
        print(f"[ERROR] Operation failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

