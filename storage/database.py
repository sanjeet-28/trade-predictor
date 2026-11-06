"""SQLite database connection and schema management.

Ensures schema initialization, WAL mode configuration for concurrent reads/writes,
and transaction integrity across threads.
"""

import os
import sqlite3
from contextlib import contextmanager
from typing import Generator

from core.logger import get_logger

logger = get_logger("storage.database")

SCHEMA_SQL = """
-- Trades table: tracks lifecycle of detected trades
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    detector_source TEXT NOT NULL,
    first_detected_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    notification_sent INTEGER NOT NULL DEFAULT 0,
    notification_sent_at TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_trades_state ON trades(state);
CREATE INDEX IF NOT EXISTS idx_trades_last_seen ON trades(last_seen_at);

-- Events table: raw detection triggers from Detector A and Detector B
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    trade_id TEXT,
    detector_type TEXT NOT NULL,
    signal_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp TEXT NOT NULL,
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    snapshot_path TEXT,
    raw_metadata_json TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_trade_id ON events(trade_id);

-- System health & stream monitoring events
CREATE TABLE IF NOT EXISTS health_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    level TEXT NOT NULL,
    component TEXT NOT NULL,
    message TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_health_timestamp ON health_logs(timestamp);

-- Notifications outbox: persistent delivery tracking and deduplication
CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    trade_id TEXT NOT NULL UNIQUE,
    chat_id TEXT NOT NULL,
    message_text TEXT NOT NULL,
    photo_path TEXT,
    status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    max_retries INTEGER NOT NULL DEFAULT 5,
    next_retry_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    sent_at TEXT,
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_notifications_status ON notifications(status);
CREATE INDEX IF NOT EXISTS idx_notifications_trade_id ON notifications(trade_id);
"""


def init_db(db_path: str = "storage/trade_state.db") -> None:
    """Initialize SQLite database with required tables, indexes, and pragmas."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)

    with get_db_connection(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    logger.info("Database initialized successfully at: %s", db_path)


@contextmanager
def get_db_connection(db_path: str = "storage/trade_state.db") -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a SQLite connection configured for WAL mode."""
    conn = sqlite3.connect(db_path, timeout=10.0)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

