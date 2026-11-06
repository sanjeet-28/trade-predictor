"""Repository layer providing type-safe CRUD operations for SQLite storage."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from core.logger import get_logger
from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    NotificationRecord,
    NotificationStatus,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from storage.database import get_db_connection

logger = get_logger("storage.repository")


class TradeRepository:
    """Manages persistence and querying of tracked trades."""

    def __init__(self, db_path: str = "storage/trade_state.db") -> None:
        self.db_path = db_path

    def save_trade(self, trade: TradeRecord) -> None:
        """Insert or replace a trade record."""
        query = """
        INSERT INTO trades (
            trade_id, state, detector_source, first_detected_at, last_seen_at,
            bbox_x, bbox_y, bbox_width, bbox_height,
            notification_sent, notification_sent_at, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO UPDATE SET
            state = excluded.state,
            last_seen_at = excluded.last_seen_at,
            bbox_x = excluded.bbox_x,
            bbox_y = excluded.bbox_y,
            bbox_width = excluded.bbox_width,
            bbox_height = excluded.bbox_height,
            notification_sent = excluded.notification_sent,
            notification_sent_at = excluded.notification_sent_at,
            metadata_json = excluded.metadata_json;
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                query,
                (
                    trade.trade_id,
                    trade.state.value,
                    trade.detector_source.value,
                    trade.first_detected_at.isoformat(),
                    trade.last_seen_at.isoformat(),
                    trade.bbox.x,
                    trade.bbox.y,
                    trade.bbox.width,
                    trade.bbox.height,
                    1 if trade.notification_sent else 0,
                    trade.notification_sent_at.isoformat() if trade.notification_sent_at else None,
                    json.dumps(trade.metadata),
                ),
            )
            conn.commit()

    def get_trade(self, trade_id: str) -> Optional[TradeRecord]:
        """Retrieve a specific trade record by its UUID."""
        query = "SELECT * FROM trades WHERE trade_id = ?;"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(query, (trade_id,)).fetchone()
            if not row:
                return None
            return self._row_to_trade(row)

    def get_active_and_baseline_trades(self) -> List[TradeRecord]:
        """Fetch all trades currently marked as ACTIVE or BASELINE."""
        query = "SELECT * FROM trades WHERE state IN (?, ?) ORDER BY first_detected_at DESC;"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(query, (TradeState.ACTIVE.value, TradeState.BASELINE.value)).fetchall()
            return [self._row_to_trade(r) for r in rows]

    def update_state(self, trade_id: str, new_state: TradeState) -> None:
        """Update lifecycle state of a trade."""
        query = "UPDATE trades SET state = ? WHERE trade_id = ?;"
        with get_db_connection(self.db_path) as conn:
            conn.execute(query, (new_state.value, trade_id))
            conn.commit()

    def mark_notified(self, trade_id: str, sent_at: Optional[datetime] = None) -> None:
        """Mark that a Telegram notification has been sent for this trade."""
        timestamp = (sent_at or datetime.now(timezone.utc)).isoformat()
        query = "UPDATE trades SET notification_sent = 1, notification_sent_at = ? WHERE trade_id = ?;"
        with get_db_connection(self.db_path) as conn:
            conn.execute(query, (timestamp, trade_id))
            conn.commit()

    def _row_to_trade(self, row: Any) -> TradeRecord:
        """Convert a SQLite row to a TradeRecord domain object."""
        return TradeRecord(
            trade_id=row["trade_id"],
            state=TradeState(row["state"]),
            detector_source=DetectorType(row["detector_source"]),
            first_detected_at=datetime.fromisoformat(row["first_detected_at"]),
            last_seen_at=datetime.fromisoformat(row["last_seen_at"]),
            bbox=BoundingBox(
                x=row["bbox_x"],
                y=row["bbox_y"],
                width=row["bbox_width"],
                height=row["bbox_height"],
            ),
            notification_sent=bool(row["notification_sent"]),
            notification_sent_at=(
                datetime.fromisoformat(row["notification_sent_at"])
                if row["notification_sent_at"]
                else None
            ),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )


class EventRepository:
    """Manages persistence of raw detection events from detectors."""

    def __init__(self, db_path: str = "storage/trade_state.db") -> None:
        self.db_path = db_path

    def save_event(self, event: DetectionEvent, trade_id: Optional[str] = None) -> None:
        """Store a detection event associated optionally with a trade using safe native upsert."""
        query = """
        INSERT INTO events (
            event_id, trade_id, detector_type, signal_type, confidence,
            timestamp, bbox_x, bbox_y, bbox_width, bbox_height,
            snapshot_path, raw_metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            trade_id = excluded.trade_id,
            detector_type = excluded.detector_type,
            signal_type = excluded.signal_type,
            confidence = excluded.confidence,
            timestamp = excluded.timestamp,
            bbox_x = excluded.bbox_x,
            bbox_y = excluded.bbox_y,
            bbox_width = excluded.bbox_width,
            bbox_height = excluded.bbox_height,
            snapshot_path = excluded.snapshot_path,
            raw_metadata_json = excluded.raw_metadata_json;
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                query,
                (
                    event.event_id,
                    trade_id,
                    event.detector_type.value,
                    event.signal_type.value,
                    event.confidence,
                    event.timestamp.isoformat(),
                    event.bbox.x,
                    event.bbox.y,
                    event.bbox.width,
                    event.bbox.height,
                    event.snapshot_path,
                    json.dumps(event.raw_metadata),
                ),
            )
            conn.commit()

    def get_events_for_trade(self, trade_id: str) -> List[DetectionEvent]:
        """Fetch all raw detection events associated with a trade."""
        query = "SELECT * FROM events WHERE trade_id = ? ORDER BY timestamp ASC;"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(query, (trade_id,)).fetchall()
            return [self._row_to_event(r) for r in rows]

    def _row_to_event(self, row: Any) -> DetectionEvent:
        """Convert a SQLite row to a DetectionEvent domain object."""
        return DetectionEvent(
            event_id=row["event_id"],
            detector_type=DetectorType(row["detector_type"]),
            signal_type=TradeSignalType(row["signal_type"]),
            confidence=float(row["confidence"]),
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bbox=BoundingBox(
                x=row["bbox_x"],
                y=row["bbox_y"],
                width=row["bbox_width"],
                height=row["bbox_height"],
            ),
            snapshot_path=row["snapshot_path"],
            raw_metadata=json.loads(row["raw_metadata_json"] or "{}"),
        )


class HealthLogRepository:
    """Manages system health and stream diagnostic records."""

    def __init__(self, db_path: str = "storage/trade_state.db") -> None:
        self.db_path = db_path

    def log_health(self, level: str, component: str, message: str) -> None:
        """Record a stream or system diagnostic event."""
        query = "INSERT INTO health_logs (timestamp, level, component, message) VALUES (?, ?, ?, ?);"
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                query,
                (
                    datetime.now(timezone.utc).isoformat(),
                    level.upper(),
                    component,
                    message,
                ),
            )
            conn.commit()

    def get_recent_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent diagnostic records."""
        query = "SELECT * FROM health_logs ORDER BY timestamp DESC LIMIT ?;"
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]


class NotificationRepository:
    """Manages persistence, queueing, and deduplication of the notification outbox."""

    def __init__(self, db_path: str = "storage/trade_state.db") -> None:
        self.db_path = db_path

    def create_notification(self, record: NotificationRecord) -> bool:
        """Insert a notification into the outbox. Returns True if inserted, False if duplicate trade."""
        query = """
        INSERT INTO notifications (
            notification_id, trade_id, chat_id, message_text, photo_path,
            status, retry_count, max_retries, next_retry_at, last_error,
            created_at, updated_at, sent_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(trade_id) DO NOTHING;
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                query,
                (
                    record.notification_id,
                    record.trade_id,
                    record.chat_id,
                    record.message_text,
                    record.photo_path,
                    record.status.value,
                    record.retry_count,
                    record.max_retries,
                    record.next_retry_at.isoformat() if record.next_retry_at else None,
                    record.last_error,
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.sent_at.isoformat() if record.sent_at else None,
                ),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_pending_notifications(self, current_time: Optional[datetime] = None) -> List[NotificationRecord]:
        """Fetch all PENDING or retryable SENDING notifications due for delivery."""
        now_str = (current_time or datetime.now(timezone.utc)).isoformat()
        query = """
        SELECT * FROM notifications
        WHERE status IN ('PENDING', 'SENDING')
          AND (next_retry_at IS NULL OR next_retry_at <= ?)
        ORDER BY created_at ASC;
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(query, (now_str,)).fetchall()
            return [self._row_to_record(r) for r in rows]

    def mark_sending(self, notification_id: str) -> None:
        """Mark notification as in-flight."""
        now_str = datetime.now(timezone.utc).isoformat()
        query = "UPDATE notifications SET status = ?, updated_at = ? WHERE notification_id = ?;"
        with get_db_connection(self.db_path) as conn:
            conn.execute(query, (NotificationStatus.SENDING.value, now_str, notification_id))
            conn.commit()

    def mark_sent(self, notification_id: str, sent_at: Optional[datetime] = None) -> None:
        """Mark notification as confirmed delivered by Telegram."""
        now = sent_at or datetime.now(timezone.utc)
        query = """
        UPDATE notifications
        SET status = ?, sent_at = ?, updated_at = ?
        WHERE notification_id = ?;
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                query,
                (NotificationStatus.SENT.value, now.isoformat(), now.isoformat(), notification_id),
            )
            conn.commit()

    def mark_failed(
        self,
        notification_id: str,
        error_message: str,
        permanent: bool = False,
        next_retry_at: Optional[datetime] = None,
    ) -> None:
        """Record delivery failure with backoff scheduling or permanent rejection."""
        status = NotificationStatus.FAILED if permanent else NotificationStatus.PENDING
        now_str = datetime.now(timezone.utc).isoformat()
        next_retry_str = next_retry_at.isoformat() if next_retry_at else None

        query = """
        UPDATE notifications
        SET status = ?,
            retry_count = retry_count + 1,
            last_error = ?,
            next_retry_at = ?,
            updated_at = ?
        WHERE notification_id = ?;
        """
        with get_db_connection(self.db_path) as conn:
            conn.execute(
                query,
                (status.value, error_message, next_retry_str, now_str, notification_id),
            )
            conn.commit()

    def get_notification_by_trade_id(self, trade_id: str) -> Optional[NotificationRecord]:
        """Query notification entry by trade UUID."""
        query = "SELECT * FROM notifications WHERE trade_id = ?;"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(query, (trade_id,)).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def get_notification(self, notification_id: str) -> Optional[NotificationRecord]:
        """Query notification entry by its notification UUID."""
        query = "SELECT * FROM notifications WHERE notification_id = ?;"
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(query, (notification_id,)).fetchone()
            if not row:
                return None
            return self._row_to_record(row)

    def recover_interrupted_sending(self, stale_threshold_sec: float = 300.0) -> int:
        """Reset interrupted SENDING records back to PENDING so they are retried.

        Recovers records left in SENDING state due to sudden process exit or crash.
        When stale_threshold_sec <= 0, resets all SENDING records unconditionally (e.g. at startup).

        Returns:
            Number of records successfully recovered back to PENDING.
        """
        now = datetime.now(timezone.utc)
        now_str = now.isoformat()

        if stale_threshold_sec <= 0:
            query = """
            UPDATE notifications
            SET status = ?, updated_at = ?
            WHERE status = ?;
            """
            params: tuple = (NotificationStatus.PENDING.value, now_str, NotificationStatus.SENDING.value)
        else:
            cutoff = (now - timedelta(seconds=stale_threshold_sec)).isoformat()
            query = """
            UPDATE notifications
            SET status = ?, updated_at = ?
            WHERE status = ? AND updated_at <= ?;
            """
            params = (NotificationStatus.PENDING.value, now_str, NotificationStatus.SENDING.value, cutoff)

        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(query, params)
            conn.commit()
            count = cursor.rowcount
            if count > 0:
                logger.warning(
                    "Recovered %d interrupted 'SENDING' notifications back to 'PENDING'.", count
                )
            return count

    def _row_to_record(self, row: Any) -> NotificationRecord:
        """Convert a SQLite row to a NotificationRecord."""
        return NotificationRecord(
            notification_id=row["notification_id"],
            trade_id=row["trade_id"],
            chat_id=row["chat_id"],
            message_text=row["message_text"],
            photo_path=row["photo_path"],
            status=NotificationStatus(row["status"]),
            retry_count=int(row["retry_count"]),
            max_retries=int(row["max_retries"]),
            next_retry_at=datetime.fromisoformat(row["next_retry_at"]) if row["next_retry_at"] else None,
            last_error=row["last_error"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            sent_at=datetime.fromisoformat(row["sent_at"]) if row["sent_at"] else None,
        )

