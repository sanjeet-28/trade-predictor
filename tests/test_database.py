"""Unit tests for SQLite schema initialization and repository operations."""

from datetime import datetime, timezone
from pathlib import Path

from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from storage.database import get_db_connection, init_db
from storage.repository import EventRepository, HealthLogRepository, TradeRepository


def test_database_init(tmp_path: Path) -> None:
    """Verify SQLite initialization creates expected tables and indexes."""
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    with get_db_connection(db_path) as conn:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
        ).fetchall()
        table_names = [t["name"] for t in tables]
        assert "trades" in table_names
        assert "events" in table_names
        assert "health_logs" in table_names


def test_trade_repository_crud(tmp_path: Path) -> None:
    """Verify saving, retrieving, updating and filtering trades."""
    db_path = str(tmp_path / "test_trades.db")
    init_db(db_path)
    repo = TradeRepository(db_path)

    # 1. Create a trade
    now = datetime.now(timezone.utc)
    trade = TradeRecord(
        trade_id="trade-xyz-123",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        first_detected_at=now,
        last_seen_at=now,
        bbox=BoundingBox(10, 20, 30, 40),
        notification_sent=False,
        metadata={"pair": "EURUSD"},
    )
    repo.save_trade(trade)

    # 2. Retrieve the trade
    fetched = repo.get_trade("trade-xyz-123")
    assert fetched is not None
    assert fetched.trade_id == "trade-xyz-123"
    assert fetched.state == TradeState.ACTIVE
    assert fetched.bbox == BoundingBox(10, 20, 30, 40)
    assert fetched.metadata == {"pair": "EURUSD"}

    # 3. Mark notification sent
    repo.mark_notified("trade-xyz-123")
    updated = repo.get_trade("trade-xyz-123")
    assert updated is not None
    assert updated.notification_sent is True
    assert updated.notification_sent_at is not None

    # 4. Update state to CLOSED
    repo.update_state("trade-xyz-123", TradeState.CLOSED)
    closed = repo.get_trade("trade-xyz-123")
    assert closed is not None
    assert closed.state == TradeState.CLOSED

    # 5. Check active filtering
    active_trades = repo.get_active_and_baseline_trades()
    assert len(active_trades) == 0


def test_event_repository(tmp_path: Path) -> None:
    """Verify persisting and querying detection events."""
    db_path = str(tmp_path / "test_events.db")
    init_db(db_path)
    trade_repo = TradeRepository(db_path)
    event_repo = EventRepository(db_path)

    # Create associated trade first (for foreign key)
    trade = TradeRecord(
        trade_id="parent-trade-1",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.BOX,
        bbox=BoundingBox(0, 0, 10, 10),
    )
    trade_repo.save_trade(trade)

    event = DetectionEvent(
        event_id="evt-999",
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        confidence=0.88,
        bbox=BoundingBox(0, 0, 10, 10),
        raw_metadata={"ratio": 1.5},
    )
    event_repo.save_event(event, trade_id="parent-trade-1")

    events = event_repo.get_events_for_trade("parent-trade-1")
    assert len(events) == 1
    assert events[0].event_id == "evt-999"
    assert events[0].confidence == 0.88
    assert events[0].raw_metadata == {"ratio": 1.5}


def test_health_log_repository(tmp_path: Path) -> None:
    """Verify logging and retrieving system diagnostic records."""
    db_path = str(tmp_path / "test_health.db")
    init_db(db_path)
    repo = HealthLogRepository(db_path)

    repo.log_health(level="WARNING", component="stream.health", message="Frame lag detected")
    repo.log_health(level="ERROR", component="stream.ffmpeg", message="Process exited with code 1")

    logs = repo.get_recent_logs(limit=10)
    assert len(logs) == 2
    assert logs[0]["component"] == "stream.ffmpeg"
    assert logs[0]["level"] == "ERROR"
    assert logs[1]["component"] == "stream.health"
    assert logs[1]["level"] == "WARNING"

