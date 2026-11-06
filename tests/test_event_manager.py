"""Comprehensive unit tests for the Central Event Manager & Deduplication Engine.

Tests cover all 20 required scenarios:
  1. Detector A alone activates a BUY trade.
  2. Detector A alone activates a SELL trade.
  3. Detector B alone activates a BUY trade.
  4. Detector B alone activates a SELL trade.
  5. A and B reporting the same trade produce one activation.
  6. A and B reporting the same trade in reverse order produce one activation.
  7. Two separate BUY trades are not merged.
  8. Two separate SELL trades are not merged.
  9. BUY and SELL trades can both be active.
  10. A new trade at a previously used location can activate after the previous trade closes.
  11. Delayed detector events are handled correctly.
  12. Out-of-order timestamps are handled correctly.
  13. Missing entry-price metadata does not break correlation.
  14. Missing bounding-box metadata does not break event processing.
  15. Duplicate events after application restart do not create duplicate activations.
  16. Database transaction failures are handled safely.
  17. Expired candidates are cleaned up.
  18. Detector A and B can independently generate events while the other detector is unavailable.
  19. Repeated events from the same detector do not generate repeated activations.
  20. A false candidate that fails temporal confirmation does not activate.
"""

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from core.event_manager import EventManager
from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    Frame,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from storage.database import init_db
from storage.repository import EventRepository, TradeRepository


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db(tmp_path: Path):
    """Create an isolated temporary SQLite database for testing."""
    db_file = str(tmp_path / "trade_test.db")
    init_db(db_file)
    trade_repo = TradeRepository(db_file)
    event_repo = EventRepository(db_file)
    return trade_repo, event_repo


def make_label_event(
    signal_type: TradeSignalType = TradeSignalType.BUY,
    x: int = 200,
    y: int = 250,
    width: int = 40,
    height: int = 20,
    timestamp: datetime = None,
) -> DetectionEvent:
    """Helper to create a synthetic Detector A (label) event."""
    return DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=signal_type,
        confidence=0.95,
        bbox=BoundingBox(x, y, width, height),
        timestamp=timestamp or datetime.now(timezone.utc),
    )


def make_box_event(
    signal_type: TradeSignalType = TradeSignalType.BUY,
    x: int = 180,
    y: int = 200,
    width: int = 120,
    height: int = 100,
    entry_y: int = 260,
    timestamp: datetime = None,
) -> DetectionEvent:
    """Helper to create a synthetic Detector B (box) event."""
    return DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=signal_type,
        confidence=0.92,
        bbox=BoundingBox(x, y, width, height),
        timestamp=timestamp or datetime.now(timezone.utc),
        raw_metadata={"entry_y": entry_y},
    )


# ---------------------------------------------------------------------------
# Test Cases 1 - 20
# ---------------------------------------------------------------------------

def test_detector_a_alone_activates_buy_trade(temp_db) -> None:
    """Scenario 1: Detector A alone activates a BUY trade."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_label_event(TradeSignalType.BUY)
    activated = manager.process_event(event)

    assert activated is not None
    assert activated.state == TradeState.ACTIVE
    assert activated.detector_source == DetectorType.LABEL
    assert activated.metadata["signal_type"] == TradeSignalType.BUY.value
    assert len(manager.get_active_trades()) == 1


def test_detector_a_alone_activates_sell_trade(temp_db) -> None:
    """Scenario 2: Detector A alone activates a SELL trade."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_label_event(TradeSignalType.SELL)
    activated = manager.process_event(event)

    assert activated is not None
    assert activated.state == TradeState.ACTIVE
    assert activated.detector_source == DetectorType.LABEL
    assert activated.metadata["signal_type"] == TradeSignalType.SELL.value


def test_detector_b_alone_activates_buy_trade(temp_db) -> None:
    """Scenario 3: Detector B alone activates a BUY trade."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_box_event(TradeSignalType.BUY, entry_y=260)
    activated = manager.process_event(event)

    assert activated is not None
    assert activated.state == TradeState.ACTIVE
    assert activated.detector_source == DetectorType.BOX
    assert activated.metadata["entry_y"] == 260


def test_detector_b_alone_activates_sell_trade(temp_db) -> None:
    """Scenario 4: Detector B alone activates a SELL trade."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_box_event(TradeSignalType.SELL, entry_y=240)
    activated = manager.process_event(event)

    assert activated is not None
    assert activated.state == TradeState.ACTIVE
    assert activated.detector_source == DetectorType.BOX
    assert activated.metadata["signal_type"] == TradeSignalType.SELL.value


def test_a_and_b_reporting_same_trade_produce_one_activation(temp_db) -> None:
    """Scenario 5: A and B reporting the same trade produce only 1 activation."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event_a = make_label_event(TradeSignalType.BUY, x=210, y=250)
    event_b = make_box_event(TradeSignalType.BUY, x=180, y=200, width=120, height=100, entry_y=260)

    # Detector A arrives first
    act_1 = manager.process_event(event_a)
    assert act_1 is not None

    # Detector B arrives second with matching position box
    act_2 = manager.process_event(event_b)
    assert act_2 is None, "Matching Detector B event must be deduplicated"

    # Only 1 trade exists
    active_trades = manager.get_active_trades()
    assert len(active_trades) == 1
    assert "BOX_DETECTOR" in active_trades[0].metadata["correlated_detectors"]
    assert "LABEL_DETECTOR" in active_trades[0].metadata["correlated_detectors"]


def test_a_and_b_reporting_same_trade_reverse_order(temp_db) -> None:
    """Scenario 6: A and B reporting the same trade in reverse order produce only 1 activation."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event_a = make_label_event(TradeSignalType.BUY, x=210, y=250)
    event_b = make_box_event(TradeSignalType.BUY, x=180, y=200, width=120, height=100, entry_y=260)

    # Detector B arrives first
    act_1 = manager.process_event(event_b)
    assert act_1 is not None

    # Detector A arrives second
    act_2 = manager.process_event(event_a)
    assert act_2 is None

    active_trades = manager.get_active_trades()
    assert len(active_trades) == 1
    assert "BOX_DETECTOR" in active_trades[0].metadata["correlated_detectors"]
    assert "LABEL_DETECTOR" in active_trades[0].metadata["correlated_detectors"]


def test_two_separate_buy_trades_not_merged(temp_db) -> None:
    """Scenario 7: Two separate BUY trades at distinct coordinates are not merged."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event1 = make_label_event(TradeSignalType.BUY, x=100, y=100)
    event2 = make_label_event(TradeSignalType.BUY, x=700, y=400)

    act_1 = manager.process_event(event1)
    act_2 = manager.process_event(event2)

    assert act_1 is not None
    assert act_2 is not None
    assert act_1.trade_id != act_2.trade_id
    assert len(manager.get_active_trades()) == 2


def test_two_separate_sell_trades_not_merged(temp_db) -> None:
    """Scenario 8: Two separate SELL trades at distinct coordinates are not merged."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event1 = make_box_event(TradeSignalType.SELL, x=150, y=100, entry_y=150)
    event2 = make_box_event(TradeSignalType.SELL, x=150, y=450, entry_y=500)

    act_1 = manager.process_event(event1)
    act_2 = manager.process_event(event2)

    assert act_1 is not None
    assert act_2 is not None
    assert act_1.trade_id != act_2.trade_id
    assert len(manager.get_active_trades()) == 2


def test_buy_and_sell_trades_both_active(temp_db) -> None:
    """Scenario 9: BUY and SELL trades can both be simultaneously active without merging."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Even at the exact same location, opposite directions must never be merged
    buy_event = make_label_event(TradeSignalType.BUY, x=200, y=200)
    sell_event = make_label_event(TradeSignalType.SELL, x=200, y=200)

    act_buy = manager.process_event(buy_event)
    act_sell = manager.process_event(sell_event)

    assert act_buy is not None
    assert act_sell is not None
    assert act_buy.trade_id != act_sell.trade_id
    assert len(manager.get_active_trades()) == 2


def test_new_trade_at_previously_used_location_after_closure(temp_db) -> None:
    """Scenario 10: New trade at same location activates after previous trade is closed."""
    trade_repo, event_repo = temp_db
    manager = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        active_trade_max_missing_sec=60.0,
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    event1 = make_label_event(TradeSignalType.BUY, x=200, y=200, timestamp=t0)
    act_1 = manager.process_event(event1)
    assert act_1 is not None

    # Time advances 100 seconds (trade becomes inactive and is pruned)
    t1 = t0 + timedelta(seconds=100)
    closed = manager.prune_inactive_trades(t1)
    assert act_1.trade_id in closed
    assert len(manager.get_active_trades()) == 0

    # Fresh trade appears at the exact same coordinate
    t2 = t1 + timedelta(seconds=10)
    event2 = make_label_event(TradeSignalType.BUY, x=200, y=200, timestamp=t2)
    act_2 = manager.process_event(event2)

    assert act_2 is not None
    assert act_2.trade_id != act_1.trade_id
    assert len(manager.get_active_trades()) == 1


def test_delayed_detector_events_handled(temp_db) -> None:
    """Scenario 11: Delayed detector event correlates cleanly with active trade."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    event_fast = make_label_event(TradeSignalType.BUY, x=200, y=250, timestamp=t0)
    act = manager.process_event(event_fast)
    assert act is not None

    # Slower detector event arrives with earlier timestamp (delayed pipeline delivery)
    t_delayed = t0 - timedelta(seconds=5)
    event_slow = make_box_event(TradeSignalType.BUY, x=180, y=200, entry_y=260, timestamp=t_delayed)
    dup = manager.process_event(event_slow)

    assert dup is None
    assert len(manager.get_active_trades()) == 1


def test_out_of_order_timestamps_handled(temp_db) -> None:
    """Scenario 12: Events with out-of-order timestamps update trade state safely."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    t_future = datetime(2026, 11, 7, 10, 0, 15, tzinfo=timezone.utc)
    t_past = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)

    event_1 = make_label_event(TradeSignalType.BUY, x=200, y=250, timestamp=t_future)
    event_2 = make_label_event(TradeSignalType.BUY, x=200, y=250, timestamp=t_past)

    act = manager.process_event(event_1)
    assert act is not None

    # Out of order older event arrives
    dup = manager.process_event(event_2)
    assert dup is None
    active_trade = manager.get_active_trades()[0]
    # last_seen_at remains the latest timestamp
    assert active_trade.last_seen_at == t_future


def test_missing_entry_price_metadata_does_not_break_correlation(temp_db) -> None:
    """Scenario 13: Empty raw_metadata does not break spatial correlation."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Box event without entry_y in metadata
    event_b = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        confidence=0.9,
        bbox=BoundingBox(200, 200, 100, 100),
        raw_metadata={},
    )
    event_a = make_label_event(TradeSignalType.BUY, x=220, y=240)

    act_1 = manager.process_event(event_b)
    assert act_1 is not None

    act_2 = manager.process_event(event_a)
    assert act_2 is None
    assert len(manager.get_active_trades()) == 1


def test_missing_bounding_box_metadata_handled_gracefully(temp_db) -> None:
    """Scenario 14: Degenerate or zero-size bounding box is handled safely without exceptions."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    zero_box_event = DetectionEvent(
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.8,
        bbox=BoundingBox(0, 0, 0, 0),
        raw_metadata={},
    )

    # Must process without throwing ValueError or ZeroDivisionError
    act = manager.process_event(zero_box_event)
    assert act is not None
    assert act.bbox.width == 0


def test_duplicate_events_after_application_restart(tmp_path: Path) -> None:
    """Scenario 15: Restarting app recovers active state and prevents duplicate alerts."""
    db_file = str(tmp_path / "restart_test.db")
    init_db(db_file)
    trade_repo = TradeRepository(db_file)
    event_repo = EventRepository(db_file)

    # Session 1: Process trade
    mgr1 = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)
    event = make_label_event(TradeSignalType.BUY, x=250, y=300)
    act1 = mgr1.process_event(event)
    assert act1 is not None

    # Session 2: Application restarts, initializes fresh EventManager with recover_on_startup=True
    mgr2 = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        recover_on_startup=True,
    )
    assert len(mgr2.get_active_trades()) == 1

    # Incoming event for same trade must NOT trigger new alert
    act2 = mgr2.process_event(event)
    assert act2 is None, "Persisted active trade must prevent duplicate alert after restart"


def test_database_transaction_failures_handled_safely() -> None:
    """Scenario 16: Database write exceptions do not crash in-memory event management."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.save_trade.side_effect = sqlite3.OperationalError("database is locked")
    mock_trade_repo.get_active_and_baseline_trades.return_value = []

    mock_event_repo = MagicMock(spec=EventRepository)
    mock_event_repo.save_event.side_effect = sqlite3.OperationalError("database is locked")

    manager = EventManager(
        trade_repo=mock_trade_repo,
        event_repo=mock_event_repo,
        baseline_calibration_frames=0,
    )

    event = make_label_event(TradeSignalType.BUY)
    # Must not raise sqlite3.OperationalError
    activated = manager.process_event(event)

    assert activated is not None
    assert activated.state == TradeState.ACTIVE
    assert len(manager.get_active_trades()) == 1


def test_expired_candidates_are_cleaned_up(temp_db) -> None:
    """Scenario 17: Unconfirmed candidates older than candidate_expiry_sec are pruned."""
    trade_repo, event_repo = temp_db
    manager = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        box_confirmation_count=2,  # Requires 2 frames to confirm
        candidate_expiry_sec=10.0,
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    event = make_box_event(TradeSignalType.BUY, timestamp=t0)

    # Observation 1: unconfirmed candidate
    act = manager.process_event(event)
    assert act is None
    assert len(manager._candidates) == 1

    # Advance time past candidate_expiry_sec
    t_expired = t0 + timedelta(seconds=15)
    manager.prune_inactive_trades(t_expired)

    assert len(manager._candidates) == 0, "Expired candidate must be pruned"


def test_detectors_independent_when_other_unavailable(temp_db) -> None:
    """Scenario 18: Either detector functions fully when the other detector produces zero events."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Case A: Detector A active, Detector B silent
    event_a = make_label_event(TradeSignalType.BUY, x=100, y=100)
    act_a = manager.process_event(event_a)
    assert act_a is not None

    # Case B: Detector B active, Detector A silent
    event_b = make_box_event(TradeSignalType.SELL, x=400, y=400)
    act_b = manager.process_event(event_b)
    assert act_b is not None

    assert len(manager.get_active_trades()) == 2


def test_repeated_events_from_same_detector_no_duplicate(temp_db) -> None:
    """Scenario 19: Consecutive identical frames from same detector produce exactly 1 activation."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_box_event(TradeSignalType.BUY, x=300, y=250)

    # First event activates trade
    first_act = manager.process_event(event)
    assert first_act is not None

    # 10 identical subsequent events
    for _ in range(10):
        subsequent_act = manager.process_event(event)
        assert subsequent_act is None

    assert len(manager.get_active_trades()) == 1


def test_false_candidate_failing_temporal_confirmation_does_not_activate(temp_db) -> None:
    """Scenario 20: Transient noise spike failing temporal confirmation count does not activate."""
    trade_repo, event_repo = temp_db
    manager = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        box_confirmation_count=3,  # Requires 3 consecutive observations
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    transient_noise = make_box_event(TradeSignalType.BUY, timestamp=t0)

    # Solitary 1-frame observation
    act = manager.process_event(transient_noise)
    assert act is None
    assert len(manager.get_active_trades()) == 0

    # Advance time and prune without further frames
    t_after = t0 + timedelta(seconds=60)
    manager.prune_inactive_trades(t_after)

    assert len(manager.get_active_trades()) == 0
    assert len(manager._candidates) == 0


# ---------------------------------------------------------------------------
# Additional Correlation & Persistence Safety Tests (Phase 5 Review)
# ---------------------------------------------------------------------------

def test_two_nearby_buy_trades_with_overlapping_boxes(temp_db) -> None:
    """Requirement 1.1: Two nearby BUY trades with overlapping boxes are not merged."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Trade 1: x in [100, 200], entry_y=250
    box1 = make_box_event(TradeSignalType.BUY, x=100, y=200, width=100, height=100, entry_y=250)
    act1 = manager.process_event(box1)
    assert act1 is not None

    # Trade 2: x in [160, 260], entry_y=250 (overlapping from x=160 to 200)
    box2 = make_box_event(TradeSignalType.BUY, x=160, y=200, width=100, height=100, entry_y=250)
    act2 = manager.process_event(box2)
    assert act2 is not None
    assert act1.trade_id != act2.trade_id
    assert len(manager.get_active_trades()) == 2

    # Label for Trade 2 arrives at Trade 2's origin (x=162, y=245)
    # Origin distance to Trade 2 is 2px; origin distance to Trade 1 is 62px (> 40px)
    label2 = make_label_event(TradeSignalType.BUY, x=162, y=245, width=30, height=20)
    dup = manager.process_event(label2)
    assert dup is None

    # Verifies label enriched Trade 2, NOT Trade 1
    t1 = [t for t in manager.get_active_trades() if t.trade_id == act1.trade_id][0]
    t2 = [t for t in manager.get_active_trades() if t.trade_id == act2.trade_id][0]
    assert "LABEL_DETECTOR" not in t1.metadata["correlated_detectors"]
    assert "LABEL_DETECTOR" in t2.metadata["correlated_detectors"]


def test_two_buy_trades_sharing_similar_entry_line(temp_db) -> None:
    """Requirement 1.2: Two BUY trades sharing similar entry_y at different coordinates are not merged."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Both share entry_y=250 and BUY direction, but x=100 vs x=450
    box1 = make_box_event(TradeSignalType.BUY, x=100, y=200, width=100, height=100, entry_y=250)
    box2 = make_box_event(TradeSignalType.BUY, x=450, y=200, width=100, height=100, entry_y=250)

    act1 = manager.process_event(box1)
    act2 = manager.process_event(box2)

    assert act1 is not None
    assert act2 is not None
    assert act1.trade_id != act2.trade_id
    assert len(manager.get_active_trades()) == 2

    # Direction and entry line alone never merge across distant coordinates
    label2 = make_label_event(TradeSignalType.BUY, x=455, y=245)
    dup = manager.process_event(label2)
    assert dup is None
    assert len(manager.get_active_trades()) == 2


def test_two_separate_trades_appearing_nearly_simultaneously(temp_db) -> None:
    """Requirement 1.3: Two separate trades appearing nearly simultaneously activate independently."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    now = datetime.now(timezone.utc)
    event_a = make_label_event(TradeSignalType.BUY, x=100, y=100, timestamp=now)
    event_b = make_box_event(TradeSignalType.BUY, x=500, y=300, entry_y=350, timestamp=now)

    activations = manager.process_events([event_a, event_b])

    assert len(activations) == 2
    assert activations[0].trade_id != activations[1].trade_id
    assert len(manager.get_active_trades()) == 2


def test_matching_ab_events_with_slightly_different_bounding_boxes(temp_db) -> None:
    """Requirement 1.4: Matching A/B events with slight bounding-box jitter correlate cleanly."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Label at (200, 245), Box at (185, 200) with entry_y=255
    event_a = make_label_event(TradeSignalType.BUY, x=200, y=245, width=40, height=20)
    event_b = make_box_event(TradeSignalType.BUY, x=185, y=200, width=120, height=100, entry_y=255)

    act = manager.process_event(event_a)
    assert act is not None

    dup = manager.process_event(event_b)
    assert dup is None
    assert len(manager.get_active_trades()) == 1
    assert "BOX_DETECTOR" in manager.get_active_trades()[0].metadata["correlated_detectors"]


def test_matching_ab_events_with_incomplete_metadata(temp_db) -> None:
    """Requirement 1.5: Matching A/B events with missing entry_y correlate via multi-evidence geometry."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Box event without entry_y in raw_metadata
    box_no_entry = DetectionEvent(
        detector_type=DetectorType.BOX,
        signal_type=TradeSignalType.BUY,
        confidence=0.90,
        bbox=BoundingBox(180, 200, 100, 90),
        raw_metadata={},
    )
    # Label inside box aligned at origin
    label = make_label_event(TradeSignalType.BUY, x=190, y=230, width=40, height=20)

    act = manager.process_event(box_no_entry)
    assert act is not None

    # Multi-evidence (containment + origin alignment) successfully matches even without entry_y
    dup = manager.process_event(label)
    assert dup is None
    assert len(manager.get_active_trades()) == 1


def test_ambiguous_correlation_avoids_silent_merge(temp_db) -> None:
    """Requirement 1.6: Ambiguous correlation between two plausible trades avoids silent merge."""
    trade_repo, event_repo = temp_db
    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Two active trades symmetrically situated
    trade1 = TradeRecord(
        trade_id="trade-1",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.BOX,
        bbox=BoundingBox(100, 200, 100, 100),
        metadata={"signal_type": "BUY", "entry_y": 250},
    )
    trade2 = TradeRecord(
        trade_id="trade-2",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.BOX,
        bbox=BoundingBox(100, 200, 100, 100),
        metadata={"signal_type": "BUY", "entry_y": 250},
    )
    manager._active_trades["trade-1"] = trade1
    manager._active_trades["trade-2"] = trade2

    # Event matching both identically
    ambiguous_event = make_label_event(TradeSignalType.BUY, x=110, y=245)
    matched = manager._find_matching_active_trade(ambiguous_event)

    # Ambiguity check must refuse to silently pick an arbitrary trade
    assert matched is None


def test_failed_activation_persistence_followed_by_retry(temp_db) -> None:
    """Requirement 3.1: Failed activation persistence is recoverable via non-blocking retry."""
    trade_repo, event_repo = temp_db

    # Simulate database lock failure on initial save
    original_save = trade_repo.save_trade
    fail_count = 0

    def mock_failing_save(trade):
        nonlocal fail_count
        fail_count += 1
        raise sqlite3.OperationalError("database is locked")

    trade_repo.save_trade = mock_failing_save

    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    event = make_label_event(TradeSignalType.BUY)
    act = manager.process_event(event)

    # Activation is NOT lost
    assert act is not None
    assert act.state == TradeState.ACTIVE
    # Persistence status is tracked as unpersisted
    assert manager.is_trade_persisted(act.trade_id) is False
    assert act.metadata["persisted"] is False
    assert len(manager.get_unpersisted_trades()) == 1

    # Database lock resolves
    trade_repo.save_trade = original_save

    # Retry persistence
    resolved = manager.retry_pending_persistence()
    assert resolved == 1
    assert manager.is_trade_persisted(act.trade_id) is True
    assert act.metadata["persisted"] is True
    assert len(manager.get_unpersisted_trades()) == 0

    # Trade is confirmed in SQLite
    saved_trade = trade_repo.get_trade(act.trade_id)
    assert saved_trade is not None
    assert saved_trade.trade_id == act.trade_id


def test_repeated_retry_without_duplicate_trade_creation(temp_db) -> None:
    """Requirement 3.2: Repeated retry calls are strictly idempotent and do not duplicate rows."""
    trade_repo, event_repo = temp_db

    manager = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Intentionally queue an unpersisted trade
    event = make_label_event(TradeSignalType.BUY)
    trade = TradeRecord(
        trade_id="retry-trade-1",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        bbox=BoundingBox(100, 100, 40, 20),
        metadata={"signal_type": "BUY", "persisted": False},
    )
    manager._active_trades["retry-trade-1"] = trade
    manager._pending_persist_trades["retry-trade-1"] = (trade, [event])

    # First retry persists successfully
    resolved = manager.retry_pending_persistence()
    assert resolved == 1
    assert manager.is_trade_persisted("retry-trade-1") is True

    # Second retry is a no-op
    resolved_again = manager.retry_pending_persistence()
    assert resolved_again == 0

    # Ensure single row in database
    trades = trade_repo.get_active_and_baseline_trades()
    matching_ids = [t.trade_id for t in trades if t.trade_id == "retry-trade-1"]
    assert len(matching_ids) == 1


def test_application_restart_after_persistence_failure(tmp_path: Path) -> None:
    """Requirement 3.3: App restart only recovers persisted trades; unpersisted trades require retry."""
    db_file = str(tmp_path / "restart_persist_test.db")
    init_db(db_file)
    trade_repo = TradeRepository(db_file)
    event_repo = EventRepository(db_file)

    mgr1 = EventManager(trade_repo=trade_repo, event_repo=event_repo, baseline_calibration_frames=0)

    # Persisted trade
    event_good = make_label_event(TradeSignalType.BUY, x=100, y=100)
    act_good = mgr1.process_event(event_good)
    assert act_good is not None
    assert mgr1.is_trade_persisted(act_good.trade_id) is True

    # Unpersisted trade
    trade_bad = TradeRecord(
        trade_id="unpersisted-trade",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.LABEL,
        bbox=BoundingBox(500, 500, 40, 20),
        metadata={"signal_type": "BUY", "persisted": False},
    )
    mgr1._active_trades["unpersisted-trade"] = trade_bad
    mgr1._pending_persist_trades["unpersisted-trade"] = (trade_bad, [])

    # Restart application
    mgr2 = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        recover_on_startup=True,
    )

    recovered = mgr2.get_active_trades()
    recovered_ids = [t.trade_id for t in recovered]
    assert act_good.trade_id in recovered_ids
    assert "unpersisted-trade" not in recovered_ids


def test_confirmation_policy_one_frame_vs_two_frame_evaluation(temp_db) -> None:
    """Requirement 2: Controlled test comparing 1-frame vs 2-frame confirmation latency & false positives."""
    trade_repo, event_repo = temp_db

    # 1-Frame Confirmation: Immediate activation, but susceptible to 1-frame transient noise
    mgr_1frame = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        label_confirmation_count=1,
        box_confirmation_count=1,
    )

    t0 = datetime(2026, 11, 7, 10, 0, 0, tzinfo=timezone.utc)
    transient_noise = make_label_event(TradeSignalType.BUY, x=200, y=200, timestamp=t0)

    # Under 1-frame policy: fires immediately on frame 1 (latency = 0 frame delay)
    act_1frame = mgr_1frame.process_event(transient_noise)
    assert act_1frame is not None, "1-frame policy activates on single observation"

    # 2-Frame Confirmation: Completely filters single-frame transient noise
    mgr_2frame = EventManager(
        trade_repo=trade_repo,
        event_repo=event_repo,
        baseline_calibration_frames=0,
        label_confirmation_count=2,
        box_confirmation_count=2,
        recover_on_startup=False,
    )

    # Frame 1: Transient noise observed but NOT activated
    act_2frame_f1 = mgr_2frame.process_event(transient_noise)
    assert act_2frame_f1 is None, "2-frame policy must NOT activate on single observation"
    assert len(mgr_2frame._candidates) == 1

    # Frame 2: Confirmed observation triggers activation (latency = 1 frame delay)
    t1 = t0 + timedelta(seconds=1)
    confirmed_noise = make_label_event(TradeSignalType.BUY, x=200, y=200, timestamp=t1)
    act_2frame_f2 = mgr_2frame.process_event(confirmed_noise)
    assert act_2frame_f2 is not None, "2-frame policy activates upon 2nd consecutive observation"


