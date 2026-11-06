"""Unit tests for foundational data models and bounding box geometry."""

from datetime import datetime, timezone
import pytest

from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    TradeRecord,
    TradeSignalType,
    TradeState,
)


def test_bounding_box_geometry() -> None:
    """Verify BoundingBox properties: x2, y2, area, centroid."""
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    assert bbox.x2 == 110
    assert bbox.y2 == 70
    assert bbox.area == 5000
    assert bbox.centroid == (60.0, 45.0)


def test_bounding_box_invalid_dimensions() -> None:
    """Verify negative dimensions raise ValueError."""
    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=-10, height=20)

    with pytest.raises(ValueError):
        BoundingBox(x=0, y=0, width=10, height=-5)


def test_bounding_box_point_containment() -> None:
    """Verify point containment inside BoundingBox."""
    bbox = BoundingBox(x=10, y=10, width=20, height=20)
    assert bbox.contains_point(15, 15) is True
    assert bbox.contains_point(10, 10) is True
    assert bbox.contains_point(30, 30) is True
    assert bbox.contains_point(5, 15) is False
    assert bbox.contains_point(35, 15) is False


def test_bounding_box_distance() -> None:
    """Verify Euclidean centroid distance calculation."""
    b1 = BoundingBox(x=0, y=0, width=20, height=20)  # centroid: (10, 10)
    b2 = BoundingBox(x=30, y=40, width=20, height=20)  # centroid: (40, 50)
    # distance: sqrt((40-10)^2 + (50-10)^2) = sqrt(900 + 1600) = 50.0
    assert b1.distance_to(b2) == pytest.approx(50.0)


def test_bounding_box_iou_disjoint() -> None:
    """Verify IoU is 0.0 for disjoint bounding boxes."""
    b1 = BoundingBox(x=0, y=0, width=10, height=10)
    b2 = BoundingBox(x=20, y=20, width=10, height=10)
    assert b1.iou(b2) == 0.0


def test_bounding_box_iou_identical() -> None:
    """Verify IoU is 1.0 for identical bounding boxes."""
    b1 = BoundingBox(x=50, y=50, width=100, height=100)
    b2 = BoundingBox(x=50, y=50, width=100, height=100)
    assert b1.iou(b2) == pytest.approx(1.0)


def test_bounding_box_iou_partial_overlap() -> None:
    """Verify IoU calculation on partially overlapping boxes."""
    # Box 1: [0, 0] to [10, 10], area = 100
    # Box 2: [5, 0] to [15, 10], area = 100
    # Intersection: [5, 0] to [10, 10], area = 5 * 10 = 50
    # Union: 100 + 100 - 50 = 150
    # IoU = 50 / 150 = 1/3 ~ 0.333333
    b1 = BoundingBox(x=0, y=0, width=10, height=10)
    b2 = BoundingBox(x=5, y=0, width=10, height=10)
    assert b1.iou(b2) == pytest.approx(1.0 / 3.0)


def test_bounding_box_serialization() -> None:
    """Verify dictionary serialization and deserialization."""
    orig = BoundingBox(x=12, y=34, width=56, height=78)
    d = orig.to_dict()
    restored = BoundingBox.from_dict(d)
    assert orig == restored


def test_detection_event_serialization() -> None:
    """Verify DetectionEvent serialization and deserialization."""
    event = DetectionEvent(
        event_id="test-evt-001",
        detector_type=DetectorType.LABEL,
        signal_type=TradeSignalType.BUY,
        confidence=0.95,
        bbox=BoundingBox(10, 20, 30, 40),
        timestamp=datetime.now(timezone.utc),
        snapshot_path="/tmp/test.png",
        raw_metadata={"ocr_text": "BUY"},
    )
    d = event.to_dict()
    restored = DetectionEvent.from_dict(d)
    assert restored.event_id == event.event_id
    assert restored.detector_type == DetectorType.LABEL
    assert restored.signal_type == TradeSignalType.BUY
    assert restored.confidence == 0.95
    assert restored.bbox == event.bbox
    assert restored.raw_metadata == {"ocr_text": "BUY"}


def test_trade_record_serialization() -> None:
    """Verify TradeRecord serialization and deserialization."""
    now = datetime.now(timezone.utc)
    record = TradeRecord(
        trade_id="test-trade-123",
        state=TradeState.ACTIVE,
        detector_source=DetectorType.BOX,
        first_detected_at=now,
        last_seen_at=now,
        bbox=BoundingBox(50, 60, 70, 80),
        notification_sent=True,
        notification_sent_at=now,
        metadata={"entry_price_level": 1.0850},
    )
    d = record.to_dict()
    restored = TradeRecord.from_dict(d)
    assert restored.trade_id == "test-trade-123"
    assert restored.state == TradeState.ACTIVE
    assert restored.detector_source == DetectorType.BOX
    assert restored.notification_sent is True
    assert restored.bbox == record.bbox

