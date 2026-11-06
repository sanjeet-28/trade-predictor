"""Central Event Manager & Multi-Detector Deduplication Engine.

Aggregates raw detection events from Detector A (LabelDetector) and Detector B (BoxDetector),
performs multi-evidence cross-detector spatial-temporal correlation, tracks multi-trade lifecycles,
enforces startup baseline calibration, and ensures independent activation without duplicates.
Includes a safe persistence retry queue to prevent lost or duplicate database records.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.config import AppConfig, DetectionConfig
from core.logger import get_logger
from core.models import (
    BoundingBox,
    DetectionEvent,
    DetectorType,
    Frame,
    TradeRecord,
    TradeSignalType,
    TradeState,
)
from storage.repository import EventRepository, TradeRepository

logger = get_logger("core.event_manager")


class EventLifecycleState(str, Enum):
    """Lifecycle states of raw detection candidates."""
    OBSERVED = "OBSERVED"     # Candidate event observed, awaiting temporal confirmation
    CONFIRMED = "CONFIRMED"   # Passed temporal confirmation policy
    ACTIVE = "ACTIVE"         # Activated as a new trade alert
    DUPLICATE = "DUPLICATE"   # Correlated with an existing active trade
    CLOSED = "CLOSED"         # Trade concluded or candidate expired


@dataclass
class CandidateEvent:
    """Represents an unconfirmed detection candidate being evaluated across frames."""
    candidate_id: str
    detector_type: DetectorType
    signal_type: TradeSignalType
    bbox: BoundingBox
    first_seen: datetime
    last_seen: datetime
    observation_count: int = 1
    state: EventLifecycleState = EventLifecycleState.OBSERVED
    raw_event: Optional[DetectionEvent] = None


class EventManager:
    """Central Event Manager coordinating multi-detector trade activations."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        trade_repo: Optional[TradeRepository] = None,
        event_repo: Optional[EventRepository] = None,
        correlation_window_sec: Optional[float] = None,
        label_confirmation_count: Optional[int] = None,
        box_confirmation_count: Optional[int] = None,
        cross_detector_spatial_threshold_px: Optional[float] = None,
        cross_detector_entry_delta_y: Optional[int] = None,
        candidate_expiry_sec: Optional[float] = None,
        active_trade_max_missing_sec: Optional[float] = None,
        baseline_calibration_frames: Optional[int] = None,
        recover_on_startup: Optional[bool] = None,
    ) -> None:
        det_cfg = config.detection if config else DetectionConfig()

        self.correlation_window_sec = (
            correlation_window_sec
            if correlation_window_sec is not None
            else getattr(det_cfg, "correlation_window_sec", 60.0)
        )
        self.label_confirmation_count = (
            label_confirmation_count
            if label_confirmation_count is not None
            else getattr(det_cfg, "label_confirmation_count", 1)
        )
        self.box_confirmation_count = (
            box_confirmation_count
            if box_confirmation_count is not None
            else getattr(det_cfg, "box_confirmation_count", 1)
        )
        self.cross_detector_spatial_threshold_px = (
            cross_detector_spatial_threshold_px
            if cross_detector_spatial_threshold_px is not None
            else getattr(det_cfg, "cross_detector_spatial_threshold_px", 60.0)
        )
        self.cross_detector_entry_delta_y = (
            cross_detector_entry_delta_y
            if cross_detector_entry_delta_y is not None
            else getattr(det_cfg, "cross_detector_entry_delta_y", 25)
        )
        self.candidate_expiry_sec = (
            candidate_expiry_sec
            if candidate_expiry_sec is not None
            else getattr(det_cfg, "candidate_expiry_sec", 30.0)
        )
        self.active_trade_max_missing_sec = (
            active_trade_max_missing_sec
            if active_trade_max_missing_sec is not None
            else getattr(det_cfg, "active_trade_max_missing_sec", 120.0)
        )
        self.baseline_calibration_frames = (
            baseline_calibration_frames
            if baseline_calibration_frames is not None
            else getattr(det_cfg, "baseline_calibration_frames", 10)
        )
        self.recover_on_startup = (
            recover_on_startup
            if recover_on_startup is not None
            else getattr(det_cfg, "recover_on_startup", True)
        )

        self.trade_repo = trade_repo
        self.event_repo = event_repo

        # In-memory tracking state
        self._active_trades: Dict[str, TradeRecord] = {}
        self._candidates: Dict[str, CandidateEvent] = {}
        self._pending_persist_trades: Dict[str, Tuple[TradeRecord, List[DetectionEvent]]] = {}
        self._frames_seen: int = 0
        self.is_calibrated: bool = self.baseline_calibration_frames <= 0

        # Recover persisted trades if configured
        if self.recover_on_startup and self.trade_repo is not None:
            self._recover_state_from_database()

    def reset(self) -> None:
        """Reset internal memory state."""
        self._active_trades.clear()
        self._candidates.clear()
        self._pending_persist_trades.clear()
        self._frames_seen = 0
        self.is_calibrated = self.baseline_calibration_frames <= 0
        logger.debug("EventManager state reset.")

    def advance_frame(self, frame: Optional[Frame] = None) -> None:
        """Advance stream frame counter, update calibration, and retry pending writes."""
        self._frames_seen += 1
        if self._frames_seen >= self.baseline_calibration_frames and not self.is_calibrated:
            self.is_calibrated = True
            logger.info(
                "EventManager startup calibration completed (%d baseline trades cataloged).",
                len(self._active_trades),
            )

        # Retry any pending unpersisted writes from previous database lock
        if self._pending_persist_trades:
            self.retry_pending_persistence()

        current_time = frame.timestamp if frame else datetime.now(timezone.utc)
        self.prune_inactive_trades(current_time)

    def process_event(
        self,
        event: DetectionEvent,
        frame: Optional[Frame] = None,
    ) -> Optional[TradeRecord]:
        """Process a single DetectionEvent from either Detector A or Detector B.

        Returns:
            TradeRecord if this event activates a GENUINELY NEW trade requiring user alert;
            None if the event was deduplicated, suppressed as baseline, or awaiting confirmation.
        """
        results = self.process_events([event], frame=frame)
        return results[0] if results else None

    def process_events(
        self,
        events: List[DetectionEvent],
        frame: Optional[Frame] = None,
    ) -> List[TradeRecord]:
        """Process a batch of events from one or both detectors.

        Returns:
            List of newly activated TradeRecord instances requiring notifications.
        """
        current_time = frame.timestamp if frame else datetime.now(timezone.utc)
        new_activations: List[TradeRecord] = []

        for event in events:
            # 1. Temporal confirmation check
            confirmed_event = self._evaluate_confirmation_policy(event, current_time)
            if confirmed_event is None:
                continue

            # 2. Check for correlation against existing active or baseline trades
            matched_trade = self._find_matching_active_trade(confirmed_event)

            if matched_trade is not None:
                # Correlated as duplicate evidence of an existing trade
                self._enrich_existing_trade(matched_trade, confirmed_event)
            else:
                # Brand-new trade setup detected
                activated = self._register_new_trade(confirmed_event)
                if activated is not None:
                    new_activations.append(activated)

        return new_activations

    def get_active_trades(self) -> List[TradeRecord]:
        """Return list of all currently tracked trades (ACTIVE and BASELINE)."""
        return list(self._active_trades.values())

    def is_trade_persisted(self, trade_id: str) -> bool:
        """Return True if trade has been successfully written to the database."""
        if trade_id in self._pending_persist_trades:
            return False
        trade = self._active_trades.get(trade_id)
        if trade:
            return bool(trade.metadata.get("persisted", False))
        return False

    def get_unpersisted_trades(self) -> List[TradeRecord]:
        """Return all active trades whose persistence to database is pending retry."""
        return [trade for trade, _ in self._pending_persist_trades.values()]

    def retry_pending_persistence(self) -> int:
        """Attempt to persist queued unpersisted trades and events. Returns count resolved."""
        if not self._pending_persist_trades:
            return 0

        resolved_ids: List[str] = []
        for trade_id, (trade, events) in list(self._pending_persist_trades.items()):
            trade_ok = True
            if self.trade_repo:
                try:
                    self.trade_repo.save_trade(trade)
                except Exception as e:
                    logger.error("Retry failed saving trade %s: %s", trade_id, e)
                    trade_ok = False

            if trade_ok and self.event_repo:
                for ev in events:
                    try:
                        self.event_repo.save_event(ev, trade_id=trade_id)
                    except Exception as e:
                        logger.error("Retry failed saving event %s: %s", ev.event_id, e)
                        trade_ok = False
                        break

            if trade_ok:
                trade.metadata["persisted"] = True
                resolved_ids.append(trade_id)
                logger.info("Successfully recovered persistence for trade %s", trade_id)

        for tid in resolved_ids:
            self._pending_persist_trades.pop(tid, None)

        return len(resolved_ids)

    def prune_inactive_trades(self, current_time: Optional[datetime] = None) -> List[str]:
        """Close trades missing beyond timeout and purge expired candidates."""
        now = current_time or datetime.now(timezone.utc)

        # 1. Prune unconfirmed candidates older than candidate_expiry_sec
        expired_cand_ids = [
            cid for cid, cand in self._candidates.items()
            if (now - cand.last_seen).total_seconds() > self.candidate_expiry_sec
        ]
        for cid in expired_cand_ids:
            del self._candidates[cid]

        # 2. Close active trades missing beyond active_trade_max_missing_sec
        closed_trade_ids: List[str] = []
        for tid, trade in list(self._active_trades.items()):
            elapsed_sec = (now - trade.last_seen_at).total_seconds()
            if elapsed_sec > self.active_trade_max_missing_sec:
                trade.state = TradeState.CLOSED
                closed_trade_ids.append(tid)
                del self._active_trades[tid]

                if self.trade_repo:
                    try:
                        self.trade_repo.update_state(tid, TradeState.CLOSED)
                    except Exception as e:
                        logger.error("Failed to update closed trade %s in database: %s", tid, e)

                logger.info(
                    "Trade %s closed after %.1fs of inactivity at %s",
                    tid, elapsed_sec, trade.bbox.to_dict(),
                )

        return closed_trade_ids

    # -----------------------------------------------------------------------
    # Internal Helpers: Confirmation, Matching, and Registration
    # -----------------------------------------------------------------------

    def _evaluate_confirmation_policy(
        self,
        event: DetectionEvent,
        current_time: datetime,
    ) -> Optional[DetectionEvent]:
        """Apply temporal confirmation count to reject single-frame transient noise."""
        required_count = (
            self.label_confirmation_count
            if event.detector_type == DetectorType.LABEL
            else self.box_confirmation_count
        )

        if required_count <= 1:
            return event

        # Match with candidate
        cand_key = self._find_matching_candidate(event)
        if cand_key is not None:
            cand = self._candidates[cand_key]
            cand.observation_count += 1
            cand.last_seen = event.timestamp
            cand.bbox = event.bbox
            if cand.observation_count >= required_count:
                # Confirmed! Remove from candidate pool and return
                del self._candidates[cand_key]
                return event
            return None
        else:
            # Register new candidate
            cand_id = str(uuid.uuid4())
            self._candidates[cand_id] = CandidateEvent(
                candidate_id=cand_id,
                detector_type=event.detector_type,
                signal_type=event.signal_type,
                bbox=event.bbox,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                observation_count=1,
                raw_event=event,
            )
            return None

    def _find_matching_candidate(self, event: DetectionEvent) -> Optional[str]:
        """Locate an existing unconfirmed candidate matching this event."""
        for cid, cand in self._candidates.items():
            if cand.detector_type != event.detector_type:
                continue
            if cand.signal_type != event.signal_type and event.signal_type != TradeSignalType.UNKNOWN:
                continue
            # Check spatial proximity or IoU
            if cand.bbox.iou(event.bbox) >= 0.30 or cand.bbox.distance_to(event.bbox) <= self.cross_detector_spatial_threshold_px:
                return cid
        return None

    def _find_matching_active_trade(self, event: DetectionEvent) -> Optional[TradeRecord]:
        """Correlate event with existing active trades using conservative multi-evidence scoring."""
        candidates: List[Tuple[TradeRecord, int, float]] = []

        for trade in self._active_trades.values():
            if trade.state not in (TradeState.ACTIVE, TradeState.BASELINE):
                continue

            # 1. Direction check: BUY and SELL trades must NEVER be merged under any circumstances
            trade_signal_str = trade.metadata.get("signal_type", TradeSignalType.UNKNOWN.value)
            if (
                event.signal_type != TradeSignalType.UNKNOWN
                and trade_signal_str != TradeSignalType.UNKNOWN.value
                and event.signal_type.value != trade_signal_str
            ):
                continue

            # 2. Temporal correlation window check
            time_delta = abs((event.timestamp - trade.last_seen_at).total_seconds())
            if time_delta > self.active_trade_max_missing_sec:
                continue

            # 3. Evaluate independent pieces of spatial/structural evidence
            is_matched, evidence_count, match_score = self._evaluate_spatial_correlation(trade, event)
            if is_matched:
                candidates.append((trade, evidence_count, match_score))

        if not candidates:
            return None

        if len(candidates) == 1:
            return candidates[0][0]

        # Multiple candidate active trades match
        candidates.sort(key=lambda item: item[2], reverse=True)
        top_trade, top_evidence, top_score = candidates[0]
        second_trade, second_evidence, second_score = candidates[1]

        # Ambiguity check: if difference in match score is negligible and evidence counts equal,
        # avoid silently merging two plausible separate trades into an arbitrary selection
        label_x = event.bbox.x if event.detector_type == DetectorType.LABEL else None
        top_origin_diff = abs(label_x - top_trade.bbox.x) if label_x is not None else 0
        second_origin_diff = abs(label_x - second_trade.bbox.x) if label_x is not None else 0

        # If top trade is clearly aligned at the entry origin while second is not, resolve to top trade
        if label_x is not None and top_origin_diff <= 15 and second_origin_diff >= 35:
            return top_trade

        if (top_score - second_score) < 10.0 and top_evidence == second_evidence:
            logger.warning(
                "Ambiguous correlation detected for event %s between trades %s and %s (scores: %.1f vs %.1f). Refusing to silently merge.",
                event.event_id, top_trade.trade_id, second_trade.trade_id, top_score, second_score,
            )
            return None

        return top_trade

    def _evaluate_spatial_correlation(
        self,
        trade: TradeRecord,
        event: DetectionEvent,
    ) -> Tuple[bool, int, float]:
        """Evaluate spatial and structural correlation evidence between a trade and an event.

        Enforces:
          - Direction alone NEVER establishes a match.
          - At least two independent pieces of spatial/structural evidence are required.
          - Conservative matching when metadata is incomplete.
          - Ambiguity prevention.
        """
        # Direction must not conflict
        trade_signal_str = trade.metadata.get("signal_type", TradeSignalType.UNKNOWN.value)
        if (
            event.signal_type != TradeSignalType.UNKNOWN
            and trade_signal_str != TradeSignalType.UNKNOWN.value
            and event.signal_type.value != trade_signal_str
        ):
            return False, 0, 0.0

        # Handle zero or degenerate bounding boxes
        if trade.bbox.width == 0 or trade.bbox.height == 0 or event.bbox.width == 0 or event.bbox.height == 0:
            return False, 0, 0.0

        evidence_count = 0
        score = 0.0

        # -------------------------------------------------------------------
        # Case A: Same detector source (BOX <-> BOX)
        # -------------------------------------------------------------------
        if trade.detector_source == event.detector_type and event.detector_type == DetectorType.BOX:
            trade_entry = trade.metadata.get("entry_y")
            event_entry = event.raw_metadata.get("entry_y")

            origin_delta = abs(trade.bbox.x - event.bbox.x)
            iou = trade.bbox.iou(event.bbox)

            # Two distinct position boxes on different candles (origin > 35px apart)
            # without substantial IoU overlap (>= 0.50) are separate trades
            if origin_delta > 35 and iou < 0.50:
                return False, 0, 0.0

            # Evidence 1: Entry line match within delta
            if trade_entry is not None and event_entry is not None:
                entry_delta = abs(trade_entry - event_entry)
                if entry_delta <= 12:
                    evidence_count += 1
                    score += 50.0 - entry_delta
                elif entry_delta > 18:
                    # Clear entry line mismatch -> Cannot be the same trade!
                    return False, 0, 0.0

            # Evidence 2: Horizontal origin alignment (box starts at the same entry candle)
            if origin_delta <= 30:
                evidence_count += 1
                score += 30.0 - origin_delta

            # Evidence 3: Significant bounding box IoU
            if iou >= 0.40:
                evidence_count += 1
                score += iou * 40.0

            # Evidence 4: Centroid proximity
            dist = trade.bbox.distance_to(event.bbox)
            if dist <= 45.0:
                evidence_count += 1
                score += max(0.0, 30.0 - dist * 0.5)

            is_matched = evidence_count >= 2
            return is_matched, evidence_count, score

        # -------------------------------------------------------------------
        # Case B: Same detector source (LABEL <-> LABEL)
        # -------------------------------------------------------------------
        if trade.detector_source == event.detector_type and event.detector_type == DetectorType.LABEL:
            # Evidence 1: Tight centroid proximity
            dist = trade.bbox.distance_to(event.bbox)
            if dist <= 35.0:
                evidence_count += 1
                score += 50.0 - dist

            # Evidence 2: Bounding box overlap or high IoU
            iou = trade.bbox.iou(event.bbox)
            if iou >= 0.30:
                evidence_count += 1
                score += iou * 30.0

            # Evidence 3: Horizontal distance
            h_dist = self._horizontal_distance(trade.bbox, event.bbox)
            if h_dist <= 20:
                evidence_count += 1
                score += 20.0 - h_dist

            is_matched = evidence_count >= 2
            return is_matched, evidence_count, score

        # -------------------------------------------------------------------
        # Case C: Cross-detector (LABEL <-> BOX)
        # -------------------------------------------------------------------
        label_bbox = event.bbox if event.detector_type == DetectorType.LABEL else trade.bbox
        box_bbox = event.bbox if event.detector_type == DetectorType.BOX else trade.bbox
        entry_y = (
            event.raw_metadata.get("entry_y")
            if event.detector_type == DetectorType.BOX
            else trade.metadata.get("entry_y")
        )

        cx, cy = label_bbox.centroid

        # Evidence 1: Point containment or significant area intersection
        if box_bbox.contains_point(cx, cy):
            evidence_count += 1
            score += 40.0
        else:
            # Check if at least 50% of the label bounding box area is inside the box
            ix1 = max(label_bbox.x, box_bbox.x)
            iy1 = max(label_bbox.y, box_bbox.y)
            ix2 = min(label_bbox.x2, box_bbox.x2)
            iy2 = min(label_bbox.y2, box_bbox.y2)
            if ix2 > ix1 and iy2 > iy1:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                if label_bbox.area > 0 and (inter_area / label_bbox.area) >= 0.50:
                    evidence_count += 1
                    score += 30.0

        # Evidence 2: Entry line vertical proximity
        if entry_y is not None:
            entry_delta = abs(cy - entry_y)
            if entry_delta <= self.cross_detector_entry_delta_y:
                evidence_count += 1
                score += 40.0 - entry_delta
            elif entry_delta > 35:
                # If entry line is explicitly known and label is > 35px vertically separated,
                # they belong to different price levels / candles!
                return False, 0, 0.0

        # Evidence 3: Horizontal origin / entry candle alignment
        # In TradingView, the label appears right at the entry candle (box.x)
        origin_delta = abs(label_bbox.x - box_bbox.x)
        if origin_delta <= 30:
            evidence_count += 1
            score += 40.0 - origin_delta
        elif abs(cx - box_bbox.x) <= 35:
            evidence_count += 1
            score += 20.0

        # Evidence 4: Centroid proximity within strict threshold
        dist = box_bbox.distance_to(label_bbox)
        if dist <= self.cross_detector_spatial_threshold_px:
            evidence_count += 1
            score += max(0.0, 30.0 - dist * 0.5)

        # Requirement: At least TWO independent pieces of evidence required
        is_matched = evidence_count >= 2
        return is_matched, evidence_count, score

    @staticmethod
    def _horizontal_distance(box1: BoundingBox, box2: BoundingBox) -> int:
        """Calculate minimum horizontal gap between two bounding boxes (0 if overlapping)."""
        overlap_start = max(box1.x, box2.x)
        overlap_end = min(box1.x2, box2.x2)
        if overlap_end >= overlap_start:
            return 0
        return min(abs(box1.x - box2.x2), abs(box2.x - box1.x2))

    def _enrich_existing_trade(self, trade: TradeRecord, event: DetectionEvent) -> None:
        """Update existing trade with newly observed event without triggering alert."""
        trade.last_seen_at = max(trade.last_seen_at, event.timestamp)

        # If incoming event is from BOX and existing trade had LABEL, upgrade bbox
        if event.detector_type == DetectorType.BOX and trade.detector_source == DetectorType.LABEL:
            trade.bbox = event.bbox
            if "entry_y" in event.raw_metadata:
                trade.metadata["entry_y"] = event.raw_metadata["entry_y"]

        # Track correlated detectors
        detectors = trade.metadata.setdefault("correlated_detectors", [])
        if event.detector_type.value not in detectors:
            detectors.append(event.detector_type.value)

        # Record event id
        event_ids = trade.metadata.setdefault("event_ids", [])
        if event.event_id not in event_ids:
            event_ids.append(event.event_id)

        # Persist event and trade update
        self._persist_event_and_trade(event, trade)

        logger.debug(
            "Event %s correlated with existing trade %s (state=%s)",
            event.event_id, trade.trade_id, trade.state.value,
        )

    def _register_new_trade(self, event: DetectionEvent) -> Optional[TradeRecord]:
        """Register a new trade setup, suppressing alerts if during startup baseline calibration."""
        is_in_calibration = (
            self._frames_seen <= self.baseline_calibration_frames
            and not self.is_calibrated
        )

        trade_state = TradeState.BASELINE if is_in_calibration else TradeState.ACTIVE
        notification_sent = is_in_calibration  # True for baseline (no alert needed)

        trade_id = str(uuid.uuid4())
        metadata: Dict[str, Any] = {
            "signal_type": event.signal_type.value,
            "correlated_detectors": [event.detector_type.value],
            "event_ids": [event.event_id],
            "persisted": False,
        }
        if "entry_y" in event.raw_metadata:
            metadata["entry_y"] = event.raw_metadata["entry_y"]

        trade = TradeRecord(
            trade_id=trade_id,
            state=trade_state,
            detector_source=event.detector_type,
            first_detected_at=event.timestamp,
            last_seen_at=event.timestamp,
            bbox=event.bbox,
            notification_sent=notification_sent,
            notification_sent_at=event.timestamp if notification_sent else None,
            metadata=metadata,
        )

        self._active_trades[trade_id] = trade
        self._persist_event_and_trade(event, trade)

        if is_in_calibration:
            logger.info(
                "Baseline trade cataloged at %s (%s, source=%s) - notification suppressed",
                trade.bbox.to_dict(), event.signal_type.value, event.detector_type.value,
            )
            return None
        else:
            logger.info(
                "NEW TRADE ACTIVATED: %s (%s) from source %s at %s",
                trade.trade_id, event.signal_type.value, event.detector_type.value, trade.bbox.to_dict(),
            )
            return trade

    def _persist_event_and_trade(self, event: DetectionEvent, trade: TradeRecord) -> bool:
        """Persist trade record and raw detection event to SQLite repository safely.

        Enqueues failed writes for non-blocking retry so database locks never drop activations.
        """
        success = True
        if self.trade_repo:
            try:
                self.trade_repo.save_trade(trade)
            except Exception as e:
                logger.error("Failed to persist trade %s to database: %s", trade.trade_id, e)
                success = False

        if self.event_repo:
            try:
                self.event_repo.save_event(event, trade_id=trade.trade_id)
            except Exception as e:
                logger.error("Failed to persist event %s to database: %s", event.event_id, e)
                success = False

        trade.metadata["persisted"] = success

        if not success:
            if trade.trade_id not in self._pending_persist_trades:
                self._pending_persist_trades[trade.trade_id] = (trade, [event])
            else:
                self._pending_persist_trades[trade.trade_id][1].append(event)
            logger.warning(
                "Trade %s persistence pending. Enqueued for automatic retry.",
                trade.trade_id,
            )
        else:
            self._pending_persist_trades.pop(trade.trade_id, None)

        return success

    def _recover_state_from_database(self) -> None:
        """Load active and baseline trades from SQLite upon startup."""
        try:
            persisted = self.trade_repo.get_active_and_baseline_trades()
            for tr in persisted:
                tr.metadata["persisted"] = True
                self._active_trades[tr.trade_id] = tr
            if persisted:
                logger.info(
                    "Recovered %d active/baseline trades from SQLite on startup.",
                    len(persisted),
                )
        except Exception as e:
            logger.error("Failed to recover trades from SQLite database: %s", e)
