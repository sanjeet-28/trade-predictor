"""Detector A: BUY and SELL visual label detection and temporal tracking.

Extracts candidate text regions using localized optical character recognition (OCR),
filters false-positive chart markings, and maintains frame-to-frame spatial tracking
with startup baseline suppression.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from detectors.ocr_adapter import OCRAdapter, create_ocr_engine

from core.logger import get_logger
from core.models import BoundingBox, DetectionEvent, DetectorType, Frame, TradeSignalType

logger = get_logger("detector.label")


# Regex patterns for robust BUY and SELL recognition with common OCR typo tolerance
BUY_PATTERN = re.compile(r"^(BUY|8UY|BU¥|BU|LONG)$", re.IGNORECASE)
SELL_PATTERN = re.compile(r"^(SELL|5ELL|SELI|SEL1|SEL|SHORT)$", re.IGNORECASE)

# Negative patterns to reject static chart labels (e.g. "Buy Zone", "SellZone")
DISQUALIFYING_PATTERNS = re.compile(
    r"(ZONE|SUPPLY|DEMAND|NEWS|TIME|UTC|PIPS|CAD|USD|XAU|MARKET|PULSE|PRICE|HIGH|LOW|DAILY|RANGE)",
    re.IGNORECASE,
)

# Regex to match HUD status headers and metrics
HUD_STATUS_PATTERN = re.compile(
    r"(POSITION|CURRENT|TREND|BEARISH|BULLISH|15MIN|30MIN|1H|4H)",
    re.IGNORECASE,
)

# Regex to match completed target execution markers (e.g. "TP1", "TP2", "TP3", "HIT")
COMPLETED_MARKER_PATTERN = re.compile(
    r"^(TP[1-3]?|HIT|CLOSED)$",
    re.IGNORECASE,
)


@dataclass
class TrackedLabel:
    """Represents a recognized BUY or SELL label tracked across frames."""
    label_id: str
    signal_type: TradeSignalType
    bbox: BoundingBox
    first_seen: datetime
    last_seen: datetime
    consecutive_frames: int = 1
    missing_frames: int = 0
    is_baseline: bool = False
    event_emitted: bool = False


class LabelDetector:
    """Detector A: Visual trade label detector with spatial tracking."""

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = (0, 50, 1025, 450),  # (x1, y1, x2, y2)
        excluded_rois: Optional[List[Tuple[int, int, int, int]]] = None,
        min_confidence: float = 0.70,
        calibration_min_confidence: float = 0.45,
        calibration_observation_threshold: int = 2,
        max_spatial_distance_px: float = 45.0,
        chart_scroll_max_drift_px: float = 150.0,
        baseline_frames: int = 15,
        max_missing_frames: int = 10,
        ocr_engine: Optional[Any] = None,
        scale_factor: float = 1.0,
    ) -> None:
        self.roi = roi
        self.excluded_rois = excluded_rois if excluded_rois is not None else [(940, 370, 1280, 480)]
        self.min_confidence = min_confidence
        self.calibration_min_confidence = calibration_min_confidence
        self.calibration_observation_threshold = calibration_observation_threshold
        self.max_spatial_distance_px = max_spatial_distance_px
        self.chart_scroll_max_drift_px = chart_scroll_max_drift_px
        self.baseline_frames = baseline_frames
        self.max_missing_frames = max_missing_frames
        self.scale_factor = scale_factor

        # Initialize OCR engine via compatibility adapter
        if ocr_engine is not None:
            if isinstance(ocr_engine, OCRAdapter):
                self._ocr = ocr_engine
            else:
                self._ocr = OCRAdapter(ocr_engine, engine_type="injected_engine")
        else:
            self._ocr = create_ocr_engine(use_cls=False)
            if self._ocr is None:
                logger.warning("No OCR engine available. LabelDetector will require an injected OCR mock.")

        self._tracked_labels: Dict[str, TrackedLabel] = {}
        self._calibration_candidates: Dict[str, Dict[str, Any]] = {}
        self._frames_seen: int = 0
        self.is_calibrated: bool = False

    def reset(self) -> None:
        """Reset tracking state and calibration counter."""
        self._tracked_labels.clear()
        self._calibration_candidates.clear()
        self._frames_seen = 0
        self.is_calibrated = False
        logger.debug("LabelDetector state reset.")

    def process_frame(self, frame: Frame) -> List[DetectionEvent]:
        """Process a video frame, detect BUY/SELL labels, and return new trade events.

        During the initial baseline_frames period, all existing labels are registered
        as baseline and no new events are returned.
        """
        if frame.data is None:
            return []

        self._frames_seen += 1
        is_in_calibration = self._frames_seen <= self.baseline_frames

        if self._frames_seen >= self.baseline_frames and not self.is_calibrated:
            self.is_calibrated = True
            logger.info(
                "LabelDetector calibration completed (%d baseline labels cataloged). Normal detection active.",
                len(self._tracked_labels),
            )

        # 1. Preprocess and crop to ROI
        roi_img, offset_x, offset_y = self._crop_roi(frame.data)

        # 2. Extract Text Candidates via OCR
        candidates = self._extract_label_candidates(
            roi_img, offset_x, offset_y, is_in_calibration=is_in_calibration
        )

        # 3. Match Candidates with Tracked Labels
        new_events: List[DetectionEvent] = []
        matched_label_ids = set()

        for signal_type, bbox, confidence, raw_text in candidates:
            if is_in_calibration:
                # Multi-observation baseline calibration (Refinement 4):
                # An element must be observed across at least calibration_observation_threshold frames
                # before it is cataloged as a confirmed baseline trade.
                matched_cand_id = None
                for cid, cand in self._calibration_candidates.items():
                    if cand["signal_type"] != signal_type:
                        continue
                    dist = cand["bbox"].distance_to(bbox)
                    x_drift = cand["bbox"].x - bbox.x
                    y_delta = abs(cand["bbox"].centroid[1] - bbox.centroid[1])
                    if dist <= self.max_spatial_distance_px or (0.0 <= x_drift <= self.chart_scroll_max_drift_px and y_delta <= 12):
                        matched_cand_id = cid
                        break

                if matched_cand_id is not None:
                    cand = self._calibration_candidates[matched_cand_id]
                    cand["count"] += 1
                    cand["bbox"] = bbox
                    cand["last_seen"] = frame.timestamp
                    if cand["count"] >= self.calibration_observation_threshold and not cand.get("cataloged"):
                        cand["cataloged"] = True
                        label_id = cand["id"]
                        tracked = TrackedLabel(
                            label_id=label_id,
                            signal_type=signal_type,
                            bbox=bbox,
                            first_seen=cand["first_seen"],
                            last_seen=frame.timestamp,
                            consecutive_frames=cand["count"],
                            missing_frames=0,
                            is_baseline=True,
                            event_emitted=True,
                        )
                        self._tracked_labels[label_id] = tracked
                        matched_label_ids.add(label_id)
                        logger.debug(
                            "Baseline label cataloged at %s (%s) after %d observations - notification suppressed",
                            bbox.to_dict(),
                            signal_type.value,
                            cand["count"],
                        )
                    elif cand.get("cataloged"):
                        label_id = cand["id"]
                        if label_id in self._tracked_labels:
                            tracked = self._tracked_labels[label_id]
                            tracked.bbox = bbox
                            tracked.last_seen = frame.timestamp
                            tracked.consecutive_frames += 1
                            tracked.missing_frames = 0
                        matched_label_ids.add(label_id)
                else:
                    cand_id = str(uuid.uuid4())
                    self._calibration_candidates[cand_id] = {
                        "id": cand_id,
                        "signal_type": signal_type,
                        "bbox": bbox,
                        "first_seen": frame.timestamp,
                        "last_seen": frame.timestamp,
                        "count": 1,
                        "cataloged": False,
                    }
                continue

            # Post-calibration normal detection:
            matched_id = self._find_matching_tracked_label(signal_type, bbox)

            if matched_id is not None:
                # Update existing tracked label
                tracked = self._tracked_labels[matched_id]
                tracked.last_seen = frame.timestamp
                tracked.bbox = bbox
                tracked.consecutive_frames += 1
                tracked.missing_frames = 0
                matched_label_ids.add(matched_id)
            else:
                # Discovered a brand-new label appearance
                label_id = str(uuid.uuid4())
                tracked = TrackedLabel(
                    label_id=label_id,
                    signal_type=signal_type,
                    bbox=bbox,
                    first_seen=frame.timestamp,
                    last_seen=frame.timestamp,
                    consecutive_frames=1,
                    missing_frames=0,
                    is_baseline=False,
                    event_emitted=True,
                )
                self._tracked_labels[label_id] = tracked
                matched_label_ids.add(label_id)

                # Emit new detection event
                event = DetectionEvent(
                    detector_type=DetectorType.LABEL,
                    signal_type=signal_type,
                    confidence=confidence,
                    bbox=bbox,
                    timestamp=frame.timestamp,
                    raw_metadata={"raw_text": raw_text, "frames_seen": self._frames_seen},
                )
                new_events.append(event)
                logger.info(
                    "NEW TRADE LABEL DETECTED: %s at %s (confidence=%.2f, text='%s')",
                    signal_type.value,
                    bbox.to_dict(),
                    confidence,
                    raw_text,
                )

        # 4. Handle Disappeared / Missing Labels
        self._prune_missing_labels(matched_label_ids)

        return new_events

        # 4. Handle Disappeared / Missing Labels
        self._prune_missing_labels(matched_label_ids)

        return new_events

    def _crop_roi(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Crop frame to region of interest (ROI) and apply optional contrast scaling."""
        h, w = image.shape[:2]
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            cropped = image[y1:y2, x1:x2]
            offset_x, offset_y = x1, y1
        else:
            cropped = image
            offset_x, offset_y = 0, 0

        if self.scale_factor != 1.0:
            cropped = cv2.resize(
                cropped,
                (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        return cropped, offset_x, offset_y

    def _extract_label_candidates(
        self,
        roi_img: np.ndarray,
        offset_x: int,
        offset_y: int,
        is_in_calibration: bool = False,
    ) -> List[Tuple[TradeSignalType, BoundingBox, float, str]]:
        """Run OCR on ROI image and return classified BUY/SELL candidates with global coordinates."""
        if self._ocr is None:
            return []

        try:
            ocr_results, _ = self._ocr(roi_img)
        except Exception as e:
            logger.error("OCR execution failed: %s", e)
            return []

        if not ocr_results:
            return []

        # First collect all detected text boxes with their coordinates in global frame
        all_boxes: List[Tuple[str, BoundingBox, float]] = []
        for item in ocr_results:
            box_points, raw_text, confidence = item[0], item[1], float(item[2])
            min_x = min(p[0] for p in box_points)
            min_y = min(p[1] for p in box_points)
            max_x = max(p[0] for p in box_points)
            max_y = max(p[1] for p in box_points)
            if self.scale_factor != 1.0:
                min_x /= self.scale_factor
                min_y /= self.scale_factor
                max_x /= self.scale_factor
                max_y /= self.scale_factor
            gx = int(round(min_x + offset_x))
            gy = int(round(min_y + offset_y))
            gw = int(round(max_x - min_x))
            gh = int(round(max_y - min_y))
            all_boxes.append((raw_text, BoundingBox(gx, gy, max(1, gw), max(1, gh)), confidence))

        candidates: List[Tuple[TradeSignalType, BoundingBox, float, str]] = []
        target_thresh = self.calibration_min_confidence if is_in_calibration else self.min_confidence

        for raw_text, bbox, confidence in all_boxes:
            if confidence < target_thresh:
                continue

            # Classify string into BUY or SELL
            signal = self._classify_text(raw_text)
            if signal is None:
                continue

            cx, cy = bbox.centroid

            # 1. Static HUD Exclusion Zone Check (Refinement 1)
            in_excluded = False
            for ex1, ey1, ex2, ey2 in self.excluded_rois:
                if ex1 <= cx <= ex2 and ey1 <= cy <= ey2:
                    in_excluded = True
                    break
            if in_excluded:
                logger.debug(
                    "Label candidate '%s' at %s rejected by static HUD exclusion",
                    raw_text,
                    bbox.to_dict(),
                )
                continue

            # 2. Contextual Status HUD Keyword Check (Refinement 1)
            is_hud_element = False
            for other_text, other_bbox, _ in all_boxes:
                if other_bbox == bbox:
                    continue
                dx = abs(other_bbox.centroid[0] - cx)
                dy = abs(other_bbox.centroid[1] - cy)
                if dx <= 90 and dy <= 30:
                    if HUD_STATUS_PATTERN.search(other_text):
                        is_hud_element = True
                        break
            if is_hud_element:
                logger.debug(
                    "Label candidate '%s' at %s rejected: adjacent to HUD status text",
                    raw_text,
                    bbox.to_dict(),
                )
                continue

            # 3. Contextual Completed Target Milestone Marker Check (Refinement 2)
            # Planned targets (e.g. "TP: 1.0850", "SL: 1.0750") on a new trade are NOT rejected.
            # Only isolated milestone stamps (e.g. strictly "TP1", "TP2", "HIT") on a completed historical candle are rejected.
            has_completed_marker = False
            for other_text, other_bbox, _ in all_boxes:
                if other_bbox == bbox:
                    continue
                dx = abs(other_bbox.centroid[0] - cx)
                dy = abs(other_bbox.centroid[1] - cy)
                if dx <= 65 and dy <= 20:
                    clean_other = other_text.strip()
                    if COMPLETED_MARKER_PATTERN.match(clean_other):
                        has_completed_marker = True
                        break
            if has_completed_marker:
                logger.debug(
                    "Label candidate '%s' at %s rejected: adjacent to completed target milestone",
                    raw_text,
                    bbox.to_dict(),
                )
                continue

            candidates.append((signal, bbox, confidence, raw_text))

        return candidates

    def _classify_text(self, text: str) -> Optional[TradeSignalType]:
        """Classify OCR text into BUY or SELL, rejecting false-positive chart zones."""
        clean = text.strip()

        # Reject phrases containing disqualifying words like "Buy Zone" or "SellZone"
        if DISQUALIFYING_PATTERNS.search(clean):
            return None

        # Check for clean tokens
        tokens = re.split(r"[\s\:\-\(\)]+", clean)
        for token in tokens:
            if not token:
                continue
            if BUY_PATTERN.match(token):
                return TradeSignalType.BUY
            if SELL_PATTERN.match(token):
                return TradeSignalType.SELL

        return None

    def _find_matching_tracked_label(
        self,
        signal_type: TradeSignalType,
        bbox: BoundingBox,
    ) -> Optional[str]:
        """Find an existing tracked label matching the signal type and spatial position."""
        best_id: Optional[str] = None
        min_distance = float("inf")

        for label_id, tracked in self._tracked_labels.items():
            if tracked.signal_type != signal_type:
                continue

            # Check IoU first (if significant overlap exists)
            if tracked.bbox.iou(bbox) >= 0.35:
                return label_id

            # Check Euclidean distance between centroids
            dist = tracked.bbox.distance_to(bbox)
            if dist <= self.max_spatial_distance_px and dist < min_distance:
                min_distance = dist
                best_id = label_id

            # Check directional leftward chart scroll drift (Refinements 2 & 3)
            # Candlesticks drift leftward: 0 <= (tracked.bbox.x - bbox.x) <= chart_scroll_max_drift_px
            # Vertical alignment: price level within 12px
            x_drift = tracked.bbox.x - bbox.x
            y_delta = abs(tracked.bbox.centroid[1] - bbox.centroid[1])
            if 0.0 <= x_drift <= self.chart_scroll_max_drift_px and y_delta <= 12:
                if dist < min_distance:
                    min_distance = dist
                    best_id = label_id

        return best_id

    def _prune_missing_labels(self, matched_label_ids: set) -> None:
        """Increment missing frame counts for unseen labels and remove expired non-baseline ones."""
        to_delete = []
        for label_id, tracked in self._tracked_labels.items():
            if label_id not in matched_label_ids:
                tracked.missing_frames += 1
                # Baseline elements must NEVER be pruned due to temporary OCR misses
                if not tracked.is_baseline and tracked.missing_frames > self.max_missing_frames:
                    to_delete.append(label_id)

        for label_id in to_delete:
            del self._tracked_labels[label_id]
            logger.debug("Pruned expired tracked label (id=%s) after %d missing frames", label_id, self.max_missing_frames)
