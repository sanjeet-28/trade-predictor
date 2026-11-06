"""Detector B: Visual trade-position box detection and temporal tracking.

Identifies trade-position structures on the chart (red/green risk-reward zones,
entry lines, and SL/TP regions), tracks box continuity across frames using IoU
and entry-line anchors, and prevents false triggers from candle movement or box resizing.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.logger import get_logger
from core.models import BoundingBox, DetectionEvent, DetectorType, Frame, TradeSignalType

logger = get_logger("detector.box")


@dataclass
class PositionBoxCandidate:
    """Represents a candidate trade position box identified in a single frame."""
    signal_type: TradeSignalType
    bbox: BoundingBox
    entry_y: int
    profit_bbox: BoundingBox
    loss_bbox: BoundingBox
    confidence: float


@dataclass
class TrackedPositionBox:
    """Represents an active trade position box tracked across consecutive frames."""
    box_id: str
    signal_type: TradeSignalType
    bbox: BoundingBox
    entry_y: int
    first_seen: datetime
    last_seen: datetime
    consecutive_frames: int = 1
    missing_frames: int = 0
    is_baseline: bool = False
    event_emitted: bool = False


class BoxDetector:
    """Detector B: Trade-position box and risk-reward geometry detector."""

    def __init__(
        self,
        roi: Optional[Tuple[int, int, int, int]] = (0, 60, 920, 600),  # (x1, y1, x2, y2)
        min_box_area: int = 400,
        min_box_width: int = 35,
        min_box_height: int = 25,
        min_aspect_ratio: float = 0.25,
        max_aspect_ratio: float = 8.0,
        iou_match_threshold: float = 0.40,
        max_entry_delta_y: int = 12,
        confirmation_frames: int = 1,
        baseline_frames: int = 10,
        max_missing_frames: int = 10,
        # Configurable HSV color ranges with sensible defaults for TradingView charts
        green_hsv_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (35, 50, 50),
            (85, 255, 255),
        ),
        red_hsv_ranges: Tuple[
            Tuple[Tuple[int, int, int], Tuple[int, int, int]],
            Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        ] = (
            ((0, 50, 50), (10, 255, 255)),
            ((170, 50, 50), (180, 255, 255)),
        ),
    ) -> None:
        self.roi = roi
        self.min_box_area = min_box_area
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.iou_match_threshold = iou_match_threshold
        self.max_entry_delta_y = max_entry_delta_y
        self.confirmation_frames = confirmation_frames
        self.baseline_frames = baseline_frames
        self.max_missing_frames = max_missing_frames
        self.green_hsv_range = green_hsv_range
        self.red_hsv_ranges = red_hsv_ranges

        self._tracked_boxes: Dict[str, TrackedPositionBox] = {}
        self._frames_seen: int = 0
        self.is_calibrated: bool = False

    def reset(self) -> None:
        """Reset internal tracking state and calibration."""
        self._tracked_boxes.clear()
        self._frames_seen = 0
        self.is_calibrated = False
        logger.debug("BoxDetector state reset.")

    def process_frame(self, frame: Frame) -> List[DetectionEvent]:
        """Process a frame, detect position boxes, and return new trade events.

        Suppresses alerts for boxes present during the initial baseline_frames period.
        """
        if frame.data is None:
            return []

        self._frames_seen += 1
        is_in_calibration = self._frames_seen <= self.baseline_frames

        if self._frames_seen >= self.baseline_frames and not self.is_calibrated:
            self.is_calibrated = True
            logger.info(
                "BoxDetector calibration completed (%d baseline position boxes cataloged). Normal detection active.",
                len(self._tracked_boxes),
            )

        # 1. Crop frame to active chart ROI
        roi_img, offset_x, offset_y = self._crop_roi(frame.data)

        # 2. Segment and extract candidate position boxes
        candidates = self._find_position_boxes(roi_img, offset_x, offset_y)

        # 3. Match candidates against tracked position boxes
        new_events: List[DetectionEvent] = []
        matched_box_ids = set()

        for candidate in candidates:
            matched_id = self._find_matching_tracked_box(candidate)

            if matched_id is not None:
                # Update existing tracked box
                tracked = self._tracked_boxes[matched_id]
                tracked.last_seen = frame.timestamp
                tracked.bbox = candidate.bbox  # Allow box to expand horizontally with candles
                tracked.entry_y = candidate.entry_y
                tracked.consecutive_frames += 1
                tracked.missing_frames = 0
                matched_box_ids.add(matched_id)

                # If confirmation_frames > 1 and not yet emitted
                if (
                    not tracked.event_emitted
                    and not tracked.is_baseline
                    and tracked.consecutive_frames >= self.confirmation_frames
                ):
                    tracked.event_emitted = True
                    event = self._create_detection_event(candidate, frame.timestamp)
                    new_events.append(event)
            else:
                # Brand-new position box detected
                box_id = str(uuid.uuid4())
                is_baseline = is_in_calibration
                tracked = TrackedPositionBox(
                    box_id=box_id,
                    signal_type=candidate.signal_type,
                    bbox=candidate.bbox,
                    entry_y=candidate.entry_y,
                    first_seen=frame.timestamp,
                    last_seen=frame.timestamp,
                    consecutive_frames=1,
                    missing_frames=0,
                    is_baseline=is_baseline,
                    event_emitted=is_baseline,  # Suppress event if cataloged during baseline
                )
                self._tracked_boxes[box_id] = tracked
                matched_box_ids.add(box_id)

                if is_baseline:
                    logger.debug(
                        "Baseline position box cataloged at %s (entry_y=%d, type=%s) - notification suppressed",
                        candidate.bbox.to_dict(),
                        candidate.entry_y,
                        candidate.signal_type.value,
                    )
                elif self.confirmation_frames <= 1:
                    tracked.event_emitted = True
                    event = self._create_detection_event(candidate, frame.timestamp)
                    new_events.append(event)
                    logger.info(
                        "NEW TRADE POSITION BOX DETECTED: %s at %s (entry_y=%d, confidence=%.2f)",
                        candidate.signal_type.value,
                        candidate.bbox.to_dict(),
                        candidate.entry_y,
                        candidate.confidence,
                    )

        # 4. Prune missing/closed position boxes
        self._prune_missing_boxes(matched_box_ids)

        return new_events

    def _create_detection_event(
        self,
        candidate: PositionBoxCandidate,
        timestamp: datetime,
    ) -> DetectionEvent:
        """Create a standardized DetectionEvent for Detector B."""
        return DetectionEvent(
            detector_type=DetectorType.BOX,
            signal_type=candidate.signal_type,
            confidence=candidate.confidence,
            bbox=candidate.bbox,
            timestamp=timestamp,
            raw_metadata={
                "entry_y": candidate.entry_y,
                "profit_bbox": candidate.profit_bbox.to_dict(),
                "loss_bbox": candidate.loss_bbox.to_dict(),
                "frames_seen": self._frames_seen,
            },
        )

    def _crop_roi(self, image: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Crop frame to active chart canvas ROI."""
        h, w = image.shape[:2]
        if self.roi is not None:
            x1, y1, x2, y2 = self.roi
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            return image[y1:y2, x1:x2], x1, y1
        return image, 0, 0

    def _find_position_boxes(
        self,
        roi_img: np.ndarray,
        offset_x: int,
        offset_y: int,
    ) -> List[PositionBoxCandidate]:
        """Identify paired green (profit) and red (loss) structures sharing an entry separator."""
        hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

        # Bipolar color segmentation using configured HSV ranges
        g_lower, g_upper = np.array(self.green_hsv_range[0]), np.array(self.green_hsv_range[1])
        green_mask = cv2.inRange(hsv, g_lower, g_upper)

        r1_lower, r1_upper = np.array(self.red_hsv_ranges[0][0]), np.array(self.red_hsv_ranges[0][1])
        r2_lower, r2_upper = np.array(self.red_hsv_ranges[1][0]), np.array(self.red_hsv_ranges[1][1])
        red_mask = cv2.bitwise_or(
            cv2.inRange(hsv, r1_lower, r1_upper),
            cv2.inRange(hsv, r2_lower, r2_upper),
        )

        # Morphological filtering to bridge dashed lines and clean single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        green_clean = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        red_clean = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

        # Find connected components / contours for both zones
        g_contours, _ = cv2.findContours(green_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r_contours, _ = cv2.findContours(red_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        g_rects = []
        for c in g_contours:
            x, y, w, h = cv2.boundingRect(c)
            # Candlestick filter: ordinary candles have width <= 8px or aspect ratio w/h < 0.20
            if w <= 8 or (w / max(1, h)) < 0.15:
                continue
            # Horizontal line filter: price grid lines have height <= 3px with huge width
            if h <= 3:
                continue
            if w >= 20 and h >= 8 and (w * h) >= 80:
                g_rects.append((x, y, w, h))

        r_rects = []
        for c in r_contours:
            x, y, w, h = cv2.boundingRect(c)
            if w <= 8 or (w / max(1, h)) < 0.15:
                continue
            if h <= 3:
                continue
            if w >= 20 and h >= 8 and (w * h) >= 80:
                r_rects.append((x, y, w, h))

        candidates: List[PositionBoxCandidate] = []
        roi_h, roi_w = roi_img.shape[:2]
        max_allowed_w = int(roi_w * 0.85)
        max_allowed_h = int(roi_h * 0.85)

        # Find vertically aligned green and red pairs sharing an entry boundary
        for gx, gy, gw, gh in g_rects:
            for rx, ry, rw, rh in r_rects:
                # 1. Check width parity and horizontal alignment / overlap
                max_w = max(gw, rw)
                min_w = min(gw, rw)
                if max_w == 0 or (min_w / max_w) < 0.40:
                    continue  # Mismatched zone widths

                h_overlap_start = max(gx, rx)
                h_overlap_end = min(gx + gw, rx + rw)
                h_overlap = max(0, h_overlap_end - h_overlap_start)

                if (h_overlap / max_w) < 0.35:
                    continue  # Inadequate horizontal overlap

                # 2. Check vertical adjacency / ordering
                is_buy = gy < ry
                is_sell = ry < gy

                if not (is_buy or is_sell):
                    continue

                if is_buy:
                    # Profit (green) above Risk (red): BUY position
                    vertical_gap = ry - (gy + gh)
                    # Must be adjacent with entry separator (tolerating small gap <= 80px or overlap <= 25px)
                    if vertical_gap > 85 or vertical_gap < -25:
                        continue
                    entry_y_local = int((gy + gh + ry) / 2)
                    signal = TradeSignalType.BUY
                else:
                    # Risk (red) above Profit (green): SELL position
                    vertical_gap = gy - (ry + rh)
                    if vertical_gap > 85 or vertical_gap < -25:
                        continue
                    entry_y_local = int((ry + rh + gy) / 2)
                    signal = TradeSignalType.SELL

                # 3. Enclosing bounding box of the entire risk/reward setup
                comb_x = min(gx, rx)
                comb_y = min(gy, ry)
                comb_x2 = max(gx + gw, rx + rw)
                comb_y2 = max(gy + gh, ry + rh)
                comb_w = comb_x2 - comb_x
                comb_h = comb_y2 - comb_y

                # 4. Geometric & Aspect Ratio Validation
                if comb_w < self.min_box_width or comb_h < self.min_box_height:
                    continue
                if comb_w > max_allowed_w or comb_h > max_allowed_h:
                    continue
                if (comb_w * comb_h) < self.min_box_area:
                    continue

                aspect_ratio = comb_w / max(1, comb_h)
                if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                    continue

                # 5. Translate to global frame coordinates
                global_bbox = BoundingBox(
                    x=comb_x + offset_x,
                    y=comb_y + offset_y,
                    width=comb_w,
                    height=comb_h,
                )
                global_entry_y = entry_y_local + offset_y
                global_profit_bbox = BoundingBox(x=gx + offset_x, y=gy + offset_y, width=gw, height=gh)
                global_loss_bbox = BoundingBox(x=rx + offset_x, y=ry + offset_y, width=rw, height=rh)

                # Confidence score proportional to horizontal alignment and structure
                confidence = float(min(0.99, 0.70 + 0.25 * (h_overlap / min_w)))

                candidates.append(
                    PositionBoxCandidate(
                        signal_type=signal,
                        bbox=global_bbox,
                        entry_y=global_entry_y,
                        profit_bbox=global_profit_bbox,
                        loss_bbox=global_loss_bbox,
                        confidence=confidence,
                    )
                )

        return self._merge_overlapping_candidates(candidates)

    def _merge_overlapping_candidates(
        self,
        candidates: List[PositionBoxCandidate],
    ) -> List[PositionBoxCandidate]:
        """Merge multi-target sub-boxes (e.g. TP1, TP2, TP3) belonging to the same position."""
        if not candidates:
            return []

        merged: List[PositionBoxCandidate] = []
        for cand in candidates:
            matched = False
            for idx, existing in enumerate(merged):
                if existing.signal_type == cand.signal_type:
                    # If entry line is virtually identical (within max_entry_delta_y)
                    if abs(existing.entry_y - cand.entry_y) <= self.max_entry_delta_y:
                        # Merge the two into a single enclosing box
                        new_x = min(existing.bbox.x, cand.bbox.x)
                        new_y = min(existing.bbox.y, cand.bbox.y)
                        new_x2 = max(existing.bbox.x2, cand.bbox.x2)
                        new_y2 = max(existing.bbox.y2, cand.bbox.y2)
                        merged[idx] = PositionBoxCandidate(
                            signal_type=existing.signal_type,
                            bbox=BoundingBox(new_x, new_y, new_x2 - new_x, new_y2 - new_y),
                            entry_y=int((existing.entry_y + cand.entry_y) / 2),
                            profit_bbox=existing.profit_bbox,
                            loss_bbox=existing.loss_bbox,
                            confidence=max(existing.confidence, cand.confidence),
                        )
                        matched = True
                        break
            if not matched:
                merged.append(cand)

        return merged

    def _find_matching_tracked_box(
        self,
        candidate: PositionBoxCandidate,
    ) -> Optional[str]:
        """Match candidate against tracked position boxes using IoU and entry-line anchors."""
        best_id: Optional[str] = None
        best_score = -1.0

        for box_id, tracked in self._tracked_boxes.items():
            if tracked.signal_type != candidate.signal_type:
                continue

            # 1. Check Entry Line Anchor (Strongest signal for trade identity)
            entry_delta = abs(tracked.entry_y - candidate.entry_y)
            if entry_delta <= self.max_entry_delta_y:
                # Check horizontal overlap
                h_overlap_start = max(tracked.bbox.x, candidate.bbox.x)
                h_overlap_end = min(tracked.bbox.x2, candidate.bbox.x2)
                h_overlap = max(0, h_overlap_end - h_overlap_start)
                min_w = min(tracked.bbox.width, candidate.bbox.width)

                if min_w > 0 and (h_overlap / min_w) >= 0.25:
                    return box_id

            # 2. Check Bounding Box IoU
            iou_score = tracked.bbox.iou(candidate.bbox)
            if iou_score >= self.iou_match_threshold and iou_score > best_score:
                best_score = iou_score
                best_id = box_id

        return best_id

    def _prune_missing_boxes(self, matched_box_ids: set) -> None:
        """Increment missing counts for unseen boxes and prune expired ones."""
        to_delete = []
        for box_id, tracked in self._tracked_boxes.items():
            if box_id not in matched_box_ids:
                tracked.missing_frames += 1
                if tracked.missing_frames > self.max_missing_frames:
                    to_delete.append(box_id)

        for box_id in to_delete:
            del self._tracked_boxes[box_id]
            logger.debug(
                "Pruned closed position box (id=%s) after %d missing frames",
                box_id,
                self.max_missing_frames,
            )
