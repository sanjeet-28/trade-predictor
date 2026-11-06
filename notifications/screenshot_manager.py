"""Screenshot storage and disk retention manager.

Saves captured frame snapshots for trade activations on demand (zero continuous recording)
and enforces bounded disk usage by pruning oldest files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from core.logger import get_logger
from core.models import Frame

logger = get_logger("notifications.screenshot")


class ScreenshotManager:
    """Manages saving and disk pruning of chart snapshots for trade activations."""

    def __init__(
        self,
        screenshot_dir: str = "captures/trades",
        max_stored_screenshots: int = 100,
    ) -> None:
        self.screenshot_dir = Path(screenshot_dir)
        self.max_stored_screenshots = max_stored_screenshots
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    def save_trade_screenshot(
        self,
        frame: Optional[Frame],
        trade_id: str,
    ) -> Optional[str]:
        """Save a frame buffer to disk as a PNG snapshot.

        Returns:
            Saved file path if successful, None if frame or image data is unavailable.
        """
        if frame is None or frame.data is None:
            return None

        try:
            self._prune_old_screenshots()

            filename = f"trade_{trade_id}.png"
            file_path = self.screenshot_dir / filename
            success = cv2.imwrite(str(file_path), frame.data)
            if success:
                logger.info("Saved trade snapshot to: %s", file_path)
                return str(file_path)
            else:
                logger.warning("cv2.imwrite failed to save snapshot: %s", file_path)
                return None
        except Exception as e:
            logger.error("Error saving trade snapshot for %s: %s", trade_id, e)
            return None

    def _prune_old_screenshots(self) -> None:
        """Ensure total stored screenshots on disk do not exceed max_stored_screenshots."""
        try:
            files = list(self.screenshot_dir.glob("trade_*.png"))
            if len(files) >= self.max_stored_screenshots:
                # Sort by last modified time ascending (oldest first)
                files.sort(key=lambda p: p.stat().st_mtime)
                excess_count = len(files) - self.max_stored_screenshots + 1
                for f in files[:excess_count]:
                    try:
                        f.unlink()
                        logger.debug("Pruned old trade snapshot: %s", f)
                    except Exception as e:
                        logger.warning("Failed to delete old snapshot %s: %s", f, e)
        except Exception as e:
            logger.error("Error during screenshot pruning: %s", e)

