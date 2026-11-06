"""YouTube live stream URL resolver using yt-dlp.

Extracts direct HLS manifests for live streams with format selection,
token expiration detection, automatic cache refresh, and exponential backoff.
"""

from __future__ import annotations

import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yt_dlp

from core.logger import get_logger

logger = get_logger("stream.resolver")


class StreamResolverError(Exception):
    """Base exception for all stream resolution failures."""


class StreamUnavailableError(StreamResolverError):
    """Raised when the video URL is invalid, private, copyright-blocked, or deleted."""


class StreamOfflineError(StreamResolverError):
    """Raised when the target livestream is offline, scheduled, or has ended."""


@dataclass
class StreamResolution:
    """Represents a resolved direct video stream manifest."""
    stream_url: str
    width: int
    height: int
    format_id: str
    resolved_at: datetime
    expires_at: Optional[datetime] = None
    is_live: bool = True

    def is_expired(self, buffer_seconds: float = 120.0) -> bool:
        """Check if the manifest URL has expired or is nearing expiration."""
        if self.expires_at is None:
            # Fallback default expiration TTL (2 hours) if no timestamp present in URL
            age = (datetime.now(timezone.utc) - self.resolved_at).total_seconds()
            return age >= 7200.0 - buffer_seconds

        now = datetime.now(timezone.utc)
        return (self.expires_at - now).total_seconds() <= buffer_seconds


class StreamResolver:
    """Extracts and manages direct streaming URLs from YouTube."""

    def __init__(
        self,
        primary_url: str,
        channel_url: Optional[str] = None,
        preferred_height: int = 720,
        max_retries: int = 4,
        initial_backoff_sec: float = 2.0,
        backoff_multiplier: float = 2.0,
    ) -> None:
        self.primary_url = primary_url
        self.channel_url = channel_url
        self.preferred_height = preferred_height
        self.max_retries = max_retries
        self.initial_backoff_sec = initial_backoff_sec
        self.backoff_multiplier = backoff_multiplier

        self._cached_resolution: Optional[StreamResolution] = None

    def resolve(self, force_refresh: bool = False) -> StreamResolution:
        """Resolve the active livestream URL with caching and retry logic."""
        if not force_refresh and self._cached_resolution is not None:
            if not self._cached_resolution.is_expired():
                logger.debug("Returning valid cached stream resolution (format=%s)", self._cached_resolution.format_id)
                return self._cached_resolution
            logger.info("Cached stream URL is nearing expiration; refreshing from source.")

        attempt = 0
        backoff = self.initial_backoff_sec
        last_exception: Optional[Exception] = None

        while attempt < self.max_retries:
            attempt += 1
            try:
                resolution = self._extract_stream_manifest(self.primary_url)
                self._cached_resolution = resolution
                logger.info(
                    "Stream resolved successfully on attempt %d: format=%s, resolution=%dx%d",
                    attempt,
                    resolution.format_id,
                    resolution.width,
                    resolution.height,
                )
                return resolution
            except StreamUnavailableError:
                # Permanent failure - no point in blind retries without channel fallback
                logger.error("Primary stream URL '%s' is unavailable or invalid.", self.primary_url)
                raise
            except StreamOfflineError:
                # Stream is not live right now
                logger.warning("Stream at '%s' is offline or ended.", self.primary_url)
                raise
            except Exception as e:
                last_exception = e
                logger.warning(
                    "Stream resolution attempt %d/%d failed: %s (%s). Retrying in %.1fs...",
                    attempt,
                    self.max_retries,
                    type(e).__name__,
                    e,
                    backoff,
                )
                if attempt < self.max_retries:
                    time.sleep(backoff)
                    backoff *= self.backoff_multiplier

        raise StreamResolverError(
            f"Failed to resolve stream after {self.max_retries} attempts: {last_exception}"
        ) from last_exception

    def _extract_stream_manifest(self, url: str) -> StreamResolution:
        """Execute yt-dlp info extraction and select optimal video stream."""
        ydl_opts: Dict[str, Any] = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "skip_download": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False, process=False)
        except yt_dlp.utils.DownloadError as e:
            msg = str(e).lower()
            if "private" in msg or "unavailable" in msg or "does not exist" in msg or "not available" in msg:
                raise StreamUnavailableError(f"Stream unavailable: {e}") from e
            raise StreamResolverError(f"Network error extracting stream info: {e}") from e
        except Exception as e:
            raise StreamResolverError(f"Unexpected yt-dlp extraction error: {e}") from e

        if not info:
            raise StreamUnavailableError(f"No stream metadata returned for {url}")

        # Check live status
        is_live = bool(info.get("is_live", False))
        live_status = info.get("live_status", "")
        was_live = bool(info.get("was_live", False))

        if was_live or live_status in ("was_live", "post_live", "not_live"):
            raise StreamOfflineError(f"Stream has already ended (status: {live_status})")

        formats = info.get("formats", [])
        if not formats:
            raise StreamUnavailableError("No formats available for the requested video.")

        # Select candidate format matching preferred height
        selected_format = self._select_optimal_format(formats)
        stream_url = selected_format.get("url")
        if not stream_url:
            raise StreamUnavailableError("Selected format does not provide a valid streaming URL.")

        expires_at = self._parse_expiration(stream_url)

        return StreamResolution(
            stream_url=stream_url,
            width=int(selected_format.get("width") or 1280),
            height=int(selected_format.get("height") or 720),
            format_id=str(selected_format.get("format_id", "unknown")),
            resolved_at=datetime.now(timezone.utc),
            expires_at=expires_at,
            is_live=is_live or (live_status == "is_live"),
        )

    def _select_optimal_format(self, formats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select HLS format nearest to preferred_height (default: 720p)."""
        # Filter formats that have a direct URL
        valid_formats = [f for f in formats if f.get("url")]
        if not valid_formats:
            raise StreamUnavailableError("No valid formats containing a URL were found.")

        # Filter formats with resolution information
        video_formats = [f for f in valid_formats if f.get("height")]

        if not video_formats:
            # Fallback to the last available format
            return valid_formats[-1]

        # Prioritize exact match of preferred height
        exact_match = [f for f in video_formats if f.get("height") == self.preferred_height]
        if exact_match:
            return exact_match[0]

        # Otherwise find format with closest height without exceeding preferred if possible
        under_preferred = [f for f in video_formats if f.get("height", 0) <= self.preferred_height]
        if under_preferred:
            return max(under_preferred, key=lambda f: f.get("height", 0))

        # Fallback to smallest available format to conserve bandwidth
        return min(video_formats, key=lambda f: f.get("height", 9999))

    def _parse_expiration(self, url: str) -> Optional[datetime]:
        """Extract 'expire' UNIX timestamp from google video HLS URLs if present."""
        try:
            parsed = urllib.parse.urlparse(url)
            # 1. Check path parameters (/expire/1794019821/)
            parts = parsed.path.split("/")
            if "expire" in parts:
                idx = parts.index("expire")
                if idx + 1 < len(parts):
                    timestamp_val = int(parts[idx + 1])
                    return datetime.fromtimestamp(timestamp_val, tz=timezone.utc)

            # 2. Check query parameters (?expire=1794019821)
            query_params = urllib.parse.parse_qs(parsed.query)
            if "expire" in query_params:
                timestamp_val = int(query_params["expire"][0])
                return datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
        except Exception as e:
            logger.debug("Could not parse expiration timestamp from URL: %s", e)

        return None

