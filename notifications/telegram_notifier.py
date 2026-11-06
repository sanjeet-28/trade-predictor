"""Telegram Bot API notification client and persistent outbox processor.

Delivers mandatory 'TRADE IS ACTIVE!' alerts with optional chart snapshots,
enforces duplicate prevention through SQLite outbox transactions, and handles
temporary network/rate-limit failures with bounded exponential backoff.
"""

from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.config import AppConfig, TelegramConfig
from core.logger import get_logger
from core.models import (
    Frame,
    NotificationRecord,
    NotificationStatus,
    TradeRecord,
)
from notifications.screenshot_manager import ScreenshotManager
from storage.repository import NotificationRepository, TradeRepository

logger = get_logger("notifications.telegram")

# Mandatory message text strictly enforced by project specifications
MANDATORY_NOTIFICATION_TEXT = "TRADE IS ACTIVE!"


def mask_token(token: str) -> str:
    """Mask Telegram bot token to prevent leaking sensitive secrets in logs."""
    if not token or len(token) < 8:
        return "***"
    return f"{token[:6]}***{token[-4:]}"


def sanitize_url(url: str, token: str) -> str:
    """Replace raw bot token in URLs with [REDACTED]."""
    if token and token in url:
        return url.replace(token, "[REDACTED_TOKEN]")
    return url


class TelegramAPIError(Exception):
    """Raised when Telegram Bot API returns an error response."""

    def __init__(self, message: str, error_code: int = 500, retry_after: Optional[int] = None) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.retry_after = retry_after
        self.is_permanent = error_code in (400, 401, 403, 404)


class TelegramNotifier:
    """Delivers trade alerts to Telegram with persistent outbox deduplication."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        notification_repo: Optional[NotificationRepository] = None,
        trade_repo: Optional[TradeRepository] = None,
        screenshot_manager: Optional[ScreenshotManager] = None,
        http_client: Optional[Callable[[urllib.request.Request, float], bytes]] = None,
        dry_run: bool = False,
    ) -> None:
        t_cfg = config.telegram if config else TelegramConfig()

        self.bot_token = t_cfg.bot_token
        self.chat_id = t_cfg.chat_id
        self.health_chat_id = t_cfg.health_chat_id or t_cfg.chat_id
        self.trade_notification_text = MANDATORY_NOTIFICATION_TEXT
        self.attach_snapshot = t_cfg.attach_snapshot
        self.request_timeout_sec = t_cfg.request_timeout_sec
        self.initial_retry_backoff_sec = t_cfg.initial_retry_backoff_sec
        self.max_retry_backoff_sec = t_cfg.max_retry_backoff_sec
        self.max_retries = t_cfg.max_retries
        self.dry_run = dry_run

        self.notification_repo = notification_repo
        self.trade_repo = trade_repo
        self.screenshot_manager = screenshot_manager or ScreenshotManager(
            screenshot_dir=t_cfg.screenshot_dir,
            max_stored_screenshots=t_cfg.max_stored_screenshots,
        )
        self._http_client = http_client

        masked = mask_token(self.bot_token)
        if self.dry_run:
            logger.info("TelegramNotifier running in DRY-RUN mode. External network requests are disabled.")
        elif not self.bot_token:
            logger.warning("Telegram bot token is not configured. Live alerts will be queued in outbox.")
        else:
            logger.info("TelegramNotifier initialized (bot_token=%s, chat_id=%s)", masked, self.chat_id)

    def recover_pending_notifications(self, reset_interrupted: bool = True) -> int:
        """Recover pending and interrupted notifications from outbox on startup."""
        if not self.notification_repo:
            return 0
        if reset_interrupted:
            self.notification_repo.recover_interrupted_sending(stale_threshold_sec=0.0)
        pending = self.notification_repo.get_pending_notifications()
        if pending:
            logger.info("Recovered %d pending outbox notifications ready for delivery.", len(pending))
        return len(pending)

    # -----------------------------------------------------------------------
    # Outbox Queueing & Delivery Orchestration
    # -----------------------------------------------------------------------

    def queue_trade_notification(
        self,
        trade: TradeRecord,
        frame: Optional[Frame] = None,
    ) -> Optional[NotificationRecord]:
        """Enqueue a new trade notification into the persistent outbox.

        Enforces:
          1. Trade activation must be successfully persisted to database.
          2. Exactly one notification record per trade_id (unique SQLite constraint).
          3. Message text is strictly 'TRADE IS ACTIVE!'.
          4. Optional screenshot attachment without continuous disk recording.
        """
        # 1. Verify that the trade activation has been successfully persisted
        if not trade.metadata.get("persisted", False):
            logger.warning(
                "Trade %s is not yet persisted to database; refusing to queue Telegram notification.",
                trade.trade_id,
            )
            return None

        if not self.notification_repo:
            logger.warning("NotificationRepository not available; cannot persist outbox record.")
            return None

        # 2. Check if a notification already exists for this trade
        existing = self.notification_repo.get_notification_by_trade_id(trade.trade_id)
        if existing is not None:
            logger.debug(
                "Notification already exists for trade %s (status=%s). Skipping duplicate queueing.",
                trade.trade_id, existing.status.value,
            )
            return None

        # 3. Optional screenshot saving
        photo_path = None
        if self.attach_snapshot and frame is not None:
            photo_path = self.screenshot_manager.save_trade_screenshot(frame, trade.trade_id)

        # 4. Construct persistent outbox record
        record = NotificationRecord(
            notification_id=str(uuid.uuid4()),
            trade_id=trade.trade_id,
            chat_id=self.chat_id,
            message_text=self.trade_notification_text,
            photo_path=photo_path,
            status=NotificationStatus.PENDING,
            retry_count=0,
            max_retries=self.max_retries,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        inserted = self.notification_repo.create_notification(record)
        if inserted:
            logger.info(
                "Queued trade notification %s for trade %s (photo=%s)",
                record.notification_id, trade.trade_id, photo_path is not None,
            )
            return record
        else:
            logger.debug("Duplicate notification insertion prevented by SQLite constraint for trade %s", trade.trade_id)
            return None

    def dispatch_pending_notifications(
        self,
        current_time: Optional[datetime] = None,
    ) -> int:
        """Process and send all due pending outbox notifications with backoff retry.

        Returns:
            Count of notifications successfully delivered.
        """
        if not self.notification_repo:
            return 0

        now = current_time or datetime.now(timezone.utc)
        pending = self.notification_repo.get_pending_notifications(now)
        if not pending:
            return 0

        delivered_count = 0
        for record in pending:
            # Handle dry-run / simulation mode: guarantee zero external network requests
            if self.dry_run:
                sent_at = datetime.now(timezone.utc)
                self.notification_repo.mark_sent(record.notification_id, sent_at)
                if self.trade_repo:
                    self.trade_repo.mark_notified(record.trade_id, sent_at)
                delivered_count += 1
                logger.info(
                    "[DRY-RUN] Simulated delivery for trade %s (notification_id=%s, photo=%s)",
                    record.trade_id,
                    record.notification_id,
                    record.photo_path is not None,
                )
                continue

            # Check credentials
            if not self.bot_token or not record.chat_id:
                err_msg = "Telegram bot_token or chat_id is missing."
                logger.error(err_msg)
                self.notification_repo.mark_failed(
                    record.notification_id,
                    error_message=err_msg,
                    permanent=True,
                )
                continue

            # Mark SENDING
            self.notification_repo.mark_sending(record.notification_id)

            try:
                # Attempt photo delivery if screenshot exists
                if record.photo_path and Path(record.photo_path).exists():
                    try:
                        self.send_photo(
                            chat_id=record.chat_id,
                            photo_path=record.photo_path,
                            caption=record.message_text,
                        )
                    except Exception as photo_err:
                        logger.warning(
                            "Photo upload failed for %s (%s); falling back to text notification.",
                            record.notification_id, photo_err,
                        )
                        # Fallback to plain text message
                        self.send_message(
                            chat_id=record.chat_id,
                            text=record.message_text,
                        )
                else:
                    # Text-only notification
                    self.send_message(
                        chat_id=record.chat_id,
                        text=record.message_text,
                    )

                # Confirm delivery
                sent_at = datetime.now(timezone.utc)
                self.notification_repo.mark_sent(record.notification_id, sent_at)

                if self.trade_repo:
                    self.trade_repo.mark_notified(record.trade_id, sent_at)

                delivered_count += 1
                logger.info(
                    "Telegram notification %s successfully sent for trade %s",
                    record.notification_id, record.trade_id,
                )

            except TelegramAPIError as api_err:
                logger.error(
                    "Telegram API rejected notification %s (code=%d, permanent=%s): %s",
                    record.notification_id, api_err.error_code, api_err.is_permanent, api_err,
                )
                if api_err.is_permanent:
                    self.notification_repo.mark_failed(
                        record.notification_id,
                        error_message=str(api_err),
                        permanent=True,
                    )
                else:
                    # Rate limit or temporary Telegram 5xx
                    backoff = api_err.retry_after or self._calculate_backoff(record.retry_count)
                    next_retry = now + timedelta(seconds=backoff)
                    self.notification_repo.mark_failed(
                        record.notification_id,
                        error_message=str(api_err),
                        permanent=False,
                        next_retry_at=next_retry,
                    )

            except Exception as net_err:
                logger.error(
                    "Network error sending notification %s: %s",
                    record.notification_id, net_err,
                )
                # Check retry limit
                if record.retry_count + 1 >= record.max_retries:
                    self.notification_repo.mark_failed(
                        record.notification_id,
                        error_message=f"Exhausted {record.max_retries} retries: {net_err}",
                        permanent=True,
                    )
                else:
                    backoff = self._calculate_backoff(record.retry_count)
                    next_retry = now + timedelta(seconds=backoff)
                    self.notification_repo.mark_failed(
                        record.notification_id,
                        error_message=str(net_err),
                        permanent=False,
                        next_retry_at=next_retry,
                    )

        return delivered_count

    # -----------------------------------------------------------------------
    # Low-Level Telegram HTTP Client
    # -----------------------------------------------------------------------

    def send_message(self, chat_id: str, text: str) -> Dict[str, Any]:
        """Send a text message via Telegram sendMessage."""
        if not self.bot_token:
            raise ValueError("Telegram bot_token is not configured.")
        if not chat_id:
            raise ValueError("Telegram chat_id is not configured.")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = json.dumps({"chat_id": chat_id, "text": text}).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ForexTradeDetector/1.0",
        }

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
        return self._execute_request(req)

    def send_photo(
        self,
        chat_id: str,
        photo_path: str,
        caption: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload and send a photo via Telegram sendPhoto."""
        if not self.bot_token:
            raise ValueError("Telegram bot_token is not configured.")
        if not chat_id:
            raise ValueError("Telegram chat_id is not configured.")

        p_path = Path(photo_path)
        if not p_path.exists() or not p_path.is_file():
            raise FileNotFoundError(f"Screenshot file not found: {photo_path}")

        url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"

        fields = {"chat_id": str(chat_id)}
        if caption:
            fields["caption"] = caption

        data_bytes, content_type = self._build_multipart_payload(
            fields=fields,
            file_field_name="photo",
            file_path=p_path,
            boundary=boundary,
        )

        headers = {
            "Content-Type": content_type,
            "User-Agent": "ForexTradeDetector/1.0",
        }

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
        return self._execute_request(req)

    def _execute_request(self, req: urllib.request.Request) -> Dict[str, Any]:
        """Execute HTTP request with SSL, timeouts, and error classification."""
        sanitized = sanitize_url(req.full_url, self.bot_token)

        try:
            if self._http_client is not None:
                response_bytes = self._http_client(req, self.request_timeout_sec)
            else:
                ssl_context = ssl.create_default_context()
                with urllib.request.urlopen(req, timeout=self.request_timeout_sec, context=ssl_context) as resp:
                    response_bytes = resp.read()

            parsed = json.loads(response_bytes.decode("utf-8"))
            if not parsed.get("ok", False):
                err_code = parsed.get("error_code", 400)
                description = parsed.get("description", "Unknown Telegram API rejection")
                retry_after = parsed.get("parameters", {}).get("retry_after")
                raise TelegramAPIError(description, error_code=err_code, retry_after=retry_after)

            return parsed

        except urllib.error.HTTPError as http_err:
            raw_body = http_err.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(raw_body)
                err_code = err_json.get("error_code", http_err.code)
                description = err_json.get("description", str(http_err))
                retry_after = err_json.get("parameters", {}).get("retry_after")
            except Exception:
                err_code = http_err.code
                description = raw_body or str(http_err)
                retry_after = None

            raise TelegramAPIError(
                f"Telegram HTTP {err_code}: {description}",
                error_code=err_code,
                retry_after=retry_after,
            ) from http_err

        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as conn_err:
            logger.warning("Connection failure communicating with %s: %s", sanitized, conn_err)
            raise

    def _build_multipart_payload(
        self,
        fields: Dict[str, str],
        file_field_name: str,
        file_path: Path,
        boundary: str,
    ) -> Tuple[bytes, str]:
        """Construct standard multipart/form-data byte payload for image upload."""
        body = bytearray()

        for name, value in fields.items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
            body.extend(f"{value}\r\n".encode("utf-8"))

        filename = file_path.name
        with open(file_path, "rb") as f:
            file_bytes = f.read()

        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"\r\n'.encode("utf-8")
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(file_bytes)
        body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))

        content_type = f"multipart/form-data; boundary={boundary}"
        return bytes(body), content_type

    def _calculate_backoff(self, retry_count: int) -> float:
        """Calculate bounded exponential backoff in seconds."""
        backoff = self.initial_retry_backoff_sec * (2 ** retry_count)
        return float(min(backoff, self.max_retry_backoff_sec))

