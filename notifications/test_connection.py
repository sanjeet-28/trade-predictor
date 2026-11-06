"""Manual connection test utility for Telegram Bot credentials.

Sends a single, explicitly identified test message when executed directly:
    python -m notifications.test_connection
"""

from __future__ import annotations

import json
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

from core.config import load_config
from notifications.telegram_notifier import mask_token, sanitize_url


def test_telegram_connection() -> bool:
    """Test Telegram Bot credentials and send a test message."""
    print("=" * 70)
    print("Telegram Bot API Connection Test Utility")
    print("=" * 70)

    cfg = load_config()
    token = cfg.telegram.bot_token
    chat_id = cfg.telegram.chat_id

    if not token:
        print("[ERROR] TELEGRAM_BOT_TOKEN is not set in .env or configuration.")
        print("Please add TELEGRAM_BOT_TOKEN to your .env file.")
        return False

    if not chat_id:
        print("[ERROR] TELEGRAM_CHAT_ID is not set in .env or configuration.")
        print("Please add TELEGRAM_CHAT_ID to your .env file.")
        return False

    print(f"Configured Bot Token: {mask_token(token)}")
    print(f"Configured Chat ID:   {chat_id}")

    # 1. Verify Bot Identity via getMe
    get_me_url = f"https://api.telegram.org/bot{token}/getMe"
    ssl_ctx = ssl.create_default_context()

    try:
        print("\n1. Verifying bot credentials with Telegram API (getMe)...")
        req = urllib.request.Request(get_me_url, method="GET")
        with urllib.request.urlopen(req, timeout=10.0, context=ssl_ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        if data.get("ok"):
            bot_info = data.get("result", {})
            username = bot_info.get("username", "Unknown")
            first_name = bot_info.get("first_name", "Bot")
            print(f"[SUCCESS] Bot Authenticated: {first_name} (@{username})")
        else:
            print(f"[FAILURE] Telegram rejected token: {data}")
            return False

    except Exception as e:
        print(f"[FAILURE] Failed to connect to Telegram API: {e}")
        return False

    # 2. Send Explicit Test Message
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    test_message = (
        f"[TEST] Forex Trade Detection System - Connection Test Successful\n"
        f"Timestamp: {now_str}\n"
        f"This is a verification test message, NOT an active trade signal."
    )

    send_url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps({"chat_id": chat_id, "text": test_message}).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    try:
        print(f"\n2. Sending test notification to Chat ID: {chat_id}...")
        req = urllib.request.Request(send_url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10.0, context=ssl_ctx) as resp:
            send_data = json.loads(resp.read().decode("utf-8"))

        if send_data.get("ok"):
            print("[SUCCESS] Test message delivered successfully! Check your Telegram app.")
            print("=" * 70)
            return True
        else:
            print(f"[FAILURE] Telegram rejected message: {send_data}")
            return False

    except Exception as e:
        print(f"[FAILURE] Failed to deliver test message: {e}")
        return False


if __name__ == "__main__":
    success = test_telegram_connection()
    sys.exit(0 if success else 1)

