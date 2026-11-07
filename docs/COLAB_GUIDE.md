# Google Colab Free Usage Guide

This guide walks you through running on-demand Forex monitoring sessions (typically 1–3 hours) using **Google Colab Free (Standard CPU Runtime)**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sanjeet-28/trade-predictor/blob/main/notebooks/Forex_Trade_Detector.ipynb)

---

## 1. Architecture Overview

* **Zero-Cost Compute**: Runs on Google Colab's standard CPU runtime. **No GPU is required**.
* **Storage Model**:
  * **Active Database**: Stored on Colab's high-speed local SSD at `/content/trade_detector/storage/trade_state.db` in WAL mode with native POSIX locks.
  * **Persistent Storage**: Google Drive is mounted at `/content/drive/MyDrive/ForexTradeDetector/` strictly for persistent database backup snapshots (`/backups/`) and log files (`/logs/`).
  * *The live SQLite database is never operated directly on mounted Google Drive to prevent FUSE filesystem latency and database corruption.*
* **Security & Secrets**: Telegram credentials are loaded securely through Google Colab Secrets (`google.colab.userdata`). Tokens are never hardcoded in notebook cells and are automatically masked in log outputs.
* **Safety First**: `DRY_RUN = True` is the default. Live alerts require an explicit user confirmation toggle (`ENABLE_LIVE_NOTIFICATIONS = True`).

---

## 2. Quickstart Instructions

### Step 1: Open the Notebook in Google Colab
Click the badge above or navigate to:
```text
https://colab.research.google.com/github/sanjeet-28/trade-predictor/blob/main/notebooks/Forex_Trade_Detector.ipynb
```

### Step 2: Add Colab Secrets (Optional for Dry-Run, Required for Live Alerts)
1. Click the **Key icon** (Secrets) in the left sidebar of Google Colab.
2. Add the following secrets:
   * `TELEGRAM_BOT_TOKEN`: Your Telegram Bot API token (from `@BotFather`).
   * `TELEGRAM_CHAT_ID`: Your numerical Telegram chat ID.
3. Toggle the switch to grant **Notebook access** to these secrets.

> [!NOTE]
> If you are running in safe **Dry-Run mode** (the default), you do **not** need to add any secrets. The notebook will run without credentials.

### Step 3: Run the Cells in Sequence
1. **Cell 1**: Verifies Python 3.10+ and standard CPU runtime.
2. **Cell 2**: Authorizes and mounts your Google Drive (`/content/drive/MyDrive/ForexTradeDetector`).
3. **Cell 3**: Clones the latest code from GitHub into `/content/trade_predictor`.
4. **Cell 4**: Installs Linux dependencies (`rapidocr-onnxruntime`, `opencv-python-headless`, `yt-dlp`).
5. **Cell 5**: Verifies system FFmpeg.
6. **Cell 6**: Configures monitoring duration and safety gate:
   * `MONITORING_DURATION_MINUTES = 60` (choose 1 to 180 minutes).
   * `DRY_RUN = True` (keep `True` to test without sending real messages).
   * `ENABLE_LIVE_NOTIFICATIONS = False` (set `True` only when ready for real alerts).
7. **Cell 7**: Scans Google Drive and restores the latest database backup if one exists from a previous session.
8. **Cell 8**: Resolves the live YouTube stream and confirms 1280x720 video feed availability.
9. **Cell 9**: Initializes the runtime configuration.
10. **Cell 10**: Starts live trade monitoring. You will see real-time frame and trade logs in the cell output.
11. **Cell 11**: Displays a summary of frames processed, trades activated, and outbox records.
12. **Cell 12**: Cleanly stops FFmpeg, executes an atomic SQLite online backup, copies the snapshot to Google Drive, and verifies its schema integrity.

---

## 3. Session Controls & Stopping

* **Automatic Duration Stop**: The session will stop automatically after `MONITORING_DURATION_MINUTES` has elapsed.
* **Manual Interrupt**: You can click the **Stop button** on Cell 10 (or interrupt the kernel) at any time. The `TradeDetectionOrchestrator` will catch `KeyboardInterrupt`, stop FFmpeg cleanly, flush records, and proceed to Cell 12 for the final backup.

---

## 4. Persistent Artifacts on Google Drive

After each session, the following files are saved to your Google Drive in `MyDrive/ForexTradeDetector/`:
* `backups/trade_state_backup_YYYYMMDD_HHMMSS.db`: Timestamped point-in-time database snapshots.
* `logs/colab_app_YYYYMMDD_HHMMSS.log`: Complete execution log for auditing.

---

## 5. Important Compliance & Best Practices

1. **No Keep-Alive Hacks**: Do not implement automated page-clickers, keep-alive loops, or reconnection scripts. Google Colab Free is intended for interactive, human-supervised sessions.
2. **Session Lifetimes**: Colab Free runtimes have a maximum execution duration of up to 12 hours and disconnect if left idle for ~30 minutes. Keep your monitoring sessions between 1 and 3 hours.
3. **Database Integrity**: Always let Cell 12 run to completion so that the final state is synced to your Google Drive.
