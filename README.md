# Forex Trade Detection & Telegram Notification System

A Python application that continuously monitors a 24/5 YouTube forex trading livestream, independently identifies active trade setups through dual visual computer vision detectors, deduplicates and correlates signals in a central event manager, and delivers immediate alerts via Telegram.

Strict Notification Contract: **`TRADE IS ACTIVE!`**

---

## Key Features

- **Continuous 24/5 Stream Ingestion**: Headless video ingestion at 720p 1.0 FPS using `yt-dlp` and `FFmpeg` with automatic stream reconnect and exponential backoff.
- **Dual Independent Detectors**:
  - **Detector A (Label Recognition)**: Real-time OCR text badge recognition (`BUY` / `SELL`) using local RapidOCR ONNX models (15–25ms CPU inference) with false-positive chart zone rejection.
  - **Detector B (Trade-Position Box)**: Geometric bipolar HSV color segmentation, candle width-parity analysis, and entry-line separator detection for Take Profit / Stop Loss structures.
- **Central Event Manager**:
  - Requires at least two independent pieces of structural/spatial evidence to correlate events.
  - Strict directional segregation (BUY and SELL are never merged).
  - Handles multi-trade concurrency, out-of-order frames, and startup baseline calibration.
- **Resilient Telegram Notifier**:
  - Transactional SQLite Outbox (`UNIQUE(trade_id)`) guarantees zero duplicate deliveries.
  - Automatic frame snapshot attachment with FIFO disk pruning.
  - Standard library `urllib.request` implementation with TLS 1.3 verification and zero cloud SDK bloat.
  - Bounded exponential backoff with intelligent temporary vs permanent HTTP error classification.
- **Cross-Platform**: Runs natively on Windows 11 (PowerShell) and Linux VPS (systemd / bash).

---

## Architecture Overview

```
                                [ YouTube Livestream ]
                                          │
                                 (yt-dlp Stream URL)
                                          │
                                   [ FFmpeg Pipe ]
                               (720p 1.0 FPS BGR Frames)
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             [ Detector A: Labels ]               [ Detector B: Boxes ]
             (RapidOCR ONNX Inference)            (Bipolar HSV Geometry)
                        │                                   │
                        └─────────────────┬─────────────────┘
                                          ▼
                              [ Central EventManager ]
                           (Multi-Evidence Correlation)
                                          │
                                 (Trade Confirmed)
                                          ▼
                             [ SQLite Outbox Storage ]
                           (Unique Trade ID Constraint)
                                          │
                                 (Outbox Dispatcher)
                                          ▼
                                [ Telegram Bot API ]
                              Alert: "TRADE IS ACTIVE!"
```

---

## Quick Start (Local Windows 11 / Linux)

### 1. Prerequisites
- Python 3.12+
- FFmpeg installed and available on system `PATH`
- A Telegram Bot token and target Chat ID (from `@BotFather`)

### 2. Environment Setup
```powershell
# Clone or enter workspace
cd "Trade detection sysytem"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows
# source .venv/bin/activate    # On Linux

# Install dependencies
pip install -r requirements.txt         # Windows
# pip install -r requirements-linux.txt # Linux VPS
```

### 3. Configuration
Copy `.env.example` to `.env` and fill in your credentials:
```env
TELEGRAM_BOT_TOKEN="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
TELEGRAM_CHAT_ID="987654321"
```

### 4. Verify Telegram Connectivity
```powershell
python -m notifications.test_connection
```

### 5. Run the Application
```powershell
# Live monitoring mode
python main.py

# Safe dry-run mode (zero Telegram alerts)
python main.py --dry-run

# Offline simulation mode (no credentials required)
python main.py --simulate
```

---

## CLI Options

| Flag | Description |
| :--- | :--- |
| `--config PATH` | Specify custom YAML configuration file path. |
| `--dry-run` | Run stream ingestion and detection while suppressing external Telegram alerts. |
| `--simulate` | Run in offline mode without requiring Telegram credentials. |
| `--local-video PATH` | Process a local video file (MP4/MKV) instead of live YouTube. |
| `--max-frames N` | Process $N$ video frames and exit cleanly. |
| `--test-connection` | Verify Telegram credentials via `getMe` and send a test message. |
| `--version` | Display system version. |

---

## Project Documentation

Detailed guides are available in the [docs/](file:///c:/Users/sanje/OneDrive/Desktop/Trade%20detection%20sysytem/docs) directory:
- [SETUP.md](file:///c:/Users/sanje/OneDrive/Desktop/Trade%20detection%20sysytem/docs/SETUP.md): Step-by-step installation, FFmpeg setup, and virtual environment guide.
- [ARCHITECTURE.md](file:///c:/Users/sanje/OneDrive/Desktop/Trade%20detection%20sysytem/docs/ARCHITECTURE.md): Comprehensive subsystem design, outbox pattern, and state lifecycles.
- [DEPLOYMENT.md](file:///c:/Users/sanje/OneDrive/Desktop/Trade%20detection%20sysytem/docs/DEPLOYMENT.md): Linux VPS setup, systemd daemon service, log rotation, and automated SQLite backups.
- [TROUBLESHOOTING.md](file:///c:/Users/sanje/OneDrive/Desktop/Trade%20detection%20sysytem/docs/TROUBLESHOOTING.md): Diagnosis and remedies for common operational issues.

