# Setup and Installation Guide

This document provides step-by-step instructions for installing and running the Forex Trade Detection System on Windows 11 and Linux.

---

## 1. System Requirements

- **Operating System**: Windows 11 (64-bit) or Linux (Ubuntu 22.04 LTS / Debian 12 / Oracle Linux).
- **Python**: Version 3.12.x (Python 3.12.10 recommended).
- **FFmpeg**: Version 6.0+ installed and accessible in the system `PATH`.
- **Hardware**: Minimum 1 vCPU, 1 GB RAM, 5 GB free disk space (low-cost VPS or local PC).

---

## 2. FFmpeg Installation

### Windows 11:
1. Open PowerShell as Administrator.
2. Run winget to install FFmpeg:
   ```powershell
   winget install Gyan.FFmpeg
   ```
3. Refresh your environment `PATH` or reopen PowerShell:
   ```powershell
   ffmpeg -version
   ```

### Linux (Ubuntu / Debian):
```bash
sudo apt update
sudo apt install -y ffmpeg
ffmpeg -version
```

---

## 3. Python Virtual Environment Setup

### Windows 11:
```powershell
cd "Trade detection sysytem"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux:
```bash
cd /opt/forex-detector
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-linux.txt
```

---

## 4. Telegram Bot Configuration

### Step 1: Create your Bot with @BotFather
1. Open the Telegram app and search for `@BotFather`.
2. Send `/newbot`.
3. Enter a display name (e.g. `Forex Trade Alert Bot`).
4. Enter a unique username ending in `bot` (e.g. `my_forex_trade_alert_bot`).
5. Copy the generated **HTTP API token** (e.g. `123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ`).

### Step 2: Obtain your Telegram Chat ID
1. Search for `@userinfobot` on Telegram and send `/start`.
2. Copy your numeric **Id** (e.g. `987654321`).
3. Open your newly created bot in Telegram and click **Start** (or send `/start`) to authorize it to message you.

### Step 3: Populate `.env`
In the project root directory, create or edit `.env`:
```env
TELEGRAM_BOT_TOKEN="123456789:ABCdefGhIJKlmNoPQRsTUVwxyZ"
TELEGRAM_CHAT_ID="987654321"
```

---

## 5. Verify Setup

Run the standalone credential verification utility:
```powershell
python -m notifications.test_connection
```
This utility calls Telegram's `getMe` endpoint to verify authentication, then sends a test message (`[TEST] Forex Trade Detection System - Connection Test Successful`).

---

## 6. Running the Complete Test Suite

Run the full automated test suite to verify all visual detectors, Central Event Manager, and notification outbox:
```powershell
pytest -v
```
All 130 tests should pass cleanly.

