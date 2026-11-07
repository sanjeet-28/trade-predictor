#!/usr/bin/env bash
# ==============================================================================
# Forex Trade Detection System - Linux Process Startup Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR"

PID_FILE="$APP_DIR/app.pid"
LOG_FILE="$APP_DIR/logs/app.log"
CONFIG_FILE="${1:-config/config.prod.yaml}"

# Check if already running
if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "[ERROR] Application is already running (PID: $PID)."
        exit 1
    else
        echo "[WARN] Removing stale PID file."
        rm -f "$PID_FILE"
    fi
fi

# Activate virtual environment if present
if [[ -f "$APP_DIR/.venv/bin/activate" ]]; then
    source "$APP_DIR/.venv/bin/activate"
fi

# Load environment variables
if [[ -f "$APP_DIR/.env" ]]; then
    set -a
    source "$APP_DIR/.env"
    set +a
fi

mkdir -p "$APP_DIR/logs" "$APP_DIR/captures/screenshots" "$APP_DIR/storage"

echo "Starting Forex Trade Detection System..."
nohup python3 main.py --config "$CONFIG_FILE" >> "$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

echo "[SUCCESS] Application started with PID: $PID"
echo "Logging output redirected to: $LOG_FILE"

