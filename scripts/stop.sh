#!/usr/bin/env bash
# ==============================================================================
# Forex Trade Detection System - Linux Process Graceful Stop Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR"

PID_FILE="$APP_DIR/app.pid"

if [[ ! -f "$PID_FILE" ]]; then
    echo "[INFO] No PID file found. Application does not appear to be running."
    exit 0
fi

PID=$(cat "$PID_FILE")

if ! kill -0 "$PID" 2>/dev/null; then
    echo "[INFO] Process $PID is not running. Removing stale PID file."
    rm -f "$PID_FILE"
    exit 0
fi

echo "Sending SIGTERM to process $PID for graceful shutdown..."
kill -SIGTERM "$PID"

# Wait up to 20 seconds for clean shutdown
WAIT_TIME=0
TIMEOUT=20

while kill -0 "$PID" 2>/dev/null && [ "$WAIT_TIME" -lt "$TIMEOUT" ]; do
    sleep 1
    WAIT_TIME=$((WAIT_TIME + 1))
    echo -n "."
done
echo ""

if kill -0 "$PID" 2>/dev/null; then
    echo "[WARN] Process $PID did not stop gracefully within ${TIMEOUT}s. Sending SIGKILL..."
    kill -SIGKILL "$PID" 2>/dev/null || true
    sleep 1
fi

rm -f "$PID_FILE"
echo "[SUCCESS] Application stopped cleanly."

