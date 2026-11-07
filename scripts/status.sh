#!/usr/bin/env bash
# ==============================================================================
# Forex Trade Detection System - Linux Process Status Script
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$APP_DIR"

PID_FILE="$APP_DIR/app.pid"
LOG_FILE="$APP_DIR/logs/app.log"
DB_FILE="$APP_DIR/storage/trade_state.db"

echo "=================================================================="
echo "Forex Trade Detection System - Process & Operational Status"
echo "=================================================================="

if [[ -f "$PID_FILE" ]]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Status:       RUNNING (PID: $PID)"
        # Display memory and CPU
        if command -v ps &>/dev/null; then
            ps -p "$PID" -o pid,vsz,rss,%cpu,%mem,etime
        fi
    else
        echo "Status:       STOPPED (Stale PID file found: $PID)"
    fi
else
    echo "Status:       STOPPED (No active process)"
fi

echo "------------------------------------------------------------------"
if [[ -f "$DB_FILE" ]] && command -v sqlite3 &>/dev/null; then
    echo "Active Trades in Database:"
    sqlite3 "$DB_FILE" "SELECT trade_id, state, detector_source, first_detected_at, notification_sent FROM trades WHERE state = 'ACTIVE' LIMIT 5;" || true
    echo ""
    echo "Pending Notifications in Outbox:"
    sqlite3 "$DB_FILE" "SELECT notification_id, trade_id, status, retry_count FROM notifications WHERE status = 'PENDING';" || true
fi

echo "------------------------------------------------------------------"
echo "Recent Application Logs (last 10 lines):"
if [[ -f "$LOG_FILE" ]]; then
    tail -n 10 "$LOG_FILE"
else
    echo "[INFO] No log file found at $LOG_FILE"
fi
echo "=================================================================="

