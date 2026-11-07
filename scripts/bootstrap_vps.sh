#!/usr/bin/env bash
# ==============================================================================
# Forex Trade Detection System - Automated Linux VPS Bootstrap Script
# Target: Ubuntu 22.04 / 24.04 LTS (ARM64 / aarch64 or x86_64)
# ==============================================================================
set -euo pipefail

# Require root/sudo
if [[ $EUID -ne 0 ]]; then
   echo "[ERROR] This bootstrap script must be run as root or with sudo:"
   echo "        sudo bash $0"
   exit 1
fi

echo "=================================================================="
echo "Forex Trade Detection System - Automated Host Bootstrap"
echo "=================================================================="

ARCH=$(uname -m)
echo "[INFO] Detected CPU Architecture: $ARCH"
if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" && "$ARCH" != "x86_64" ]]; then
    echo "[WARN] Untested architecture ($ARCH). Proceeding with standard Linux setup..."
fi

# 1. Update system package index and install prerequisites
echo "[INFO] Step 1/7: Installing system packages (ffmpeg, sqlite3, git, python3-venv)..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    sqlite3 \
    git \
    curl \
    ca-certificates

# 2. Create non-root application user and group
echo "[INFO] Step 2/7: Creating dedicated system user 'forex'..."
if ! id -u forex &>/dev/null; then
    useradd -m -s /bin/bash forex
    echo "[SUCCESS] Created service user 'forex'."
else
    echo "[INFO] User 'forex' already exists."
fi

# 3. Provision application directory
echo "[INFO] Step 3/7: Preparing application directory /opt/forex-detector..."
mkdir -p /opt/forex-detector
chown -R forex:forex /opt/forex-detector

# 4. Clone or update repository
echo "[INFO] Step 4/7: Synchronizing code from GitHub..."
REPO_URL="https://github.com/sanjeet-28/trade-predictor.git"
if [[ ! -d "/opt/forex-detector/.git" ]]; then
    sudo -u forex git clone "$REPO_URL" /opt/forex-detector
    echo "[SUCCESS] Cloned repository into /opt/forex-detector."
else
    echo "[INFO] Repository exists. Updating from origin/main..."
    cd /opt/forex-detector
    sudo -u forex git fetch origin
    sudo -u forex git checkout main
    sudo -u forex git pull origin main
fi

# 5. Initialize Python virtual environment and install dependencies
echo "[INFO] Step 5/7: Setting up Python virtual environment (aarch64 binary wheels)..."
cd /opt/forex-detector
if [[ ! -d "/opt/forex-detector/.venv" ]]; then
    sudo -u forex python3 -m venv /opt/forex-detector/.venv
fi

sudo -u forex /opt/forex-detector/.venv/bin/pip install --upgrade pip
sudo -u forex /opt/forex-detector/.venv/bin/pip install -r /opt/forex-detector/requirements-linux.txt

# 6. Initialize .env file with safe dry-run default
echo "[INFO] Step 6/7: Checking environment secrets configuration..."
if [[ ! -f "/opt/forex-detector/.env" ]]; then
    cat << 'EOF' > /opt/forex-detector/.env
# Telegram Bot Credentials (Fill with your actual credentials)
TELEGRAM_BOT_TOKEN="your_bot_token_here"
TELEGRAM_CHAT_ID="your_chat_id_here"
TELEGRAM_HEALTH_CHAT_ID=""

# Stream URL Overrides (leave empty to use config.prod.yaml defaults)
STREAM_PRIMARY_URL=""
STREAM_CHANNEL_URL=""

# Safe Dry-Run Mode (true: process stream without sending real Telegram alerts)
DRY_RUN=true
EOF
    chown forex:forex /opt/forex-detector/.env
    chmod 600 /opt/forex-detector/.env
    echo "[SUCCESS] Created /opt/forex-detector/.env template with chmod 600 permissions."
else
    chmod 600 /opt/forex-detector/.env
    echo "[INFO] Existing .env found. Ensured chmod 600 permissions."
fi

# 7. Install systemd service and cron backup
echo "[INFO] Step 7/7: Installing systemd service and scheduled backup cron..."
cp /opt/forex-detector/deployment/forex-trade-detector.service /etc/systemd/system/forex-trade-detector.service
systemctl daemon-reload

# Configure daily SQLite backup at 00:00 UTC for user 'forex'
BACKUP_CMD="/opt/forex-detector/.venv/bin/python -m scripts.backup_db >> /opt/forex-detector/logs/backup.log 2>&1"
CRON_JOB="0 0 * * * $BACKUP_CMD"
if ! sudo -u forex crontab -l 2>/dev/null | grep -Fq "scripts.backup_db"; then
    (sudo -u forex crontab -l 2>/dev/null || true; echo "$CRON_JOB") | sudo -u forex crontab -
    echo "[SUCCESS] Scheduled daily SQLite backup cron at 00:00 UTC."
else
    echo "[INFO] Backup cron job already installed."
fi

# Ensure logging and storage directories exist
sudo -u forex mkdir -p /opt/forex-detector/logs /opt/forex-detector/storage /opt/forex-detector/captures/screenshots /opt/forex-detector/backups

echo "=================================================================="
echo "Bootstrap Completed Successfully!"
echo "=================================================================="
echo ""
echo "Next Steps for the Operator:"
echo "1. Edit Telegram credentials in /opt/forex-detector/.env:"
echo "     nano /opt/forex-detector/.env"
echo ""
echo "2. Test Telegram connection reachability:"
echo "     sudo -u forex /opt/forex-detector/.venv/bin/python main.py --test-connection"
echo ""
echo "3. Enable and start the systemd service in safe DRY-RUN mode:"
echo "     sudo systemctl enable --now forex-trade-detector.service"
echo ""
echo "4. Monitor live service logs:"
echo "     journalctl -u forex-trade-detector.service -f"
echo "     # or: tail -f /opt/forex-detector/logs/app.log"
echo ""
echo "5. After verifying stream decoding and 0 false positives, switch to live alerts:"
echo "     sed -i 's/DRY_RUN=true/DRY_RUN=false/' /opt/forex-detector/.env"
echo "     sudo systemctl restart forex-trade-detector.service"
echo "=================================================================="
