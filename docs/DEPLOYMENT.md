# Production Deployment Guide: Linux VPS / Systemd

This guide covers deploying the Forex Trade Detection System on a low-cost Linux VPS (Ubuntu 22.04 LTS / Debian 12 / Oracle Linux) for 24/5 continuous monitoring.

---

## 1. Server Provisioning

A minimal VPS with **1 vCPU and 1 GB to 2 GB RAM** is sufficient:
- **Oracle Cloud Always Free**: VM.Standard.A1.Flex (ARM) or VM.Standard.E2.1.Micro (x86_64).
- **Hetzner / DigitalOcean / Linode**: Basic 1 vCPU / 1GB RAM droplet.

### System Updates & Core Packages:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-venv python3-pip ffmpeg sqlite3 git curl
```

---

## 2. Dedicated User & Directory Setup

To avoid running application code as `root`:
```bash
sudo useradd -m -s /bin/bash forex
sudo mkdir -p /opt/forex-detector
sudo chown -R forex:forex /opt/forex-detector
```

Switch to the `forex` user and copy or clone the repository into `/opt/forex-detector`:
```bash
su - forex
cd /opt/forex-detector
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-linux.txt
```

---

## 3. Configuration & Secrets

Create the production `.env` file with restrictive file permissions:
```bash
cat << 'EOF' > /opt/forex-detector/.env
TELEGRAM_BOT_TOKEN="your_bot_token_here"
TELEGRAM_CHAT_ID="your_chat_id_here"
EOF

chmod 600 /opt/forex-detector/.env
```

Create production configuration at `/opt/forex-detector/config/config.prod.yaml`:
- Verify the primary livestream URL: `https://www.youtube.com/watch?v=3H4IVQejlDE`.
- Ensure `box_confirmation_frames: 2` and `label_confirmation_count: 2` are set for production noise suppression.

---

## 4. Systemd Service Installation

Install the pre-packaged systemd service file:
```bash
sudo cp /opt/forex-detector/deployment/forex-trade-detector.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable forex-trade-detector.service
sudo systemctl start forex-trade-detector.service
```

### Checking Service Status:
```bash
sudo systemctl status forex-trade-detector.service
```

### Viewing Real-Time Logs:
```bash
sudo journalctl -u forex-trade-detector.service -f
# Or tail the application log directly:
tail -f /opt/forex-detector/logs/app.log
```

---

## 5. Automated SQLite Database Backups

Schedule non-blocking point-in-time database backups via `cron`:
```bash
crontab -e -u forex
```
Add an entry to create daily backups at 00:00 UTC:
```cron
0 0 * * * /opt/forex-detector/.venv/bin/python -m scripts.backup_db >> /opt/forex-detector/logs/backup.log 2>&1
```

The backup utility:
- Uses SQLite's native `sqlite3.Connection.backup` API (zero write locks, zero interference with live stream processing).
- Maintains the 14 most recent backups in `/opt/forex-detector/backups/`.
- Automatically prunes older backups.

---

## 6. Log Rotation Strategy

The application natively configures a Python `RotatingFileHandler` (10 MB per file, 5 backups). To integrate with system `logrotate`:

Create `/etc/logrotate.d/forex-detector`:
```logrotate
/opt/forex-detector/logs/*.log {
    weekly
    rotate 8
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
}
```

---

## 7. Recovery & Reboot Survival

- **System Reboot**: Systemd automatically restarts the service upon server reboot (`Restart=always`, `WantedBy=multi-user.target`).
- **Outbox Recovery**: On startup, the application calls `recover_pending_notifications()` to inspect SQLite for any `PENDING` or interrupted `SENDING` records, re-dispatching alerts for newly activated trades.
- **Baseline Re-calibration**: The first 10 frames after reboot are treated as baseline chart state to ensure pre-existing trades on the chart do not trigger false alerts.

