# Troubleshooting & Diagnostic Guide

This document provides resolutions for operational, network, and detection issues.

---

## 1. Stream & Ingestion Issues

### Symptom: `StreamResolverError: No video formats found`
- **Cause**: YouTube stream URL has changed or the channel is offline between trading sessions.
- **Remedy**:
  1. Inspect the channel URL (`https://www.youtube.com/@DTradingTips`) to check if a new live URL was generated.
  2. Update `stream.primary_url` in `config/config.prod.yaml` or `.env`.
  3. The resolver automatically falls back to channel metadata extraction if configured.

### Symptom: `FFmpegFrameReader: Process exited with return code 1`
- **Cause**: Network timeout during HLS manifest segment download.
- **Remedy**:
  - The pipeline automatically applies exponential backoff (5s, 10s, 20s up to 60s) and forces an HLS URL refresh on reconnect. No manual intervention is required.

### Symptom: Health Monitor reports `Frozen video detected`
- **Cause**: Livestream broadcast software (OBS) has hung or stream is buffering static frames.
- **Remedy**:
  - `StreamHealthMonitor` logs a warning after 30 seconds of identical frames. If freezing persists, it restarts the FFmpeg reader.

---

## 2. Telegram Notification Issues

### Symptom: `TelegramAPIError: 401 Unauthorized`
- **Cause**: Invalid bot token in `.env`.
- **Remedy**: Verify token with `@BotFather`. Ensure no quotes or whitespace are accidentally included.

### Symptom: `TelegramAPIError: 403 Forbidden: bot was blocked by the user`
- **Cause**: The target chat ID has not sent `/start` to the bot, or the user blocked the bot.
- **Remedy**: Open the bot in Telegram and send `/start`.

### Symptom: `TelegramAPIError: 429 Too Many Requests`
- **Cause**: Telegram rate limiting (max 1 message/sec per chat).
- **Remedy**:
  - `TelegramNotifier` automatically extracts the `retry_after` response header and schedules the outbox delivery attempt after the required backoff window.

### Symptom: Outbox Record stuck in `SENDING`
- **Cause**: Server hard power-off occurred while an HTTP request was in flight.
- **Remedy**:
  - Upon restart, `recover_interrupted_sending()` automatically resets `SENDING` records older than 5 minutes back to `PENDING` for redelivery.

---

## 3. Computer Vision & Detection Issues

### Symptom: Trade box is visible on chart but not detected
- **Cause**:
  1. Box is smaller than `min_box_area` (400 pixels) or width is $< 35\text{px}$.
  2. Livestream resolution is below 720p, distorting color segmentation.
  3. Stream compression artifacts altered the HSV green/red ranges.
- **Remedy**:
  - Adjust `detection.box_min_area` or `detection.box_min_width` in configuration.
  - Review diagnostic frames in `captures/validation/`.

### Symptom: `RapidOCR ONNX` inference is slow ($> 1\text{s}$)
- **Cause**: CPU saturation on low-end single-core VPS.
- **Remedy**:
  - RapidOCR default multi-threading utilizes all available cores. On single-core instances, set `OMP_NUM_THREADS=1`.
  - At 1.0 FPS, inference times up to 800ms still provide 20% headroom.

---

## 4. Database & Storage Issues

### Symptom: `sqlite3.OperationalError: database is locked`
- **Cause**: Concurrent write access without WAL mode or prolonged external read transaction.
- **Remedy**:
  - Ensure `init_db()` was called (enables `PRAGMA journal_mode=WAL;`).
  - `EventManager` automatically catches lock errors, stages the trade in `_pending_persist_trades`, and retries every cycle.

### Symptom: Disk space filling up
- **Cause**: Large number of screenshot images or oversized log files.
- **Remedy**:
  - `ScreenshotManager` automatically prunes old screenshots (retaining at most `max_stored_screenshots`, default 50).
  - Python `RotatingFileHandler` caps logs at 10 MB $\times$ 5 backups (50 MB max).
  - SQLite point-in-time backups are pruned to the latest 14 files by `scripts/backup_db.py`.

