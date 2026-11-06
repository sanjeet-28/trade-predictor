"""FFmpeg headless raw video frame reader.

Launches FFmpeg as a headless subprocess piping raw frames directly over stdout
into memory as NumPy arrays without saving continuous video to disk.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Optional

import numpy as np

from core.logger import get_logger
from core.models import Frame

logger = get_logger("stream.ffmpeg")


class FFmpegProcessError(Exception):
    """Raised when the FFmpeg subprocess fails, crashes, or returns an error."""


def find_ffmpeg_binary(binary_name: str = "ffmpeg") -> str:
    """Locate the ffmpeg or ffprobe executable on the current system.

    Checks PATH, Windows User Registry PATH, and standard WinGet package directories.
    """
    # 1. Standard PATH lookup
    found = shutil.which(binary_name)
    if found:
        return found

    # 2. Windows-specific fallback locations
    if sys.platform == "win32":
        local_app_data = os.getenv("LOCALAPPDATA", "")
        candidates = [
            # WinGet Packages directory
            os.path.join(local_app_data, "Microsoft", "WinGet", "Packages", "**", f"{binary_name}.exe"),
            # WinGet Links directory
            os.path.join(local_app_data, "Microsoft", "WinGet", "Links", f"{binary_name}.exe"),
            # Program Files fallback
            os.path.join(r"C:\Program Files", "ffmpeg", "bin", f"{binary_name}.exe"),
        ]
        for pattern in candidates:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                resolved = matches[0]
                # Prepend directory to os.environ["PATH"] for child processes
                bin_dir = os.path.dirname(resolved)
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                logger.debug("Discovered %s binary at fallback location: %s", binary_name, resolved)
                return resolved

    raise FileNotFoundError(
        f"'{binary_name}' executable not found in system PATH. "
        "Please ensure FFmpeg is installed (e.g. via 'winget install Gyan.FFmpeg')."
    )


class FFmpegFrameReader:
    """Manages an FFmpeg subprocess reading live HLS or local video streams."""

    def __init__(
        self,
        source: str,
        width: int = 1280,
        height: int = 720,
        fps: float = 1.0,
        pix_fmt: str = "bgr24",
        is_local_file: bool = False,
        ffmpeg_path: Optional[str] = None,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.pix_fmt = pix_fmt
        self.is_local_file = is_local_file
        self.ffmpeg_path = ffmpeg_path or find_ffmpeg_binary("ffmpeg")

        self.bytes_per_pixel = 3 if pix_fmt in ("bgr24", "rgb24") else 4
        self.frame_byte_size = self.width * self.height * self.bytes_per_pixel

        self._process: Optional[subprocess.Popen] = None
        self._frame_sequence: int = 0
        self._is_running: bool = False

    @property
    def is_running(self) -> bool:
        """Return True if the FFmpeg subprocess is active and running."""
        return self._is_running and self._process is not None and self._process.poll() is None

    def start(self) -> None:
        """Construct command line arguments and launch the FFmpeg subprocess."""
        if self.is_running:
            logger.warning("FFmpeg frame reader is already running.")
            return

        # Determine if source is a local file
        is_file = self.is_local_file or Path(self.source).exists()

        cmd = [
            self.ffmpeg_path,
            "-nostdin",
            "-loglevel", "error",
        ]

        if not is_file:
            # Network stream resilient flags
            cmd.extend([
                "-fflags", "nobuffer",
                "-flags", "low_delay",
                "-reconnect", "1",
                "-reconnect_streamed", "1",
                "-reconnect_delay_max", "5",
            ])

        cmd.extend([
            "-i", self.source,
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", self.pix_fmt,
            "pipe:1",
        ])

        logger.info(
            "Spawning FFmpeg process (target=%dx%d @ %.1f FPS, source=%s)",
            self.width,
            self.height,
            self.fps,
            self.source[:60] if len(self.source) > 60 else self.source,
        )

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self.frame_byte_size * 2,  # Buffer at most 2 frames
            )
            self._is_running = True
            self._frame_sequence = 0
        except Exception as e:
            self._is_running = False
            logger.error("Failed to launch FFmpeg process: %s", e)
            raise FFmpegProcessError(f"Could not spawn FFmpeg process: {e}") from e

    def read_frame(self) -> Optional[Frame]:
        """Read a single raw frame from the FFmpeg stdout pipe into a NumPy array."""
        if not self._is_running or self._process is None or self._process.stdout is None:
            return None

        try:
            raw_data = self._process.stdout.read(self.frame_byte_size)
        except Exception as e:
            logger.error("Error reading frame buffer from FFmpeg: %s", e)
            raise FFmpegProcessError(f"Pipe read error: {e}") from e

        if not raw_data:
            # Reached end-of-stream or process terminated
            try:
                retcode = self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                retcode = self._process.poll()

            stderr_msg = ""
            if self._process.stderr:
                try:
                    stderr_msg = self._process.stderr.read().decode("utf-8", errors="ignore").strip()
                except Exception:
                    pass

            if retcode is not None and retcode != 0:
                logger.error("FFmpeg exited with error code %d: %s", retcode, stderr_msg)
                raise FFmpegProcessError(f"FFmpeg process exited abnormally (code={retcode}): {stderr_msg}")

            logger.info("FFmpeg pipe closed (EOF). Retcode: %s", retcode)
            self.stop()
            return None

        if len(raw_data) < self.frame_byte_size:
            logger.warning(
                "Truncated frame received: %d bytes (expected %d). Stream may have closed.",
                len(raw_data),
                self.frame_byte_size,
            )
            self.stop()
            return None

        # Convert raw byte array to NumPy image
        frame_array = np.frombuffer(raw_data, dtype=np.uint8).reshape((self.height, self.width, self.bytes_per_pixel))
        self._frame_sequence += 1

        return Frame(
            frame_id=self._frame_sequence,
            timestamp=datetime.now(timezone.utc),
            width=self.width,
            height=self.height,
            data=frame_array,
        )

    def frames(self) -> Generator[Frame, None, None]:
        """Generator that yields frames sequentially as they arrive."""
        if not self.is_running:
            self.start()

        while self.is_running:
            frame = self.read_frame()
            if frame is None:
                break
            yield frame

    def stop(self) -> None:
        """Gracefully shut down the FFmpeg process and close pipes."""
        self._is_running = False
        if self._process is not None:
            logger.info("Stopping FFmpeg process (PID=%s)", self._process.pid)
            try:
                if self._process.stdout:
                    self._process.stdout.close()
                if self._process.stderr:
                    self._process.stderr.close()

                self._process.terminate()
                try:
                    self._process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process did not terminate within timeout; killing PID=%s", self._process.pid)
                    self._process.kill()
                    self._process.wait(timeout=1.0)
            except Exception as e:
                logger.debug("Error while terminating FFmpeg process: %s", e)
            finally:
                self._process = None
