"""Unit tests for YouTube stream manifest resolver using yt-dlp."""

import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest

from core.config import AppConfig, StreamConfig
from stream.resolver import (
    StreamOfflineError,
    StreamResolution,
    StreamResolver,
    StreamResolverError,
    StreamUnavailableError,
)


@pytest.fixture
def mock_formats() -> list:
    """Mock HLS formats returned by yt-dlp."""
    exp = int(time.time()) + 21600
    return [
        {
            "format_id": "229",
            "height": 240,
            "width": 426,
            "url": f"https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/{exp}/itag/229/playlist.m3u8",
        },
        {
            "format_id": "232",
            "height": 720,
            "width": 1280,
            "url": f"https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/{exp}/itag/232/playlist.m3u8",
        },
        {
            "format_id": "270",
            "height": 1080,
            "width": 1920,
            "url": f"https://manifest.googlevideo.com/api/manifest/hls_playlist/expire/{exp}/itag/270/playlist.m3u8",
        },
    ]


def test_resolve_optimal_format(mock_formats) -> None:
    """Verify resolver selects 720p format when preferred_height is 720."""
    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_live",
        preferred_height=720,
    )

    mock_info = {
        "is_live": True,
        "live_status": "is_live",
        "was_live": False,
        "formats": mock_formats,
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        resolution = resolver.resolve()

        assert isinstance(resolution, StreamResolution)
        assert resolution.format_id == "232"
        assert resolution.height == 720
        assert resolution.width == 1280
        assert "itag/232" in resolution.stream_url
        assert resolution.is_live is True
        assert resolution.expires_at is not None


def test_resolve_caching_and_expiration(mock_formats) -> None:
    """Verify stream URL is cached and refreshed only when expired or forced."""
    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_live",
        preferred_height=720,
    )

    mock_info = {
        "is_live": True,
        "live_status": "is_live",
        "formats": mock_formats,
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        # First call hits extractor
        res1 = resolver.resolve()
        assert mock_instance.extract_info.call_count == 1

        # Second call returns cached resolution
        res2 = resolver.resolve(force_refresh=False)
        assert res1 is res2
        assert mock_instance.extract_info.call_count == 1

        # Force refresh makes a new call
        res3 = resolver.resolve(force_refresh=True)
        assert mock_instance.extract_info.call_count == 2


def test_resolve_offline_stream() -> None:
    """Verify offline or ended stream raises StreamOfflineError."""
    resolver = StreamResolver(primary_url="https://www.youtube.com/watch?v=mock_offline")

    mock_info = {
        "is_live": False,
        "was_live": True,
        "live_status": "was_live",
        "formats": [],
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        with pytest.raises(StreamOfflineError):
            resolver.resolve()


def test_resolve_unavailable_stream() -> None:
    """Verify deleted or private video raises StreamUnavailableError."""
    import yt_dlp.utils

    resolver = StreamResolver(primary_url="https://www.youtube.com/watch?v=invalid_id")

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = yt_dlp.utils.DownloadError("Video unavailable")
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        with pytest.raises(StreamUnavailableError):
            resolver.resolve()


def test_resolve_transient_retry_and_backoff(mock_formats) -> None:
    """Verify transient extraction errors retry with exponential backoff."""
    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_transient",
        max_retries=3,
        initial_backoff_sec=0.01,
        backoff_multiplier=1.5,
    )

    mock_info = {
        "is_live": True,
        "live_status": "is_live",
        "formats": mock_formats,
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        # Fail first 2 attempts with network glitch, succeed on 3rd
        mock_instance.extract_info.side_effect = [
            RuntimeError("Connection reset"),
            RuntimeError("Gateway timeout"),
            mock_info,
        ]
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        resolution = resolver.resolve()
        assert resolution.format_id == "232"
        assert mock_instance.extract_info.call_count == 3


def test_stream_resolver_constructor_variants() -> None:
    """Verify StreamResolver accepts both direct arguments and config objects without TypeError."""
    stream_cfg = StreamConfig(
        primary_url="https://www.youtube.com/watch?v=3H4IVQejlDE",
        channel_url="https://www.youtube.com/@DTradingTips",
        frame_height=720,
    )

    # 1. Standard explicit arguments (Cell 8 pattern)
    r1 = StreamResolver(
        primary_url=stream_cfg.primary_url,
        channel_url=stream_cfg.channel_url,
        preferred_height=720,
    )
    assert r1.primary_url == "https://www.youtube.com/watch?v=3H4IVQejlDE"
    assert r1.channel_url == "https://www.youtube.com/@DTradingTips"
    assert r1.preferred_height == 720

    # 2. Config object as keyword argument (StreamConfig)
    r2 = StreamResolver(config=stream_cfg)
    assert r2.primary_url == "https://www.youtube.com/watch?v=3H4IVQejlDE"
    assert r2.channel_url == "https://www.youtube.com/@DTradingTips"
    assert r2.preferred_height == 720

    # 3. Config object as keyword argument (AppConfig)
    app_cfg = AppConfig()
    r3 = StreamResolver(config=app_cfg)
    assert r3.primary_url == app_cfg.stream.primary_url

    # 4. Missing primary URL raises ValueError
    with pytest.raises(ValueError, match="StreamResolver requires a valid primary_url"):
        StreamResolver()


def test_stream_resolver_unauthenticated_ydl_opts(mock_formats) -> None:
    """Verify resolver does not include cookiefile in ydl_opts when no cookie_file is provided."""
    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_live",
        cookie_file=None,
    )
    assert resolver.cookie_file is None

    mock_info = {
        "is_live": True,
        "live_status": "is_live",
        "formats": mock_formats,
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        resolver.resolve()

        assert mock_ydl_cls.call_count == 1
        opts_used = mock_ydl_cls.call_args[0][0]
        assert "cookiefile" not in opts_used


def test_stream_resolver_authenticated_ydl_opts(tmp_path, mock_formats) -> None:
    """Verify resolver includes cookiefile in ydl_opts when a valid cookie file exists."""
    dummy_cookie = tmp_path / "cookies.txt"
    dummy_cookie.write_text("# Netscape HTTP Cookie File\n.youtube.com\tTRUE\t/\tTRUE\t2147483647\tSID\tmock_val\n")

    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_live",
        cookie_file=str(dummy_cookie),
    )
    assert resolver.cookie_file == str(dummy_cookie)

    mock_info = {
        "is_live": True,
        "live_status": "is_live",
        "formats": mock_formats,
    }

    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.return_value = mock_info
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        resolver.resolve()

        assert mock_ydl_cls.call_count == 1
        opts_used = mock_ydl_cls.call_args[0][0]
        assert opts_used.get("cookiefile") == str(dummy_cookie)


def test_stream_resolver_nonexistent_cookie_file(tmp_path, caplog) -> None:
    """Verify resolver handles non-existent cookie file path gracefully by falling back to unauthenticated."""
    missing_cookie = tmp_path / "does_not_exist_cookies.txt"
    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_live",
        cookie_file=str(missing_cookie),
    )
    assert resolver.cookie_file is None
    assert "Specified cookie file was not found" in caplog.text


def test_stream_resolver_cookie_privacy_and_anti_bot_error(tmp_path, caplog) -> None:
    """Verify cookie token privacy: secret token never appears in logs, errors or repr, and anti-bot challenge is handled."""
    secret_token = "TOP_SECRET_SESSION_TOKEN_XYZ987"
    secret_cookie_file = tmp_path / "super_secret_cookies.txt"
    secret_cookie_file.write_text(f"# Netscape\n.youtube.com\tTRUE\t/\tTRUE\t0\tHSID\t{secret_token}\n")

    resolver = StreamResolver(
        primary_url="https://www.youtube.com/watch?v=mock_bot_check",
        cookie_file=str(secret_cookie_file),
    )

    # Verify repr/str doesn't expose cookie contents
    assert secret_token not in str(resolver)
    assert secret_token not in repr(resolver)

    import yt_dlp.utils
    with patch("yt_dlp.YoutubeDL") as mock_ydl_cls:
        mock_instance = MagicMock()
        mock_instance.extract_info.side_effect = yt_dlp.utils.DownloadError(
            f"ERROR: [youtube] mock_bot_check: Sign in to confirm you're not a bot (token={secret_token})"
        )
        mock_ydl_cls.return_value.__enter__.return_value = mock_instance

        with pytest.raises(StreamUnavailableError) as exc_info:
            resolver.resolve()

        # Check that error instructs user about cookies without echoing the secret token
        assert "Sign in to confirm you're not a bot" in str(exc_info.value)
        assert "Please provide a YouTube cookies.txt file" in str(exc_info.value)
        assert secret_token not in str(exc_info.value)
        assert secret_token not in caplog.text



