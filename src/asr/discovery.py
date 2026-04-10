"""Media file discovery helpers."""

from __future__ import annotations

from pathlib import Path

SUPPORTED_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".m4a",
    ".flac",
    ".aac",
    ".mp4",
    ".mov",
    ".m4v",
    ".mkv",
    ".webm",
}


def is_supported_media_file(path: Path) -> bool:
    """Return True when a path points to a supported media file."""

    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def discover_media_files(path: Path, recursive: bool = False) -> list[Path]:
    """Discover supported media files from a file or directory path."""

    candidate = path.expanduser()
    if is_supported_media_file(candidate):
        return [candidate]
    if not candidate.exists() or not candidate.is_dir():
        return []

    pattern = "**/*" if recursive else "*"
    return sorted(item for item in candidate.glob(pattern) if is_supported_media_file(item))
