"""Media probing helpers for provider windowing."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List


_SILENCE_START = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


def probe_duration_sec(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def parse_silence_anchors(stderr_text: str) -> List[float]:
    starts = [float(match.group(1)) for match in _SILENCE_START.finditer(stderr_text)]
    ends = [float(match.group(1)) for match in _SILENCE_END.finditer(stderr_text)]
    return [(start + end) / 2.0 for start, end in zip(starts, ends)]
