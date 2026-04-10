"""Provider abstraction boundary."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from asr.models import TranscriptionDocument


@runtime_checkable
class Provider(Protocol):
    """Provider interface for transcription backends."""

    name: str

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        """Transcribe a prepared audio file into the canonical model."""
