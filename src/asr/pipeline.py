"""ASR processing pipeline."""

from __future__ import annotations

from pathlib import Path

from asr.media import MediaPreparer
from asr.models import TranscriptionDocument
from asr.providers.base import Provider


def process_media_file(
    *,
    source_path: Path,
    provider: Provider,
    media_preparer: MediaPreparer,
) -> TranscriptionDocument:
    """Prepare a media file and transcribe it with the selected provider."""

    prepared_path = media_preparer.prepare(source_path)
    document = provider.transcribe(prepared_path)
    source_media = dict(document.source_media or {})
    source_media["prepared_audio_path"] = str(prepared_path)
    document.source_media = source_media
    document.source_path = str(source_path)
    return document
