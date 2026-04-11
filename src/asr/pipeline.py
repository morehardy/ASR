"""ASR processing pipeline."""

from __future__ import annotations

from pathlib import Path

from asr.media import MediaPreparer
from asr.models import TranscriptionDocument
from asr.observability.observer import Observer
from asr.observability.timing import observe_step
from asr.providers.base import Provider


def process_media_file(
    *,
    source_path: Path,
    provider: Provider,
    media_preparer: MediaPreparer,
    observer: Observer | None = None,
    run_id: str = "run-unknown",
    file_id: str | None = None,
) -> TranscriptionDocument:
    """Prepare a media file and transcribe it with the selected provider."""

    with observe_step(
        observer,
        run_id=run_id,
        file_id=file_id,
        source_path=str(source_path),
        step="prepare",
    ):
        prepared_path = media_preparer.prepare(source_path)
    with observe_step(
        observer,
        run_id=run_id,
        file_id=file_id,
        source_path=str(source_path),
        step="transcribe",
    ):
        document = provider.transcribe(prepared_path)
    source_media = dict(document.source_media or {})
    source_media["prepared_audio_path"] = str(prepared_path)
    document.source_media = source_media
    document.source_path = str(source_path)
    return document
