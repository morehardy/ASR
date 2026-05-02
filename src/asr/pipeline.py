"""ASR processing pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path

from asr.media import MediaPreparer
from asr.models import TranscriptionDocument
from asr.observability.events import ObservabilityEvent
from asr.observability.observer import Observer
from asr.observability.timing import observe_step
from asr.providers.base import Provider
from asr.vad import (
    SileroVadPreprocessor,
    SpeechPlan,
    VadPreprocessor,
    disabled_speech_plan,
    failed_speech_plan,
    speech_plan_metadata,
)


def process_media_file(
    *,
    source_path: Path,
    provider: Provider,
    media_preparer: MediaPreparer,
    observer: Observer | None = None,
    run_id: str = "run-unknown",
    file_id: str | None = None,
    vad_enabled: bool = True,
    vad_preprocessor: VadPreprocessor | None = None,
) -> TranscriptionDocument:
    """Prepare a media file, optionally VAD-plan it, and transcribe it."""

    with observe_step(
        observer,
        run_id=run_id,
        file_id=file_id,
        source_path=str(source_path),
        step="prepare",
    ):
        prepared_path = media_preparer.prepare(source_path)

    speech_plan = disabled_speech_plan()
    if vad_enabled:
        preprocessor = vad_preprocessor or SileroVadPreprocessor()
        speech_plan = _run_vad_preprocessor(
            preprocessor=preprocessor,
            prepared_path=prepared_path,
            observer=observer,
            run_id=run_id,
            file_id=file_id,
            source_path=str(source_path),
        )

    if speech_plan.status == "ok" and not speech_plan.super_chunks:
        document = TranscriptionDocument(
            source_path=str(prepared_path),
            provider_name=provider.name,
            segments=[],
        )
    else:
        with observe_step(
            observer,
            run_id=run_id,
            file_id=file_id,
            source_path=str(source_path),
            step="transcribe",
        ):
            document = _transcribe_provider(
                provider=provider,
                prepared_path=prepared_path,
                speech_plan=speech_plan,
            )

    _attach_source_metadata(
        document=document,
        source_path=source_path,
        prepared_path=prepared_path,
        speech_plan=speech_plan,
    )
    return document


def _transcribe_provider(
    *,
    provider: Provider,
    prepared_path: Path,
    speech_plan: SpeechPlan,
) -> TranscriptionDocument:
    if (
        speech_plan.status == "ok"
        and speech_plan.super_chunks
        and _provider_accepts_speech_plan(provider)
    ):
        return provider.transcribe(prepared_path, speech_plan=speech_plan)  # type: ignore[call-arg]
    return provider.transcribe(prepared_path)


def _provider_accepts_speech_plan(provider: Provider) -> bool:
    try:
        signature = inspect.signature(provider.transcribe)
    except (TypeError, ValueError):
        return False
    parameter = signature.parameters.get("speech_plan")
    return (
        parameter is not None
        and parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        and _parameter_is_optional_extension(parameter)
    )


def _parameter_is_optional_extension(parameter: inspect.Parameter) -> bool:
    if parameter.default is not inspect.Parameter.empty:
        return True

    annotation = parameter.annotation
    if annotation is inspect.Parameter.empty:
        return False
    if annotation is None or annotation is type(None):
        return True
    return type(None) in getattr(annotation, "__args__", ())


def _run_vad_preprocessor(
    *,
    preprocessor: VadPreprocessor,
    prepared_path: Path,
    observer: Observer | None,
    run_id: str,
    file_id: str | None,
    source_path: str,
) -> SpeechPlan:
    start_meta = {"enabled": True}
    if observer is not None:
        observer.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id=run_id,
                file_id=file_id,
                source_path=source_path,
                step="preprocess_vad",
                meta=dict(start_meta),
            )
        )

    try:
        plan = preprocessor.build_plan(prepared_path)
    except Exception as exc:
        plan = failed_speech_plan(
            duration_sec=0.0,
            error=str(exc) or type(exc).__name__,
        )
        if observer is not None:
            error_meta = dict(start_meta)
            error_meta.update(_speech_plan_event_meta(plan))
            observer.on_event(
                ObservabilityEvent(
                    event_type="step_error",
                    run_id=run_id,
                    file_id=file_id,
                    source_path=source_path,
                    step="preprocess_vad",
                    meta=error_meta,
                )
            )
        return plan

    if observer is not None:
        end_meta = dict(start_meta)
        end_meta.update(_speech_plan_event_meta(plan))
        observer.on_event(
            ObservabilityEvent(
                event_type="step_error" if plan.status == "failed" else "step_end",
                run_id=run_id,
                file_id=file_id,
                source_path=source_path,
                step="preprocess_vad",
                meta=end_meta,
            )
        )
    return plan


def _speech_plan_event_meta(plan: SpeechPlan) -> dict[str, object]:
    meta: dict[str, object] = {
        "status": plan.status,
        "duration_sec": plan.duration_sec,
        "raw_span_count": len(plan.raw_spans),
        "super_chunk_count": len(plan.super_chunks),
    }
    if plan.error is not None:
        meta["error"] = plan.error
    return meta


def _attach_source_metadata(
    *,
    document: TranscriptionDocument,
    source_path: Path,
    prepared_path: Path,
    speech_plan: SpeechPlan,
) -> None:
    source_media = dict(document.source_media or {})
    source_media["prepared_audio_path"] = str(prepared_path)
    source_media["vad"] = speech_plan_metadata(speech_plan)
    document.source_media = source_media
    document.source_path = str(source_path)
