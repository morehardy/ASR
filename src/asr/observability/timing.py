"""Helpers for emitting timed step events."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional

from asr.observability.events import ObservabilityEvent
from asr.observability.observer import Observer


@contextmanager
def observe_step(
    observer: Optional[Observer],
    *,
    run_id: str,
    file_id: Optional[str],
    source_path: Optional[str],
    step: str,
    meta: Optional[dict[str, object]] = None,
) -> Iterator[None]:
    """Emit start/end or error events for one logical step."""

    if observer is None:
        yield
        return

    payload = dict(meta or {})
    observer.on_event(
        ObservabilityEvent(
            event_type="step_start",
            run_id=run_id,
            file_id=file_id,
            source_path=source_path,
            step=step,
            meta=payload,
        )
    )
    try:
        yield
    except Exception as exc:
        error_meta = dict(payload)
        error_meta["error"] = str(exc)
        observer.on_event(
            ObservabilityEvent(
                event_type="step_error",
                run_id=run_id,
                file_id=file_id,
                source_path=source_path,
                step=step,
                meta=error_meta,
            )
        )
        raise
    else:
        observer.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id=run_id,
                file_id=file_id,
                source_path=source_path,
                step=step,
                meta=payload,
            )
        )
