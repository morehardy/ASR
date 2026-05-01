"""Voice activity detection preprocessing helpers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol


@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold: float = 0.25
    min_speech_duration_ms: int = 80
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 1200
    merge_gap_sec: float = 12.0
    chunk_padding_sec: float = 4.0


DEFAULT_VAD_CONFIG = VadConfig()


@dataclass(frozen=True, slots=True)
class SpeechSpan:
    start: float
    end: float
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"start": self.start, "end": self.end}
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        return payload


@dataclass(frozen=True, slots=True)
class SuperChunk:
    index: int
    speech_start: float
    speech_end: float
    chunk_start: float
    chunk_end: float
    source_span_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    enabled: bool
    status: Literal["disabled", "ok", "failed"]
    duration_sec: float
    raw_spans: list[SpeechSpan]
    super_chunks: list[SuperChunk]
    config: VadConfig
    error: str | None = None


class VadPreprocessor(Protocol):
    def build_plan(self, audio_path: Path) -> SpeechPlan:
        """Build a speech plan for a prepared audio file."""


def disabled_speech_plan(
    *,
    config: VadConfig = DEFAULT_VAD_CONFIG,
    duration_sec: float = 0.0,
) -> SpeechPlan:
    return SpeechPlan(
        enabled=False,
        status="disabled",
        duration_sec=max(0.0, duration_sec),
        raw_spans=[],
        super_chunks=[],
        config=config,
    )


def failed_speech_plan(
    *,
    duration_sec: float,
    error: str,
    config: VadConfig = DEFAULT_VAD_CONFIG,
) -> SpeechPlan:
    return SpeechPlan(
        enabled=True,
        status="failed",
        duration_sec=max(0.0, duration_sec),
        raw_spans=[],
        super_chunks=[],
        config=config,
        error=error,
    )


def build_speech_plan(
    *,
    duration_sec: float,
    raw_spans: Iterable[SpeechSpan],
    config: VadConfig = DEFAULT_VAD_CONFIG,
) -> SpeechPlan:
    duration = _safe_duration(duration_sec)
    sanitized = sanitize_speech_spans(raw_spans, duration_sec=duration)
    return SpeechPlan(
        enabled=True,
        status="ok",
        duration_sec=duration,
        raw_spans=sanitized,
        super_chunks=build_super_chunks(
            sanitized,
            duration_sec=duration,
            config=config,
        ),
        config=config,
    )


def sanitize_speech_spans(
    raw_spans: Iterable[SpeechSpan],
    *,
    duration_sec: float,
) -> list[SpeechSpan]:
    duration = _safe_duration(duration_sec)
    sanitized: list[SpeechSpan] = []
    for span in raw_spans:
        if not math.isfinite(span.start) or not math.isfinite(span.end):
            continue
        start = min(max(0.0, span.start), duration)
        end = min(max(0.0, span.end), duration)
        if end <= start:
            continue
        confidence = span.confidence
        if confidence is not None and not math.isfinite(confidence):
            confidence = None
        sanitized.append(SpeechSpan(start=start, end=end, confidence=confidence))
    sanitized.sort(key=lambda span: (span.start, span.end))
    return sanitized


def build_super_chunks(
    spans: Iterable[SpeechSpan],
    *,
    duration_sec: float,
    config: VadConfig = DEFAULT_VAD_CONFIG,
) -> list[SuperChunk]:
    duration = _safe_duration(duration_sec)
    chunks: list[SuperChunk] = []
    current_speech_start: float | None = None
    current_speech_end: float | None = None
    current_chunk_start: float | None = None
    current_chunk_end: float | None = None
    current_count = 0

    for span in sanitize_speech_spans(spans, duration_sec=duration):
        padded_start = max(0.0, span.start - config.chunk_padding_sec)
        padded_end = min(duration, span.end + config.chunk_padding_sec)
        if current_chunk_start is None or current_chunk_end is None:
            current_speech_start = span.start
            current_speech_end = span.end
            current_chunk_start = padded_start
            current_chunk_end = padded_end
            current_count = 1
            continue

        gap = padded_start - current_chunk_end
        if gap <= config.merge_gap_sec:
            current_speech_end = span.end if current_speech_end is None else max(current_speech_end, span.end)
            current_chunk_end = max(current_chunk_end, padded_end)
            current_count += 1
            continue

        chunks.append(
            SuperChunk(
                index=len(chunks),
                speech_start=current_speech_start if current_speech_start is not None else current_chunk_start,
                speech_end=current_speech_end if current_speech_end is not None else current_chunk_end,
                chunk_start=current_chunk_start,
                chunk_end=current_chunk_end,
                source_span_count=current_count,
            )
        )
        current_speech_start = span.start
        current_speech_end = span.end
        current_chunk_start = padded_start
        current_chunk_end = padded_end
        current_count = 1

    if current_chunk_start is not None and current_chunk_end is not None:
        chunks.append(
            SuperChunk(
                index=len(chunks),
                speech_start=current_speech_start if current_speech_start is not None else current_chunk_start,
                speech_end=current_speech_end if current_speech_end is not None else current_chunk_end,
                chunk_start=current_chunk_start,
                chunk_end=current_chunk_end,
                source_span_count=current_count,
            )
        )

    return chunks


def speech_plan_metadata(plan: SpeechPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": plan.enabled,
        "status": plan.status,
        "duration_sec": plan.duration_sec,
        "raw_span_count": len(plan.raw_spans),
        "super_chunk_count": len(plan.super_chunks),
        "config": asdict(plan.config),
        "super_chunks": [chunk.to_dict() for chunk in plan.super_chunks],
    }
    if plan.error is not None:
        payload["error"] = plan.error
    return payload


def _safe_duration(duration_sec: float) -> float:
    if not math.isfinite(duration_sec):
        return 0.0
    return max(0.0, duration_sec)
