"""Voice activity detection preprocessing helpers."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol

_LOGGER = logging.getLogger(__name__)

VAD_MISSING_DEPENDENCY_ERROR_CODE = "vad_dependency_missing"
VAD_MISSING_DEPENDENCY_INSTALL_HINT = (
    'pipx install --force --python python3.14 "echoalign-asr-mlx[mlx]"'
)


@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold: float = 0.25
    min_speech_duration_ms: int = 80
    min_silence_duration_ms: int = 300
    speech_pad_ms: int = 1200
    merge_gap_sec: float = 3.0
    input_padding_sec: float = 0.8
    max_alignment_unit_sec: float = 180.0


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
class AlignmentUnit:
    index: int
    speech_start: float
    speech_end: float
    input_start: float
    input_end: float
    source_span_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    enabled: bool
    status: Literal["disabled", "ok", "failed"]
    duration_sec: float
    raw_spans: list[SpeechSpan]
    alignment_units: list[AlignmentUnit]
    config: VadConfig
    error: str | None = None
    error_code: str | None = None
    install_hint: str | None = None


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
        duration_sec=_safe_duration(duration_sec),
        raw_spans=[],
        alignment_units=[],
        config=config,
    )


def failed_speech_plan(
    *,
    duration_sec: float,
    error: str,
    config: VadConfig = DEFAULT_VAD_CONFIG,
    error_code: str | None = None,
    install_hint: str | None = None,
) -> SpeechPlan:
    return SpeechPlan(
        enabled=True,
        status="failed",
        duration_sec=_safe_duration(duration_sec),
        raw_spans=[],
        alignment_units=[],
        config=config,
        error=error,
        error_code=error_code,
        install_hint=install_hint,
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
        alignment_units=build_alignment_units(
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


def build_alignment_units(
    spans: Iterable[SpeechSpan],
    *,
    duration_sec: float,
    config: VadConfig = DEFAULT_VAD_CONFIG,
) -> list[AlignmentUnit]:
    duration = _safe_duration(duration_sec)
    sanitized = sanitize_speech_spans(spans, duration_sec=duration)
    units: list[AlignmentUnit] = []
    current: list[SpeechSpan] = []

    for span in sanitized:
        if not current:
            current = [span]
            continue

        current_start = current[0].start
        current_end = max(item.end for item in current)
        gap = span.start - current_end
        candidate_duration = max(span.end, current_end) - current_start

        if (
            gap <= config.merge_gap_sec
            and candidate_duration <= config.max_alignment_unit_sec
        ):
            current.append(span)
            continue

        units.append(_alignment_unit_from_spans(len(units), current, duration, config))
        current = [span]

    if current:
        units.append(_alignment_unit_from_spans(len(units), current, duration, config))

    return _trim_overlapping_unit_inputs(units)


def _alignment_unit_from_spans(
    index: int,
    spans: list[SpeechSpan],
    duration_sec: float,
    config: VadConfig,
) -> AlignmentUnit:
    speech_start = spans[0].start
    speech_end = max(span.end for span in spans)
    input_start = max(0.0, speech_start - config.input_padding_sec)
    input_end = min(duration_sec, speech_end + config.input_padding_sec)
    return AlignmentUnit(
        index=index,
        speech_start=speech_start,
        speech_end=speech_end,
        input_start=input_start,
        input_end=input_end,
        source_span_count=len(spans),
    )


def _trim_overlapping_unit_inputs(units: list[AlignmentUnit]) -> list[AlignmentUnit]:
    if len(units) < 2:
        return units

    trimmed = list(units)
    for index in range(len(trimmed) - 1):
        left = trimmed[index]
        right = trimmed[index + 1]
        if left.input_end <= right.input_start:
            continue
        midpoint = (left.speech_end + right.speech_start) / 2.0
        trimmed[index] = AlignmentUnit(
            index=left.index,
            speech_start=left.speech_start,
            speech_end=left.speech_end,
            input_start=left.input_start,
            input_end=max(left.input_start, midpoint),
            source_span_count=left.source_span_count,
        )
        trimmed[index + 1] = AlignmentUnit(
            index=right.index,
            speech_start=right.speech_start,
            speech_end=right.speech_end,
            input_start=min(right.input_end, midpoint),
            input_end=right.input_end,
            source_span_count=right.source_span_count,
        )

    return trimmed


def speech_plan_metadata(plan: SpeechPlan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "enabled": plan.enabled,
        "status": plan.status,
        "duration_sec": plan.duration_sec,
        "raw_span_count": len(plan.raw_spans),
        "alignment_unit_count": len(plan.alignment_units),
        "config": asdict(plan.config),
        "alignment_units": [unit.to_dict() for unit in plan.alignment_units],
    }
    if plan.error is not None:
        payload["error"] = plan.error
    if plan.error_code is not None:
        payload["error_code"] = plan.error_code
    if plan.install_hint is not None:
        payload["install_hint"] = plan.install_hint
    return payload


class SileroVadPreprocessor:
    """Build speech plans with Silero VAD, degrading to failed plans on errors."""

    def __init__(
        self,
        *,
        config: VadConfig = DEFAULT_VAD_CONFIG,
        sample_rate: int = 16_000,
        model_loader: Any | None = None,
        audio_reader: Any | None = None,
        timestamp_getter: Any | None = None,
        duration_probe: Any | None = None,
    ) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._model_loader = model_loader
        self._audio_reader = audio_reader
        self._timestamp_getter = timestamp_getter
        self._duration_probe = duration_probe
        self._model: Any | None = None

    def build_plan(self, audio_path: str | PathLike[str] | Path) -> SpeechPlan:
        path = Path(audio_path)
        try:
            duration_sec = self._probe_duration(path)
        except Exception as exc:
            _LOGGER.debug("VAD duration probe failed for %s", path, exc_info=True)
            return failed_speech_plan(
                duration_sec=0.0,
                error=f"duration probe failed: {str(exc) or type(exc).__name__}",
                config=self.config,
            )
        try:
            model = self._load_model()
            wav = self._read_audio(path)
            timestamps = self._get_timestamps(wav, model)
            spans = [self._timestamp_to_span(item) for item in timestamps]
            return build_speech_plan(
                duration_sec=duration_sec,
                raw_spans=[span for span in spans if span is not None],
                config=self.config,
            )
        except Exception as exc:
            _LOGGER.debug("Silero VAD preprocessing failed for %s", path, exc_info=True)
            return _failed_speech_plan_from_vad_exception(
                duration_sec=duration_sec,
                exc=exc,
                config=self.config,
            )

    def _probe_duration(self, audio_path: Path) -> float:
        duration_probe = self._duration_probe
        if duration_probe is None:
            from asr.providers.media_probe import probe_duration_sec

            duration_probe = probe_duration_sec
        duration_sec = float(duration_probe(audio_path))
        if not math.isfinite(duration_sec) or duration_sec < 0.0:
            raise ValueError("duration probe returned an invalid duration")
        return duration_sec

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        loader = self._model_loader
        if loader is None:
            from silero_vad import load_silero_vad

            loader = load_silero_vad
        self._model = loader()
        return self._model

    def _read_audio(self, audio_path: Path) -> Any:
        reader = self._audio_reader
        if reader is None:
            from silero_vad import read_audio

            reader = read_audio
        return reader(str(audio_path), sampling_rate=self.sample_rate)

    def _get_timestamps(self, wav: Any, model: Any) -> Iterable[dict[str, Any]]:
        getter = self._timestamp_getter
        if getter is None:
            from silero_vad import get_speech_timestamps

            getter = get_speech_timestamps
        return getter(
            wav,
            model,
            sampling_rate=self.sample_rate,
            threshold=self.config.threshold,
            min_speech_duration_ms=self.config.min_speech_duration_ms,
            min_silence_duration_ms=self.config.min_silence_duration_ms,
            speech_pad_ms=self.config.speech_pad_ms,
            return_seconds=True,
        )

    def _timestamp_to_span(self, item: dict[str, Any]) -> SpeechSpan | None:
        raw_start = item.get("start")
        raw_end = item.get("end")
        if raw_start is None or raw_end is None:
            return None
        start = float(raw_start)
        end = float(raw_end)
        return SpeechSpan(start=start, end=end)


def _safe_duration(duration_sec: float) -> float:
    if not math.isfinite(duration_sec):
        return 0.0
    return max(0.0, duration_sec)


def _failed_speech_plan_from_vad_exception(
    *,
    duration_sec: float,
    exc: Exception,
    config: VadConfig,
) -> SpeechPlan:
    if _is_missing_vad_dependency(exc):
        return failed_speech_plan(
            duration_sec=duration_sec,
            error="VAD dependencies are missing (silero-vad or torchcodec); VAD preprocessing was skipped.",
            config=config,
            error_code=VAD_MISSING_DEPENDENCY_ERROR_CODE,
            install_hint=VAD_MISSING_DEPENDENCY_INSTALL_HINT,
        )
    return failed_speech_plan(
        duration_sec=duration_sec,
        error=str(exc) or type(exc).__name__,
        config=config,
    )


def _is_missing_vad_dependency(exc: Exception) -> bool:
    dependency_names = {"silero_vad", "torchcodec"}
    message = str(exc)
    missing_name = getattr(exc, "name", None)
    if isinstance(exc, ModuleNotFoundError) and missing_name in dependency_names:
        return True
    return any(name in message for name in dependency_names)
