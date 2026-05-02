# VAD Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add default high-recall Silero VAD preprocessing before provider execution, passing speech super-chunks to Qwen provider while preserving original-media timestamps and safe fallback behavior.

**Architecture:** Media preparation stays first. A new public `asr.vad` module builds `SpeechPlan` metadata from prepared mono 16 kHz audio; `process_media_file()` runs VAD before transcription and attaches VAD metadata to `source_media`. `QwenMlxProvider` accepts an optional speech plan and applies its existing bounded window planner inside each super-chunk.

**Tech Stack:** Python 3.14, `uv`, `unittest`, stdlib dataclasses/protocols/inspect, existing ffmpeg/ffprobe helpers, `silero-vad>=6.2.1,<7` in the `mlx` optional dependency set.

---

## File Structure

- Create `src/asr/vad.py`
  - Owns `VadConfig`, `SpeechSpan`, `SuperChunk`, `SpeechPlan`, metadata serialization, span sanitization, super-chunk merging, and the Silero-backed preprocessor.
  - Keeps VAD public and provider-independent.

- Create `tests/test_vad.py`
  - Focused unit tests for VAD planning, metadata, and Silero adapter behavior with injected fakes.

- Modify `src/asr/pipeline.py`
  - Runs VAD after media preparation.
  - Skips provider execution when VAD succeeds with zero super-chunks.
  - Passes speech plans only to providers that accept `speech_plan`.
  - Attaches VAD metadata beside provider metadata.

- Modify `tests/test_pipeline.py`
  - Adds pipeline tests for enabled VAD, disabled VAD, VAD failure fallback, no-speech short-circuit, and `preprocess_vad` observability.

- Modify `src/asr/cli.py`
  - Adds public `--no-vad`.
  - Passes `vad_enabled=not no_vad` into `process_media_file()`.

- Modify `tests/test_cli.py`
  - Verifies parser support and CLI handoff for default VAD and `--no-vad`.

- Modify `pyproject.toml`
  - Adds `silero-vad>=6.2.1,<7` to the `mlx` extra.

- Modify `src/asr/providers/windowing.py`
  - Adds optional `super_chunk_index` to `AlignmentWindow`.

- Modify `src/asr/providers/qwen_mlx.py`
  - Accepts optional `SpeechPlan`.
  - Plans bounded windows inside super-chunk envelopes.
  - Emits `vad_super_chunk_windowed_bounded_alignment` strategy metadata when VAD chunks are active.
  - Records `super_chunk_index` in diagnostics and provider window events.

- Modify `tests/test_qwen_provider_windowed.py`
  - Adds provider tests for super-chunk planning, global timestamp offsets, long-chunk bounded windows, and diagnostics.

- Modify `src/asr/observability/metrics.py`
  - Preserves step metadata in verbose metrics JSON so `preprocess_vad` counts are inspectable.

- Modify `tests/test_observability.py`
  - Verifies VAD step metadata is serialized.

- Modify `tests/test_exporters.py`
  - Verifies JSON output carries VAD metadata through `source_media`; SRT/VTT remain unaware of VAD internals.

- Modify `README.md`
  - Documents default VAD behavior and `--no-vad`.

---

### Task 1: Core VAD Data Model And Super-Chunk Planner

**Files:**
- Create: `src/asr/vad.py`
- Create: `tests/test_vad.py`

- [ ] **Step 1: Write failing VAD planner tests**

Create `tests/test_vad.py` with:

```python
import math
import unittest

from asr.vad import (
    DEFAULT_VAD_CONFIG,
    SpeechSpan,
    VadConfig,
    build_speech_plan,
    disabled_speech_plan,
    failed_speech_plan,
    speech_plan_metadata,
)


class VadPlanningTest(unittest.TestCase):
    def test_build_speech_plan_sanitizes_pads_merges_and_clamps(self) -> None:
        config = VadConfig(
            threshold=0.25,
            min_speech_duration_ms=80,
            min_silence_duration_ms=300,
            speech_pad_ms=1200,
            merge_gap_sec=12.0,
            chunk_padding_sec=4.0,
        )
        raw_spans = [
            SpeechSpan(start=10.0, end=12.0),
            SpeechSpan(start=20.0, end=20.0),
            SpeechSpan(start=math.nan, end=22.0),
            SpeechSpan(start=24.0, end=25.0),
            SpeechSpan(start=50.0, end=51.0),
            SpeechSpan(start=70.0, end=75.0),
        ]

        plan = build_speech_plan(
            duration_sec=60.0,
            raw_spans=raw_spans,
            config=config,
        )

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(10.0, 12.0), (24.0, 25.0), (50.0, 51.0)])
        self.assertEqual(len(plan.super_chunks), 2)
        first = plan.super_chunks[0]
        second = plan.super_chunks[1]
        self.assertEqual(first.index, 0)
        self.assertEqual(first.source_span_count, 2)
        self.assertEqual((first.speech_start, first.speech_end), (10.0, 25.0))
        self.assertEqual((first.chunk_start, first.chunk_end), (6.0, 29.0))
        self.assertEqual(second.index, 1)
        self.assertEqual(second.source_span_count, 1)
        self.assertEqual((second.speech_start, second.speech_end), (50.0, 51.0))
        self.assertEqual((second.chunk_start, second.chunk_end), (46.0, 55.0))

    def test_build_speech_plan_returns_ok_empty_plan_for_no_speech(self) -> None:
        plan = build_speech_plan(
            duration_sec=120.0,
            raw_spans=[],
            config=DEFAULT_VAD_CONFIG,
        )

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.status, "ok")
        self.assertEqual(plan.raw_spans, [])
        self.assertEqual(plan.super_chunks, [])

    def test_disabled_and_failed_plans_serialize_to_metadata(self) -> None:
        disabled = disabled_speech_plan(config=DEFAULT_VAD_CONFIG)
        failed = failed_speech_plan(
            duration_sec=12.5,
            error="silero import failed",
            config=DEFAULT_VAD_CONFIG,
        )

        disabled_meta = speech_plan_metadata(disabled)
        failed_meta = speech_plan_metadata(failed)

        self.assertEqual(disabled.status, "disabled")
        self.assertFalse(disabled_meta["enabled"])
        self.assertEqual(disabled_meta["status"], "disabled")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed_meta["status"], "failed")
        self.assertEqual(failed_meta["duration_sec"], 12.5)
        self.assertIn("silero import failed", failed_meta["error"])
        self.assertEqual(failed_meta["config"]["threshold"], 0.25)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the VAD planner tests to verify they fail**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_vad -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'asr.vad'`.

- [ ] **Step 3: Implement VAD data structures and planner helpers**

Create `src/asr/vad.py` with:

```python
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
```

- [ ] **Step 4: Run the VAD planner tests to verify they pass**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_vad -v
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/asr/vad.py tests/test_vad.py
git commit -m "feat: add vad speech plan model"
```

---

### Task 2: Silero VAD Adapter

**Files:**
- Modify: `src/asr/vad.py`
- Modify: `tests/test_vad.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Add failing Silero adapter tests**

Append this test class to `tests/test_vad.py` before the `if __name__ == "__main__":` block:

```python
class SileroVadPreprocessorTest(unittest.TestCase):
    def test_silero_preprocessor_requests_second_timestamps(self) -> None:
        from asr.vad import SileroVadPreprocessor

        seen_kwargs: dict[str, object] = {}

        def timestamp_getter(wav: object, model: object, **kwargs: object) -> list[dict[str, float]]:
            seen_kwargs.update(kwargs)
            return [
                {"start": 1.0, "end": 2.0},
                {"start": 3.0, "end": 3.5},
            ]

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=lambda path, sampling_rate: [0.0] * sampling_rate,
            timestamp_getter=timestamp_getter,
            duration_probe=lambda path: 4.0,
        )

        plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(1.0, 2.0), (3.0, 3.5)])
        self.assertTrue(seen_kwargs["return_seconds"])
        self.assertEqual(plan.config.threshold, 0.25)
        self.assertEqual(len(plan.super_chunks), 1)

    def test_silero_preprocessor_accepts_second_timestamps(self) -> None:
        from asr.vad import SileroVadPreprocessor

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=lambda path, sampling_rate: [0.0] * sampling_rate,
            timestamp_getter=lambda wav, model, **kwargs: [
                {"start": 1.25, "end": 2.5},
            ],
            duration_probe=lambda path: 5.0,
        )

        plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(1.25, 2.5)])

    def test_silero_preprocessor_returns_failed_plan_when_backend_raises(self) -> None:
        from asr.vad import SileroVadPreprocessor

        def explode() -> object:
            raise RuntimeError("backend unavailable")

        preprocessor = SileroVadPreprocessor(
            model_loader=explode,
            duration_probe=lambda path: 9.0,
        )

        plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "failed")
        self.assertEqual(plan.duration_sec, 9.0)
        self.assertIn("backend unavailable", plan.error or "")
```

- [ ] **Step 2: Run adapter tests to verify they fail**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_vad -v
```

Expected: FAIL with `ImportError` or `AttributeError` because `SileroVadPreprocessor` is not defined.

- [ ] **Step 3: Implement the Silero-backed preprocessor**

In `src/asr/vad.py`, add this import near the top:

```python
from os import PathLike

from asr.providers.media_probe import probe_duration_sec
```

In `src/asr/vad.py`, append this class after `speech_plan_metadata()` and before `_safe_duration()`:

```python
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
        duration_probe: Any = probe_duration_sec,
    ) -> None:
        self.config = config
        self.sample_rate = sample_rate
        self._model_loader = model_loader
        self._audio_reader = audio_reader
        self._timestamp_getter = timestamp_getter
        self._duration_probe = duration_probe
        self._model: Any | None = None

    def build_plan(self, audio_path: str | PathLike[str] | Path) -> SpeechPlan:
        duration_sec = self._probe_duration_safely(audio_path)
        try:
            model = self._load_model()
            wav = self._read_audio(audio_path)
            timestamps = self._get_timestamps(wav, model)
            spans = [self._timestamp_to_span(item) for item in timestamps]
            return build_speech_plan(
                duration_sec=duration_sec,
                raw_spans=[span for span in spans if span is not None],
                config=self.config,
            )
        except Exception as exc:
            return failed_speech_plan(
                duration_sec=duration_sec,
                error=str(exc) or type(exc).__name__,
                config=self.config,
            )

    def _probe_duration_safely(self, audio_path: str | PathLike[str] | Path) -> float:
        try:
            return _safe_duration(float(self._duration_probe(Path(audio_path))))
        except Exception:
            return 0.0

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        loader = self._model_loader
        if loader is None:
            from silero_vad import load_silero_vad

            loader = load_silero_vad
        self._model = loader()
        return self._model

    def _read_audio(self, audio_path: str | PathLike[str] | Path) -> Any:
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
```

- [ ] **Step 4: Add Silero dependency to the MLX extra**

Modify `pyproject.toml`:

```toml
[project.optional-dependencies]
mlx = [
  "mlx-audio>=0.3.1",
  "silero-vad>=6.2.1,<7",
]
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_vad -v
```

Expected: PASS.

- [ ] **Step 6: Update the lockfile**

Run:

```bash
uv lock
```

Expected: command exits 0 and `uv.lock` includes `silero-vad` and its resolved dependencies.

- [ ] **Step 7: Commit Task 2**

```bash
git add pyproject.toml uv.lock src/asr/vad.py tests/test_vad.py
git commit -m "feat: add silero vad preprocessor"
```

---

### Task 3: Pipeline VAD Integration

**Files:**
- Modify: `src/asr/pipeline.py`
- Modify: `tests/test_pipeline.py`

- [ ] **Step 1: Add failing pipeline tests for VAD routing**

Add these imports to `tests/test_pipeline.py`:

```python
from asr.vad import DEFAULT_VAD_CONFIG, SpeechPlan, SpeechSpan, SuperChunk, failed_speech_plan
```

Add these helpers after `IdentityPreparer`:

```python
class FakeVadPreprocessor:
    def __init__(self, plan: SpeechPlan) -> None:
        self.plan = plan
        self.calls: list[Path] = []

    def build_plan(self, audio_path: Path) -> SpeechPlan:
        self.calls.append(audio_path)
        return self.plan


class PlanAwareProvider:
    name = "plan-aware"

    def __init__(self) -> None:
        self.received_plan: SpeechPlan | None = None
        self.calls: list[Path] = []

    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
    ) -> TranscriptionDocument:
        self.calls.append(audio_path)
        self.received_plan = speech_plan
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            source_media={"provider_metadata": {"processing_strategy": "fake"}},
            segments=[],
        )


class LegacyProvider:
    name = "legacy"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        self.calls.append(audio_path)
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[],
        )


class ExplodingProvider:
    name = "explode"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        raise AssertionError("provider should not be called when VAD found no speech")
```

Add these tests to `PipelineTest`:

```python
    def test_pipeline_passes_ok_speech_plan_to_plan_aware_provider(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=100.0,
            raw_spans=[SpeechSpan(start=10.0, end=12.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=10.0,
                    speech_end=12.0,
                    chunk_start=6.0,
                    chunk_end=16.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )
        provider = PlanAwareProvider()
        vad = FakeVadPreprocessor(speech_plan)

        document = process_media_file(
            source_path=Path("clip.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=vad,
        )

        self.assertIs(provider.received_plan, speech_plan)
        self.assertEqual(vad.calls, [Path("clip.wav")])
        self.assertEqual(document.source_media["vad"]["status"], "ok")
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 1)
        self.assertEqual(document.source_media["prepared_audio_path"], "clip.wav")

    def test_pipeline_skips_provider_when_vad_finds_no_speech(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=90.0,
            raw_spans=[],
            super_chunks=[],
            config=DEFAULT_VAD_CONFIG,
        )

        document = process_media_file(
            source_path=Path("quiet.mp4"),
            provider=ExplodingProvider(),
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
        )

        self.assertEqual(document.provider_name, "explode")
        self.assertEqual(document.segments, [])
        self.assertEqual(document.source_path, "quiet.mp4")
        self.assertEqual(document.source_media["vad"]["status"], "ok")
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 0)

    def test_pipeline_falls_back_without_speech_plan_when_vad_fails(self) -> None:
        failed_plan = failed_speech_plan(
            duration_sec=5.0,
            error="backend unavailable",
            config=DEFAULT_VAD_CONFIG,
        )
        provider = LegacyProvider()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(failed_plan),
        )

        self.assertEqual(provider.calls, [Path("demo.wav")])
        self.assertEqual(document.provider_name, "legacy")
        self.assertEqual(document.source_media["vad"]["status"], "failed")
        self.assertIn("backend unavailable", document.source_media["vad"]["error"])

    def test_pipeline_can_disable_vad_explicitly(self) -> None:
        provider = LegacyProvider()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_enabled=False,
        )

        self.assertEqual(provider.calls, [Path("demo.wav")])
        self.assertEqual(document.source_media["vad"]["status"], "disabled")
        self.assertFalse(document.source_media["vad"]["enabled"])
```

Update `PipelineObservabilityTest.test_pipeline_emits_prepare_then_transcribe_steps` so it uses a fake successful speech plan:

```python
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=30.0,
            raw_spans=[SpeechSpan(start=1.0, end=2.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=1.0,
                    speech_end=2.0,
                    chunk_start=0.0,
                    chunk_end=6.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )
```

Then pass the fake VAD preprocessor to `process_media_file()` in that test:

```python
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
```

Update the expected events to:

```python
        self.assertEqual(
            events,
            [
                ("step_start", "prepare"),
                ("step_end", "prepare"),
                ("step_start", "preprocess_vad"),
                ("step_end", "preprocess_vad"),
                ("step_start", "transcribe"),
                ("step_end", "transcribe"),
            ],
        )
        vad_end = next(
            event
            for event in observer.events
            if event.event_type == "step_end" and event.step == "preprocess_vad"
        )
        self.assertEqual(vad_end.meta["status"], "ok")
        self.assertEqual(vad_end.meta["raw_span_count"], 1)
        self.assertEqual(vad_end.meta["super_chunk_count"], 1)
```

- [ ] **Step 2: Run pipeline tests to verify they fail**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_pipeline -v
```

Expected: FAIL because `process_media_file()` has no `vad_preprocessor` or `vad_enabled` parameters and does not emit `preprocess_vad`.

- [ ] **Step 3: Implement VAD-aware pipeline routing**

Replace `src/asr/pipeline.py` with:

```python
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
    return "speech_plan" in signature.parameters


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
    if observer is None:
        return preprocessor.build_plan(prepared_path)

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

    end_meta = dict(start_meta)
    end_meta.update(_speech_plan_event_meta(plan))
    observer.on_event(
        ObservabilityEvent(
            event_type="step_end",
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
```

- [ ] **Step 4: Run pipeline tests to verify they pass**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_pipeline -v
```

Expected: PASS.

- [ ] **Step 5: Run focused regression tests**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_pipeline tests.test_exporters -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/asr/pipeline.py tests/test_pipeline.py
git commit -m "feat: run vad before provider transcription"
```

---

### Task 4: CLI Flag And Public Dependency Wiring

**Files:**
- Modify: `src/asr/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `README.md`

- [ ] **Step 1: Add failing CLI tests for default VAD and `--no-vad`**

Add this parser test to `CliParserTest` in `tests/test_cli.py`:

```python
    def test_no_vad_flag_is_opt_out(self) -> None:
        parser = build_parser()
        default_args = parser.parse_args([])
        disabled_args = parser.parse_args(["--no-vad", "demo.mp4"])

        self.assertFalse(default_args.no_vad)
        self.assertTrue(disabled_args.no_vad)
```

Add these integration tests to `CliObservabilityIntegrationTest`:

```python
    @patch("asr.cli.discover_cli_sources")
    @patch("asr.cli.run_environment_preflight")
    @patch("asr.cli.process_media_file")
    def test_main_enables_vad_by_default(
        self,
        mock_process,
        mock_preflight,
        mock_discover,
    ) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "demo.mov"
            source.write_text("x", encoding="utf-8")
            output_root = Path(tmp) / "outputs"
            mock_discover.return_value = [(source, Path(tmp))]
            mock_preflight.return_value = (True, "")
            mock_process.return_value = TranscriptionDocument(
                source_path=str(source.with_suffix(".wav")),
                provider_name="fake",
                source_media={"vad": {"status": "ok"}},
                segments=[],
            )

            exit_code = main([str(source), "--output-dir", str(output_root)])

        self.assertEqual(exit_code, 0)
        self.assertTrue(mock_process.call_args.kwargs["vad_enabled"])

    @patch("asr.cli.discover_cli_sources")
    @patch("asr.cli.run_environment_preflight")
    @patch("asr.cli.process_media_file")
    def test_main_passes_vad_disabled_when_no_vad_is_used(
        self,
        mock_process,
        mock_preflight,
        mock_discover,
    ) -> None:
        with TemporaryDirectory() as tmp:
            source = Path(tmp) / "demo.mov"
            source.write_text("x", encoding="utf-8")
            output_root = Path(tmp) / "outputs"
            mock_discover.return_value = [(source, Path(tmp))]
            mock_preflight.return_value = (True, "")
            mock_process.return_value = TranscriptionDocument(
                source_path=str(source.with_suffix(".wav")),
                provider_name="fake",
                source_media={"vad": {"status": "disabled"}},
                segments=[],
            )

            exit_code = main(["--no-vad", str(source), "--output-dir", str(output_root)])

        self.assertEqual(exit_code, 0)
        self.assertFalse(mock_process.call_args.kwargs["vad_enabled"])
```

- [ ] **Step 2: Run CLI tests to verify they fail**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_cli -v
```

Expected: FAIL because `--no-vad` does not exist and `_run_transcription()` does not pass `vad_enabled`.

- [ ] **Step 3: Add `--no-vad` to argparse compatibility parser**

In `src/asr/cli.py`, add this argument inside `build_parser()` after `--granularity`:

```python
    parser.add_argument(
        "--no-vad",
        action="store_true",
        help="Disable voice activity detection preprocessing.",
    )
```

- [ ] **Step 4: Add VAD enablement to `_run_transcription()`**

Change the `_run_transcription()` signature in `src/asr/cli.py` to:

```python
def _run_transcription(
    inputs: Sequence[str],
    recursive: bool,
    output_dir: Path | None,
    granularity: str,
    verbose: bool,
    vad_enabled: bool,
) -> int:
```

Inside `_run_transcription()`, change the `process_media_file()` call to include:

```python
                        vad_enabled=vad_enabled,
```

The full call should become:

```python
                    document = process_media_file(
                        source_path=source_path,
                        provider=provider,
                        media_preparer=media_preparer,
                        observer=observer,
                        run_id=run_id,
                        file_id=file_id,
                        vad_enabled=vad_enabled,
                    )
```

- [ ] **Step 5: Add `--no-vad` to Typer root command**

In `src/asr/cli.py`, add this parameter to `root()` after `verbose`:

```python
    no_vad: bool = typer.Option(
        False,
        "--no-vad",
        help="Disable voice activity detection preprocessing.",
    ),
```

Then update the `_run_transcription()` call in `root()`:

```python
    code = _run_transcription(
        inputs=inputs or [],
        recursive=recursive,
        output_dir=output_dir,
        granularity=granularity,
        verbose=verbose,
        vad_enabled=not no_vad,
    )
```

- [ ] **Step 6: Document VAD behavior in README**

In `README.md`, add this subsection near the existing CLI usage options:

````markdown
### VAD preprocessing

VAD preprocessing is enabled by default. `easr` first scans prepared audio with
Silero VAD to find high-recall speech candidates, merges them into padded
super-chunks, and asks the provider to process only those ranges. Final subtitle
timestamps remain on the original media timeline.

Use `--no-vad` to restore full-duration provider processing:

```bash
uv run --python 3.14 --extra mlx easr ./demo.mp4 --no-vad
```

If VAD fails, transcription falls back to the full-duration provider path. If VAD
successfully finds no speech, `easr` writes successful empty subtitle outputs.
````

- [ ] **Step 7: Run CLI tests to verify they pass**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_cli -v
```

Expected: PASS.

- [ ] **Step 8: Commit Task 4**

```bash
git add src/asr/cli.py tests/test_cli.py README.md
git commit -m "feat: add no-vad cli option"
```

---

### Task 5: Qwen Provider Super-Chunk Consumption

**Files:**
- Modify: `src/asr/providers/windowing.py`
- Modify: `src/asr/providers/qwen_mlx.py`
- Modify: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Add failing provider tests for speech-plan windowing**

Add this import to `tests/test_qwen_provider_windowed.py`:

```python
from asr.vad import DEFAULT_VAD_CONFIG, SpeechPlan, SpeechSpan, SuperChunk
```

Add this helper method inside `QwenProviderWindowedTest`:

```python
    def _speech_plan(self, chunks: list[SuperChunk], duration_sec: float = 400.0) -> SpeechPlan:
        return SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=duration_sec,
            raw_spans=[
                SpeechSpan(start=chunk.speech_start, end=chunk.speech_end)
                for chunk in chunks
            ],
            super_chunks=chunks,
            config=DEFAULT_VAD_CONFIG,
        )
```

Add these tests to `QwenProviderWindowedTest`:

```python
    def test_provider_processes_vad_super_chunks_on_global_timeline(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("first chunk", language="en"),
                FakeChunk("second chunk", language="en"),
            ],
            align_responses=[
                [
                    FakeChunk("first", start_time=0.00, end_time=0.40),
                    FakeChunk("chunk", start_time=0.41, end_time=0.90),
                ],
                [
                    FakeChunk("second", start_time=0.00, end_time=0.50),
                    FakeChunk("chunk", start_time=0.51, end_time=1.00),
                ],
            ],
        )
        plan = self._speech_plan(
            [
                SuperChunk(0, 105.0, 120.0, 100.0, 130.0, 1),
                SuperChunk(1, 285.0, 300.0, 280.0, 310.0, 1),
            ]
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)

        metadata = doc.source_media["provider_metadata"]
        diagnostics = metadata["window_diagnostics"]
        self.assertEqual(metadata["processing_strategy"], "vad_super_chunk_windowed_bounded_alignment")
        self.assertEqual(metadata["super_chunk_count"], 2)
        self.assertEqual([item["super_chunk_index"] for item in diagnostics], [0, 1])
        self.assertEqual(len(asr_model.calls), 2)
        self.assertEqual(len(align_model.calls), 2)
        self.assertEqual([round(token.start_time, 2) for segment in doc.segments for token in segment.tokens], [100.0, 100.41, 280.0, 280.51])

    def test_provider_splits_long_super_chunk_with_existing_hard_window_budget(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("alpha", language="en"),
                FakeChunk("beta", language="en"),
                FakeChunk("gamma", language="en"),
            ],
            align_responses=[
                [FakeChunk("alpha", start_time=0.0, end_time=0.4)],
                [FakeChunk("beta", start_time=0.0, end_time=0.4)],
                [FakeChunk("gamma", start_time=0.0, end_time=0.4)],
            ],
        )
        plan = self._speech_plan(
            [SuperChunk(0, 20.0, 330.0, 10.0, 350.0, 1)],
            duration_sec=400.0,
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)

        diagnostics = doc.source_media["provider_metadata"]["window_diagnostics"]
        self.assertGreater(len(diagnostics), 1)
        self.assertEqual(len(asr_model.calls), len(diagnostics))
        self.assertEqual(len(align_model.calls), len(diagnostics))
        for diagnostic in diagnostics:
            self.assertLessEqual(
                diagnostic["context_end"] - diagnostic["context_start"],
                provider.window_config.max_alignment_window_sec,
            )
            self.assertEqual(diagnostic["super_chunk_index"], 0)
```

- [ ] **Step 2: Run provider tests to verify they fail**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_qwen_provider_windowed -v
```

Expected: FAIL because `QwenMlxProvider.transcribe()` does not accept `speech_plan`.

- [ ] **Step 3: Add optional super-chunk index to alignment windows**

In `src/asr/providers/windowing.py`, modify `AlignmentWindow`:

```python
@dataclass(frozen=True)
class AlignmentWindow:
    """A planned alignment window and its surrounding context."""

    index: int
    core_start: float
    core_end: float
    context_start: float
    context_end: float
    super_chunk_index: int | None = None
```

- [ ] **Step 4: Update Qwen provider imports and transcribe signature**

In `src/asr/providers/qwen_mlx.py`, add:

```python
from asr.vad import SpeechPlan
```

Change the `transcribe()` signature:

```python
    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
    ) -> TranscriptionDocument:
```

Inside `transcribe()`, change:

```python
                windows = self._plan_windows(total_duration_sec)
```

to:

```python
                windows = self._plan_windows(
                    total_duration_sec,
                    speech_plan=speech_plan,
                )
```

Change both `_build_document(...)` calls in `transcribe()` to include:

```python
                    speech_plan_used=self._uses_speech_plan(speech_plan),
                    super_chunk_count=self._super_chunk_count(speech_plan),
```

For the non-empty path, include the same keyword arguments:

```python
                speech_plan_used=self._uses_speech_plan(speech_plan),
                super_chunk_count=self._super_chunk_count(speech_plan),
```

- [ ] **Step 5: Implement super-chunk-aware planning helpers**

Replace `_plan_windows()` in `src/asr/providers/qwen_mlx.py` with:

```python
    def _plan_windows(
        self,
        total_duration_sec: float,
        *,
        speech_plan: SpeechPlan | None = None,
    ) -> List[AlignmentWindow]:
        planner = WindowPlanner(
            self.window_config,
            anchor_resolver=self._resolve_silence_anchor,
        )
        if not self._uses_speech_plan(speech_plan):
            return planner.plan(total_duration_sec)

        assert speech_plan is not None
        windows: List[AlignmentWindow] = []
        next_index = 0
        for chunk in speech_plan.super_chunks:
            chunk_duration = max(0.0, chunk.chunk_end - chunk.chunk_start)
            if chunk_duration <= 0.0:
                continue
            for local_window in planner.plan(chunk_duration):
                windows.append(
                    AlignmentWindow(
                        index=next_index,
                        core_start=chunk.chunk_start + local_window.core_start,
                        core_end=chunk.chunk_start + local_window.core_end,
                        context_start=chunk.chunk_start + local_window.context_start,
                        context_end=chunk.chunk_start + local_window.context_end,
                        super_chunk_index=chunk.index,
                    )
                )
                next_index += 1
        return windows
```

Add these helper methods near `_plan_windows()`:

```python
    def _uses_speech_plan(self, speech_plan: SpeechPlan | None) -> bool:
        return (
            speech_plan is not None
            and speech_plan.status == "ok"
            and bool(speech_plan.super_chunks)
        )

    def _super_chunk_count(self, speech_plan: SpeechPlan | None) -> int:
        if speech_plan is None or speech_plan.status != "ok":
            return 0
        return len(speech_plan.super_chunks)
```

- [ ] **Step 6: Include super-chunk metadata in provider events and diagnostics**

In `_execute_window()`, replace the inline `meta={...}` with:

```python
            meta = {"window_index": window_index, "window_count": window_count}
            if window.super_chunk_index is not None:
                meta["super_chunk_index"] = window.super_chunk_index
            with observe_step(
                self._observer,
                run_id=self._observer_run_id,
                file_id=self._observer_file_id,
                source_path=self._observer_source_path,
                step="provider_window",
                meta=meta,
            ):
                return self._transcribe_window(audio_path, window)
```

Change `_build_document()` signature:

```python
    def _build_document(
        self,
        *,
        audio_path: Path,
        total_duration_sec: float,
        windows: List[AlignmentWindow],
        window_runs: List[WindowRun],
        segments: List[Segment],
        speech_plan_used: bool = False,
        super_chunk_count: int = 0,
    ) -> TranscriptionDocument:
```

In provider metadata, replace:

```python
            "processing_strategy": "windowed_bounded_alignment",
```

with:

```python
            "processing_strategy": (
                "vad_super_chunk_windowed_bounded_alignment"
                if speech_plan_used
                else "windowed_bounded_alignment"
            ),
            "super_chunk_count": super_chunk_count,
```

In `_build_window_diagnostic()`, add:

```python
        if window_run.window.super_chunk_index is not None:
            diagnostic["super_chunk_index"] = window_run.window.super_chunk_index
```

immediately after the initial `diagnostic = {...}` block.

- [ ] **Step 7: Run provider tests**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_qwen_provider_windowed tests.test_windowing -v
```

Expected: PASS.

- [ ] **Step 8: Run pipeline/provider focused regression**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_pipeline tests.test_qwen_provider_windowed tests.test_window_merge tests.test_quality -v
```

Expected: PASS.

- [ ] **Step 9: Commit Task 5**

```bash
git add src/asr/providers/windowing.py src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "feat: process vad super chunks in qwen provider"
```

---

### Task 6: Metadata, Metrics, Exporter Contract, And Final Verification

**Files:**
- Modify: `src/asr/observability/metrics.py`
- Modify: `tests/test_observability.py`
- Modify: `tests/test_exporters.py`

- [ ] **Step 1: Add failing metrics test for VAD step metadata**

Add this test to `MetricsCollectorObserverTest` in `tests/test_observability.py`:

```python
    def test_writes_vad_step_metadata(self) -> None:
        collector = MetricsCollectorObserver()
        collector.on_event(ObservabilityEvent(event_type="run_start", run_id="run-1", perf_counter=0.0))
        collector.on_event(
            ObservabilityEvent(
                event_type="file_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="preprocess_vad",
                perf_counter=1.0,
                meta={"enabled": True},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="preprocess_vad",
                perf_counter=1.4,
                meta={
                    "status": "ok",
                    "raw_span_count": 2,
                    "super_chunk_count": 1,
                },
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="file_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"status": "ok"},
            )
        )
        collector.on_event(ObservabilityEvent(event_type="run_end", run_id="run-1", perf_counter=2.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "demo.metrics.json"
            collector.write_file_metrics(file_id="1", target_path=target)
            payload = json.loads(target.read_text(encoding="utf-8"))

        vad_step = payload["steps"][0]
        self.assertEqual(vad_step["name"], "preprocess_vad")
        self.assertEqual(vad_step["meta"]["status"], "ok")
        self.assertEqual(vad_step["meta"]["raw_span_count"], 2)
        self.assertEqual(vad_step["meta"]["super_chunk_count"], 1)
```

- [ ] **Step 2: Add exporter test for VAD metadata passthrough**

Add this test to `ExporterTest` in `tests/test_exporters.py`:

```python
    def test_json_preserves_vad_metadata_without_affecting_subtitles(self) -> None:
        document = TranscriptionDocument(
            source_path="quiet.wav",
            provider_name="fake",
            source_media={
                "vad": {
                    "enabled": True,
                    "status": "ok",
                    "duration_sec": 60.0,
                    "raw_span_count": 0,
                    "super_chunk_count": 0,
                    "config": {"threshold": 0.25},
                    "super_chunks": [],
                }
            },
            segments=[],
        )

        srt_text = render_srt(document)
        vtt_text = render_vtt(document)
        payload = json.loads(render_json(document))

        self.assertEqual(srt_text, "")
        self.assertEqual(vtt_text, "WEBVTT\n")
        self.assertEqual(payload["source_media"]["vad"]["status"], "ok")
        self.assertEqual(payload["source_media"]["vad"]["super_chunk_count"], 0)
```

- [ ] **Step 3: Run metadata tests to verify at least metrics test fails**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_observability tests.test_exporters -v
```

Expected: FAIL because metrics `_serialize_step()` does not include `meta`.

- [ ] **Step 4: Preserve metrics step metadata**

In `src/asr/observability/metrics.py`, replace `_serialize_step()` with:

```python
    def _serialize_step(self, step: dict) -> dict:
        return {
            "name": step["name"],
            "status": step["status"],
            "started_at": step["started_at"],
            "ended_at": step["ended_at"],
            "duration_ms": step["duration_ms"],
            "error": step["error"],
            "meta": dict(step.get("meta", {})),
        }
```

- [ ] **Step 5: Run metadata tests to verify they pass**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_observability tests.test_exporters -v
```

Expected: PASS.

- [ ] **Step 6: Run all unit tests**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
```

Expected: PASS with all tests green.

- [ ] **Step 7: Run package help smoke test**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 easr --help
```

Expected: exit code 0 and help output includes `--no-vad`.

- [ ] **Step 8: Commit Task 6**

```bash
git add src/asr/observability/metrics.py tests/test_observability.py tests/test_exporters.py
git commit -m "test: preserve vad metadata in outputs"
```

---

## Final Verification

- [ ] **Step 1: Confirm working tree contains only intended implementation changes**

Run:

```bash
git status --short
```

Expected: either clean or only intended files from the current task before the final commit.

- [ ] **Step 2: Run full test suite**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
```

Expected: PASS.

- [ ] **Step 3: Verify CLI help**

Run:

```bash
PYTHONPATH=src uv run --python 3.14 easr --help
```

Expected: PASS and `--no-vad` is listed.

- [ ] **Step 4: Inspect commits**

Run:

```bash
git log --oneline -6
```

Expected: shows one commit per task, ending with the metadata/export verification commit.

---

## Self-Review

- Spec coverage: Tasks 1 and 2 cover VAD data model, high-recall defaults, Silero scan, sanitization, and super-chunk construction. Task 3 covers pipeline placement, no-speech short-circuit, failure fallback, and VAD metadata. Task 4 covers default public behavior, `--no-vad`, dependency wiring, and README documentation. Task 5 covers Qwen provider speech-plan consumption, bounded alignment inside super-chunks, global timestamp offsets, and diagnostics. Task 6 covers metrics and exporter contract preservation.
- Scope check: This is one focused preprocessing feature. It does not add public VAD tuning flags, diarization, translation, provider selection, or VAD-owned subtitle segmentation.
- Type consistency: The plan consistently uses `VadConfig`, `SpeechSpan`, `SuperChunk`, `SpeechPlan`, `SileroVadPreprocessor`, `speech_plan`, and `preprocess_vad`.
- Verification check: Every code-producing task includes a failing test command, a passing focused test command, and a commit step.
