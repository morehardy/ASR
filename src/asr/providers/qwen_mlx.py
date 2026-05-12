"""MLX provider for the planned Qwen3 ASR + ForcedAligner backend."""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional

from asr.models import Segment, Token, TranscriptionDocument
from asr.observability.observer import Observer
from asr.observability.timing import observe_step
from asr.providers.authority import (
    ProjectedToken,
    build_transcript_tokens,
    project_timing_onto_transcript_detailed,
    repair_unmatched_timings,
)
from asr.providers.media_probe import parse_silence_anchors, probe_duration_sec
from asr.providers.quality import QualityResult, QualityThresholds, evaluate_quality
from asr.providers.timing_validation import (
    TimingValidationPolicy,
    token_crosses_non_speech_gap,
    token_has_suspicious_duration,
)
from asr.providers.window_merge import (
    WindowSpan,
    merge_adjacent_windows,
    token_overlaps_core,
)
from asr.providers.windowing import AlignmentWindow, WindowBudgetConfig, WindowPlanner
from asr.vad import SpeechPlan, SpeechSpan


DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"


@dataclass(frozen=True, slots=True)
class WindowDisplayBounds:
    start_time: float
    end_time: float
    alignment_unit_index: int


@dataclass(slots=True)
class WindowRun:
    window: AlignmentWindow
    text: str = ""
    language: Optional[str] = None
    tokens: List[Token] = field(default_factory=list)
    core_tokens: List[Token] = field(default_factory=list)
    left_overlap_tokens: List[Token] = field(default_factory=list)
    right_overlap_tokens: List[Token] = field(default_factory=list)
    core_text: str = ""
    quality: Optional[QualityResult] = None
    error: Optional[str] = None
    projected_tokens: List[ProjectedToken] = field(default_factory=list)
    timing_source_counts: dict[str, int] = field(default_factory=dict)
    has_timing_anchor: bool = False
    display_bounds: WindowDisplayBounds | None = None
    speech_spans: List[SpeechSpan] = field(default_factory=list)


@dataclass
class QwenMlxProvider:
    """Provider that wraps the planned MLX Qwen3 model pair."""

    asr_model_id: str = DEFAULT_ASR_MODEL
    aligner_model_id: str = DEFAULT_ALIGNER_MODEL
    name: str = "qwen-mlx"
    window_config: WindowBudgetConfig = field(default_factory=WindowBudgetConfig)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)
    timing_validation_policy: TimingValidationPolicy = field(
        default_factory=TimingValidationPolicy
    )
    vad_display_lead_pad_sec: float = 0.20
    vad_display_tail_pad_sec: float = 0.35

    def __post_init__(self) -> None:
        self._asr_model: Optional[Any] = None
        self._aligner_model: Optional[Any] = None
        self._active_audio_path: Optional[Path] = None
        self._silence_anchor_cache: dict[str, List[float]] = {}
        self._context_clip_dir: Optional[Path] = None
        self._observer: Optional[Observer] = None
        self._observer_run_id: str = "run-unknown"
        self._observer_file_id: Optional[str] = None
        self._observer_source_path: Optional[str] = None

    def bind_observer(
        self,
        *,
        observer: Observer,
        run_id: str,
        file_id: str,
        source_path: str,
    ) -> None:
        self._observer = observer
        self._observer_run_id = run_id
        self._observer_file_id = file_id
        self._observer_source_path = source_path

    def clear_observer(self) -> None:
        self._observer = None
        self._observer_run_id = "run-unknown"
        self._observer_file_id = None
        self._observer_source_path = None

    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
    ) -> TranscriptionDocument:
        load = self._load_backend()
        self._asr_model = self._asr_model or load(self.asr_model_id)
        self._aligner_model = self._aligner_model or load(self.aligner_model_id)

        with observe_step(
            self._observer,
            run_id=self._observer_run_id,
            file_id=self._observer_file_id,
            source_path=self._observer_source_path,
            step="provider_plan_windows",
        ):
            total_duration_sec = self._probe_duration_sec(audio_path)
            if (
                self._uses_speech_plan(speech_plan)
                and math.isfinite(speech_plan.duration_sec)
            ):
                total_duration_sec = max(total_duration_sec, speech_plan.duration_sec)
            self._active_audio_path = audio_path
            try:
                windows = self._plan_windows(total_duration_sec, speech_plan=speech_plan)
            finally:
                self._active_audio_path = None

        self._begin_context_windowing()
        speech_plan_used = self._uses_speech_plan(speech_plan)
        alignment_unit_count = self._alignment_unit_count(speech_plan)
        try:
            if not windows:
                return self._build_document(
                    audio_path=audio_path,
                    total_duration_sec=total_duration_sec,
                    windows=windows,
                    window_runs=[],
                    segments=[],
                    speech_plan_used=speech_plan_used,
                    alignment_unit_count=alignment_unit_count,
                )

            window_runs: List[WindowRun] = []
            for index, window in enumerate(windows, start=1):
                window_runs.append(
                    self._execute_window(
                        audio_path,
                        window,
                        window_index=index,
                        window_count=len(windows),
                        display_bounds=self._display_bounds_for_window(
                            window,
                            speech_plan=speech_plan,
                            total_duration_sec=total_duration_sec,
                        ),
                        speech_spans=self._speech_spans_for_window(
                            window,
                            speech_plan=speech_plan,
                        ),
                    )
                )
            self._evaluate_window_qualities(window_runs)
            self._raise_if_all_windows_failed(window_runs)

            with observe_step(
                self._observer,
                run_id=self._observer_run_id,
                file_id=self._observer_file_id,
                source_path=self._observer_source_path,
                step="provider_merge",
            ):
                merged_tokens = self._merge_window_runs(window_runs)
                segments = self._tokens_to_segments(merged_tokens)
                fallback_segments = self._fallback_segments_from_windows(
                    [
                        run
                        for run in window_runs
                        if self._needs_text_fallback(run)
                    ]
                )
                fallback_segments = self._append_segments(
                    fallback_segments,
                    self._unresolved_fallback_segments_from_windows(window_runs),
                )
                if fallback_segments:
                    segments = self._append_segments(segments, fallback_segments)
                if not segments:
                    segments = self._fallback_segments_from_windows(window_runs)
                segments = self._stabilize_segment_boundaries(
                    segments,
                    total_duration_sec=total_duration_sec,
                    display_bounds=[
                        run.display_bounds
                        for run in window_runs
                        if run.display_bounds is not None
                    ],
                )

            return self._build_document(
                audio_path=audio_path,
                total_duration_sec=total_duration_sec,
                windows=windows,
                window_runs=window_runs,
                segments=segments,
                speech_plan_used=speech_plan_used,
                alignment_unit_count=alignment_unit_count,
            )
        finally:
            self._cleanup_context_windowing()

    def _plan_windows(
        self,
        total_duration_sec: float,
        *,
        speech_plan: SpeechPlan | None = None,
    ) -> List[AlignmentWindow]:
        if not self._uses_speech_plan(speech_plan):
            planner = WindowPlanner(
                self.window_config,
                anchor_resolver=self._resolve_silence_anchor,
            )
            return planner.plan(total_duration_sec)

        windows: List[AlignmentWindow] = []
        for unit in speech_plan.alignment_units:
            unit_start, unit_end = self._clamped_alignment_unit_bounds(
                unit.input_start,
                unit.input_end,
                total_duration_sec,
            )
            if unit_start is None or unit_end is None:
                continue

            planner = WindowPlanner(
                self.window_config,
                anchor_resolver=self._unit_anchor_resolver(unit_start),
            )
            local_windows = planner.plan(unit_end - unit_start)
            for local_window in local_windows:
                offset = unit_start
                windows.append(
                    AlignmentWindow(
                        index=len(windows),
                        core_start=local_window.core_start + offset,
                        core_end=local_window.core_end + offset,
                        context_start=local_window.context_start + offset,
                        context_end=local_window.context_end + offset,
                        alignment_unit_index=unit.index,
                    )
                )
        return windows

    def _unit_anchor_resolver(
        self, unit_start: float
    ) -> Callable[[float, float, float], Optional[float]]:
        def resolve(
            target_split_sec: float,
            search_start_sec: float,
            search_end_sec: float,
        ) -> Optional[float]:
            resolved = self._resolve_silence_anchor(
                target_split_sec + unit_start,
                search_start_sec + unit_start,
                search_end_sec + unit_start,
            )
            if resolved is None:
                return None
            return resolved - unit_start

        return resolve

    def _clamped_alignment_unit_bounds(
        self,
        input_start: float,
        input_end: float,
        total_duration_sec: float,
    ) -> tuple[float | None, float | None]:
        if not (
            math.isfinite(input_start)
            and math.isfinite(input_end)
            and math.isfinite(total_duration_sec)
        ):
            return None, None

        clamped_start = min(max(0.0, input_start), total_duration_sec)
        clamped_end = min(max(0.0, input_end), total_duration_sec)
        if clamped_end <= clamped_start:
            return None, None
        return clamped_start, clamped_end

    def _uses_speech_plan(self, speech_plan: SpeechPlan | None) -> bool:
        return (
            speech_plan is not None
            and speech_plan.status == "ok"
            and bool(speech_plan.alignment_units)
        )

    def _alignment_unit_count(self, speech_plan: SpeechPlan | None) -> int:
        if speech_plan is None or speech_plan.status != "ok":
            return 0
        return len(speech_plan.alignment_units)

    def _display_bounds_for_window(
        self,
        window: AlignmentWindow,
        *,
        speech_plan: SpeechPlan | None,
        total_duration_sec: float,
    ) -> WindowDisplayBounds | None:
        if not self._uses_speech_plan(speech_plan):
            return None
        if window.alignment_unit_index is None:
            return None
        assert speech_plan is not None
        unit = next(
            (
                candidate
                for candidate in speech_plan.alignment_units
                if candidate.index == window.alignment_unit_index
            ),
            None,
        )
        if unit is None:
            return None
        return WindowDisplayBounds(
            start_time=max(0.0, unit.speech_start - self.vad_display_lead_pad_sec),
            end_time=min(
                total_duration_sec,
                unit.speech_end + self.vad_display_tail_pad_sec,
            ),
            alignment_unit_index=unit.index,
        )

    def _speech_spans_for_window(
        self,
        window: AlignmentWindow,
        *,
        speech_plan: SpeechPlan | None,
    ) -> List[SpeechSpan]:
        if speech_plan is None or window.alignment_unit_index is None:
            return []
        unit = next(
            (
                candidate
                for candidate in speech_plan.alignment_units
                if candidate.index == window.alignment_unit_index
            ),
            None,
        )
        if unit is None:
            return []
        return [
            span
            for span in speech_plan.raw_spans
            if span.end >= unit.speech_start and span.start <= unit.speech_end
        ]

    def _transcribe_window(
        self,
        audio_path: Path,
        window: AlignmentWindow,
        *,
        display_bounds: WindowDisplayBounds | None = None,
        speech_spans: List[SpeechSpan] | None = None,
    ) -> WindowRun:
        context_input = self._context_input_path(audio_path, window)
        context_kwargs = self._context_generate_kwargs(window)
        transcription = self._asr_model.generate(context_input, **context_kwargs)
        text = getattr(transcription, "text", "").strip()
        language = self._normalize_language(getattr(transcription, "language", None))

        align_kwargs = dict(context_kwargs)
        align_kwargs["text"] = text
        if language:
            align_kwargs["language"] = language

        aligned_items = list(self._aligner_model.generate(context_input, **align_kwargs))
        aligner_tokens = [
            self._item_to_token(item, language=language)
            for item in aligned_items
            if getattr(item, "text", "").strip()
        ]
        transcript_tokens = self._build_authoritative_tokens(text, language)
        repaired_projected_tokens = repair_unmatched_timings(
            project_timing_onto_transcript_detailed(
                transcript_tokens,
                aligner_tokens,
            ),
            clip_duration_sec=max(0.0, window.context_end - window.context_start),
        )
        (
            downgraded_projected_tokens,
            prefer_next_anchor_indexes,
        ) = self._downgrade_suspicious_projected_tokens(
            repaired_projected_tokens,
            speech_spans=speech_spans or [],
            window=window,
        )
        repaired_projected_tokens = repair_unmatched_timings(
            downgraded_projected_tokens,
            clip_duration_sec=max(0.0, window.context_end - window.context_start),
            prefer_next_anchor_indexes=prefer_next_anchor_indexes,
        )
        timing_source_counts = self._timing_source_counts(repaired_projected_tokens)
        has_timing_anchor = timing_source_counts.get("aligner", 0) > 0
        global_projected_tokens = self._offset_projected_tokens(
            repaired_projected_tokens,
            window.context_start,
        )

        usable_projected_tokens = (
            self._usable_projected_tokens(global_projected_tokens)
            if has_timing_anchor
            else []
        )
        global_tokens = [projected.token for projected in usable_projected_tokens]

        left_overlap_tokens, core_tokens, right_overlap_tokens = (
            self._split_window_tokens(global_tokens, window)
        )
        core_text = self._join_tokens(core_tokens) if core_tokens else text

        return WindowRun(
            window=window,
            text=text,
            language=language,
            tokens=global_tokens,
            core_tokens=core_tokens,
            left_overlap_tokens=left_overlap_tokens,
            right_overlap_tokens=right_overlap_tokens,
            core_text=core_text,
            projected_tokens=global_projected_tokens,
            timing_source_counts=timing_source_counts,
            has_timing_anchor=has_timing_anchor,
            display_bounds=display_bounds,
            speech_spans=list(speech_spans or []),
        )

    def _execute_window(
        self,
        audio_path: Path,
        window: AlignmentWindow,
        *,
        window_index: int,
        window_count: int,
        display_bounds: WindowDisplayBounds | None = None,
        speech_spans: List[SpeechSpan] | None = None,
    ) -> WindowRun:
        meta = {"window_index": window_index, "window_count": window_count}
        if window.alignment_unit_index is not None:
            meta["alignment_unit_index"] = window.alignment_unit_index
        try:
            with observe_step(
                self._observer,
                run_id=self._observer_run_id,
                file_id=self._observer_file_id,
                source_path=self._observer_source_path,
                step="provider_window",
                meta=meta,
            ):
                return self._transcribe_window(
                    audio_path,
                    window,
                    display_bounds=display_bounds,
                    speech_spans=speech_spans,
                )
        except Exception as exc:
            return WindowRun(
                window=window,
                error=str(exc),
                display_bounds=display_bounds,
                speech_spans=list(speech_spans or []),
            )

    def _evaluate_window_qualities(self, window_runs: List[WindowRun]) -> None:
        for index, window_run in enumerate(window_runs):
            if window_run.error is not None:
                continue
            left_overlap_tokens, right_overlap_tokens = self._quality_boundary_inputs(
                window_runs,
                index,
            )
            window_run.quality = evaluate_quality(
                tokens=window_run.tokens,
                left_overlap_tokens=left_overlap_tokens,
                right_overlap_tokens=right_overlap_tokens,
                core_text=window_run.core_text or window_run.text,
                context_text=window_run.text,
                thresholds=self.quality_thresholds,
                timing_source_counts=window_run.timing_source_counts,
                has_timing_anchor=window_run.has_timing_anchor,
            )

    def _quality_boundary_inputs(
        self,
        window_runs: List[WindowRun],
        index: int,
    ) -> tuple[List[Token], List[Token]]:
        comparisons: List[tuple[List[Token], List[Token]]] = []
        current = window_runs[index]
        previous = self._adjacent_successful_neighbor(window_runs, index, step=-1)
        following = self._adjacent_successful_neighbor(window_runs, index, step=1)

        if previous is not None:
            comparisons.append(
                (previous.right_overlap_tokens, current.left_overlap_tokens)
            )
        if following is not None:
            comparisons.append(
                (current.right_overlap_tokens, following.left_overlap_tokens)
            )

        left_tokens: List[Token] = []
        right_tokens: List[Token] = []
        for left_comparison, right_comparison in comparisons:
            if not left_comparison or not right_comparison:
                continue
            left_tokens.extend(left_comparison)
            right_tokens.extend(right_comparison)

        return left_tokens, right_tokens

    def _adjacent_successful_neighbor(
        self,
        window_runs: List[WindowRun],
        index: int,
        *,
        step: int,
    ) -> Optional[WindowRun]:
        cursor = index + step
        if cursor < 0 or cursor >= len(window_runs):
            return None

        candidate = window_runs[cursor]
        if candidate.error is not None or not candidate.tokens:
            return None

        expected_index = window_runs[index].window.index + step
        if candidate.window.index != expected_index:
            return None
        if not self._same_alignment_unit_scope(window_runs[index], candidate):
            return None

        return candidate

    def _same_alignment_unit_scope(self, left: WindowRun, right: WindowRun) -> bool:
        return left.window.alignment_unit_index == right.window.alignment_unit_index

    def _raise_if_all_windows_failed(self, window_runs: List[WindowRun]) -> None:
        if any(window_run.error is None for window_run in window_runs):
            return

        error_details = ", ".join(
            f"window {window_run.window.index}: {window_run.error or 'unknown error'}"
            for window_run in window_runs
        )
        raise RuntimeError(f"All transcription windows failed: {error_details}")

    def _load_backend(self):
        try:
            from mlx_audio.stt import load
        except ImportError as exc:
            raise RuntimeError(
                "mlx-audio is required for the default Qwen MLX provider. "
                "Install the optional dependency set with `pip install 'echoalign-asr-mlx[mlx]'` "
                "(published package) or `pip install '.[mlx]'` from a source checkout."
            ) from exc
        return load

    def _probe_duration_sec(self, audio_path: Path) -> float:
        return probe_duration_sec(audio_path)

    def _resolve_silence_anchor(
        self, target_split_sec: float, search_start_sec: float, search_end_sec: float
    ) -> Optional[float]:
        if self._active_audio_path is None:
            return None

        anchors = self._silence_anchors_for_audio(self._active_audio_path)
        bounded_anchors = [
            anchor
            for anchor in anchors
            if search_start_sec <= anchor <= search_end_sec
        ]
        if not bounded_anchors:
            return None

        return min(
            bounded_anchors,
            key=lambda anchor: (abs(anchor - target_split_sec), anchor),
        )

    def _silence_anchors_for_audio(self, audio_path: Path) -> List[float]:
        cache_key = str(audio_path)
        cached = self._silence_anchor_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-nostats",
                    "-i",
                    str(audio_path),
                    "-af",
                    "silencedetect=n=-35dB:d=0.3",
                    "-f",
                    "null",
                    "-",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            anchors = parse_silence_anchors(result.stderr or "")
        except (OSError, ValueError):
            anchors = []

        self._silence_anchor_cache[cache_key] = anchors
        return anchors

    def _begin_context_windowing(self) -> None:
        self._cleanup_context_windowing()
        self._context_clip_dir = Path(tempfile.mkdtemp(prefix="asr-window-"))

    def _cleanup_context_windowing(self) -> None:
        if self._context_clip_dir is None:
            return
        shutil.rmtree(self._context_clip_dir, ignore_errors=True)
        self._context_clip_dir = None

    def _context_input_path(self, audio_path: Path, window: AlignmentWindow) -> str:
        source = audio_path.expanduser().resolve()
        if not source.exists():
            return str(audio_path)

        return str(self._materialize_window_clip(source, window))

    def _materialize_window_clip(
        self,
        audio_path: Path,
        window: AlignmentWindow,
    ) -> Path:
        if self._context_clip_dir is None:
            self._context_clip_dir = Path(tempfile.mkdtemp(prefix="asr-window-"))

        clip_path = self._context_clip_dir / f"window-{window.index:04d}.wav"
        if clip_path.exists():
            return clip_path

        command = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{window.context_start:.3f}",
            "-to",
            f"{window.context_end:.3f}",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(clip_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required but was not found on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                exc.stderr.strip() or "ffmpeg failed to extract bounded window audio."
            ) from exc

        return clip_path

    def _context_generate_kwargs(self, window: AlignmentWindow) -> dict[str, float]:
        _ = window
        return {}

    def _build_authoritative_tokens(
        self,
        text: str,
        language: Optional[str],
    ) -> List[Token]:
        transcript_tokens = build_transcript_tokens(text, language=language)
        return [
            Token(
                text=token.text,
                start_time=token.start_time,
                end_time=token.end_time,
                unit=self._infer_unit(text=token.text, language=language),
                language=token.language,
            )
            for token in transcript_tokens
        ]

    def _offset_tokens(self, tokens: Iterable[Token], offset_sec: float) -> List[Token]:
        return [
            Token(
                text=token.text,
                start_time=token.start_time + offset_sec,
                end_time=token.end_time + offset_sec,
                unit=token.unit,
                language=token.language,
            )
            for token in tokens
        ]

    def _timing_source_counts(
        self,
        projected_tokens: Iterable[ProjectedToken],
    ) -> dict[str, int]:
        counts = {"aligner": 0, "estimated": 0, "unresolved": 0}
        for projected in projected_tokens:
            counts[projected.timing_source] = counts.get(projected.timing_source, 0) + 1
        return counts

    def _downgrade_suspicious_projected_tokens(
        self,
        projected_tokens: List[ProjectedToken],
        *,
        speech_spans: List[SpeechSpan],
        window: AlignmentWindow,
    ) -> tuple[List[ProjectedToken], set[int]]:
        if not projected_tokens:
            return [], set()
        context_duration = max(0.0, window.context_end - window.context_start)
        local_spans: List[SpeechSpan] = []
        for span in speech_spans:
            start = min(context_duration, max(0.0, span.start - window.context_start))
            end = min(context_duration, max(0.0, span.end - window.context_start))
            if end <= start:
                continue
            local_spans.append(
                SpeechSpan(start=start, end=end, confidence=span.confidence)
            )
        downgraded: List[ProjectedToken] = []
        prefer_next_anchor_indexes: set[int] = set()
        for index, projected in enumerate(projected_tokens):
            token = projected.token
            suspicious = (
                projected.timing_source == "aligner"
                and (
                    token_has_suspicious_duration(token, self.timing_validation_policy)
                    or token_crosses_non_speech_gap(token, local_spans)
                )
            )
            if not suspicious:
                downgraded.append(projected)
                continue
            if (
                projected.transcript_index is not None
                and self._should_repair_suspicious_token_from_next_anchor(
                    projected_tokens,
                    index,
                )
            ):
                prefer_next_anchor_indexes.add(projected.transcript_index)
            downgraded.append(
                ProjectedToken(
                    Token(
                        text=token.text,
                        start_time=0.0,
                        end_time=0.0,
                        unit=token.unit,
                        language=token.language,
                    ),
                    "unresolved",
                    projected.aligner_index,
                    projected.transcript_index,
                )
            )
        return downgraded, prefer_next_anchor_indexes

    def _should_repair_suspicious_token_from_next_anchor(
        self,
        projected_tokens: List[ProjectedToken],
        index: int,
    ) -> bool:
        next_projected = self._nearest_projected_with_timing(
            projected_tokens,
            start_index=index + 1,
            step=1,
        )
        if next_projected is None:
            return False
        previous = self._nearest_projected_with_timing(
            projected_tokens,
            start_index=index - 1,
            step=-1,
        )
        if previous is None:
            return True
        token = projected_tokens[index].token
        if self._ends_segment(previous.token.text):
            return True
        return token.start_time - previous.token.end_time >= 1.0

    def _nearest_projected_with_timing(
        self,
        projected_tokens: List[ProjectedToken],
        *,
        start_index: int,
        step: int,
    ) -> ProjectedToken | None:
        index = start_index
        while 0 <= index < len(projected_tokens):
            projected = projected_tokens[index]
            if projected.timing_source in {"aligner", "estimated"}:
                return projected
            index += step
        return None

    def _offset_projected_tokens(
        self,
        projected_tokens: Iterable[ProjectedToken],
        offset_sec: float,
    ) -> List[ProjectedToken]:
        return [
            ProjectedToken(
                Token(
                    text=projected.token.text,
                    start_time=projected.token.start_time + offset_sec,
                    end_time=projected.token.end_time + offset_sec,
                    unit=projected.token.unit,
                    language=projected.token.language,
                ),
                projected.timing_source,
                projected.aligner_index,
                projected.transcript_index,
            )
            for projected in projected_tokens
        ]

    def _usable_projected_tokens(
        self,
        projected_tokens: Iterable[ProjectedToken],
    ) -> List[ProjectedToken]:
        return [
            projected
            for projected in projected_tokens
            if projected.timing_source in {"aligner", "estimated"}
        ]

    def _split_window_tokens(
        self,
        tokens: Iterable[Token],
        window: AlignmentWindow,
    ) -> tuple[List[Token], List[Token], List[Token]]:
        left_overlap: List[Token] = []
        core_tokens: List[Token] = []
        right_overlap: List[Token] = []

        for token in tokens:
            if token_overlaps_core(
                token,
                core_start=window.core_start,
                core_end=window.core_end,
            ):
                core_tokens.append(token)
            elif token.start_time < window.core_start:
                left_overlap.append(token)
            else:
                right_overlap.append(token)

        return left_overlap, core_tokens, right_overlap

    def _merge_window_runs(self, window_runs: List[WindowRun]) -> List[Token]:
        merged_tokens: List[Token] = []
        passing_block: List[WindowRun] = []

        for window_run in window_runs:
            if window_run.error is not None:
                merged_tokens = self._append_tokens(
                    merged_tokens,
                    self._merge_passing_block(passing_block),
                    enforce_monotonic=True,
                )
                passing_block = []
                continue

            if self._needs_text_fallback(window_run):
                merged_tokens = self._append_tokens(
                    merged_tokens,
                    self._merge_passing_block(passing_block),
                    enforce_monotonic=True,
                )
                passing_block = []
                continue

            if window_run.quality is not None and window_run.quality.passed:
                if passing_block and not self._same_alignment_unit_scope(
                    passing_block[-1],
                    window_run,
                ):
                    merged_tokens = self._append_tokens(
                        merged_tokens,
                        self._merge_passing_block(passing_block),
                        enforce_monotonic=True,
                    )
                    passing_block = []
                passing_block.append(window_run)
                continue

            merged_tokens = self._append_tokens(
                merged_tokens,
                self._merge_passing_block(passing_block),
                enforce_monotonic=True,
            )
            passing_block = []
            merged_tokens = self._append_tokens(
                merged_tokens,
                self._fallback_tokens_for_run(window_run),
                enforce_monotonic=True,
            )

        return self._append_tokens(
            merged_tokens,
            self._merge_passing_block(passing_block),
            enforce_monotonic=True,
        )

    def _merge_passing_block(self, window_runs: List[WindowRun]) -> List[Token]:
        if not window_runs:
            return []
        if len(window_runs) == 1:
            return self._fallback_tokens_for_run(window_runs[0])

        merged_tokens: List[Token] = list(window_runs[0].tokens)
        current_span = self._window_span(window_runs[0].window)

        for window_run in window_runs[1:]:
            next_span = self._window_span(window_run.window)
            merged_tokens = merge_adjacent_windows(
                merged_tokens,
                window_run.tokens,
                current_span,
                next_span,
            )
            current_span = next_span

        owned_tokens = self._owned_tokens_for_block(merged_tokens, window_runs)
        if owned_tokens:
            return owned_tokens

        fallback_tokens: List[Token] = []
        for window_run in window_runs:
            fallback_tokens = self._append_tokens(
                fallback_tokens,
                self._fallback_tokens_for_run(window_run),
                enforce_monotonic=True,
            )
        return fallback_tokens

    def _owned_tokens_for_block(
        self,
        tokens: List[Token],
        window_runs: List[WindowRun],
    ) -> List[Token]:
        owned_tokens = [
            token
            for token in tokens
            if any(
                token_overlaps_core(
                    token,
                    core_start=window_run.window.core_start,
                    core_end=window_run.window.core_end,
                )
                for window_run in window_runs
            )
        ]
        return owned_tokens

    def _fallback_tokens_for_run(self, window_run: WindowRun) -> List[Token]:
        return self._preferred_tokens_for_window(window_run)

    def _needs_text_fallback(self, window_run: WindowRun) -> bool:
        return (
            window_run.error is None
            and bool(window_run.text.strip())
            and not window_run.has_timing_anchor
            and not window_run.tokens
        )

    def _append_tokens(
        self,
        existing: List[Token],
        new_tokens: List[Token],
        *,
        enforce_monotonic: bool = False,
    ) -> List[Token]:
        if not new_tokens:
            return existing

        merged = list(existing)
        for token in new_tokens:
            if merged and self._same_token(merged[-1], token):
                continue
            if enforce_monotonic and merged and token.start_time < merged[-1].start_time:
                token = self._coerce_monotonic_token(merged[-1], token)
            merged.append(token)

        return merged

    def _append_segments(
        self, existing: List[Segment], new_segments: List[Segment]
    ) -> List[Segment]:
        if not new_segments:
            return existing
        merged = list(existing)
        for segment in new_segments:
            merged.append(
                Segment(
                    id=f"seg-{len(merged) + 1}",
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    language=segment.language,
                    tokens=list(segment.tokens),
                    speaker=segment.speaker,
                )
            )
        merged.sort(key=lambda segment: (segment.start_time, segment.end_time))
        for index, segment in enumerate(merged, start=1):
            segment.id = f"seg-{index}"
        return merged

    def _coerce_monotonic_token(self, previous: Token, token: Token) -> Token:
        start_time = previous.start_time
        end_time = max(token.end_time, start_time)
        return Token(
            text=token.text,
            start_time=start_time,
            end_time=end_time,
            unit=token.unit,
            language=token.language,
        )

    def _build_document(
        self,
        *,
        audio_path: Path,
        total_duration_sec: float,
        windows: List[AlignmentWindow],
        window_runs: List[WindowRun],
        segments: List[Segment],
        speech_plan_used: bool = False,
        alignment_unit_count: int = 0,
    ) -> TranscriptionDocument:
        detected_language = next(
            (run.language for run in window_runs if run.language is not None),
            None,
        )
        document = TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            detected_language=detected_language,
            segments=segments,
        )
        provider_metadata = {
            "processing_strategy": (
                "vad_alignment_unit_bounded_alignment"
                if speech_plan_used
                else "windowed_bounded_alignment"
            ),
            "window_count": len(windows),
            "duration_sec": total_duration_sec,
            "quality_pass_count": sum(
                1 for run in window_runs if run.quality is not None and run.quality.passed
            ),
            "failed_window_count": sum(1 for run in window_runs if run.error is not None),
            "window_diagnostics": [
                self._build_window_diagnostic(run) for run in window_runs
            ],
        }
        if speech_plan_used:
            provider_metadata["alignment_unit_count"] = alignment_unit_count
        document.ensure_source_media()["provider_metadata"] = provider_metadata
        return document

    def _preferred_tokens_for_window(self, window_run: WindowRun) -> List[Token]:
        if not window_run.core_tokens:
            return list(window_run.tokens)

        protected_prefix = self._protected_prefix_tokens(window_run)
        protected_suffix = self._protected_suffix_tokens(window_run)
        return protected_prefix + list(window_run.core_tokens) + protected_suffix

    def _protected_prefix_tokens(self, window_run: WindowRun) -> List[Token]:
        if not window_run.left_overlap_tokens or not window_run.core_tokens:
            return []
        first_core = window_run.core_tokens[0]
        expected_index = self._transcript_index_for_token(window_run, first_core)
        prefix: List[Token] = []
        for token in reversed(window_run.left_overlap_tokens):
            if first_core.start_time - token.end_time > 0.35:
                break
            if self._timing_source_for_token(window_run, token) != "estimated":
                break
            if not self._short_edge_token(token):
                break
            token_index = self._transcript_index_for_token(window_run, token)
            if expected_index is not None:
                if token_index != expected_index - 1:
                    break
                expected_index = token_index
            prefix.append(token)
        prefix.reverse()
        return prefix

    def _protected_suffix_tokens(self, window_run: WindowRun) -> List[Token]:
        if not window_run.right_overlap_tokens or not window_run.core_tokens:
            return []
        last_core = window_run.core_tokens[-1]
        expected_index = self._transcript_index_for_token(window_run, last_core)
        suffix: List[Token] = []
        for token in window_run.right_overlap_tokens:
            if token.start_time - last_core.end_time > 0.35:
                break
            if self._timing_source_for_token(window_run, token) != "estimated":
                break
            if not self._short_edge_token(token):
                break
            token_index = self._transcript_index_for_token(window_run, token)
            if expected_index is not None:
                if token_index != expected_index + 1:
                    break
                expected_index = token_index
            suffix.append(token)
        return suffix

    def _short_edge_token(self, token: Token) -> bool:
        text = token.text.strip()
        if token.unit == "char":
            return len(text) == 1
        return 0 < len(text) <= 3

    def _timing_source_for_token(
        self, window_run: WindowRun, token: Token
    ) -> str | None:
        projected = self._projected_for_token(window_run, token)
        return projected.timing_source if projected is not None else None

    def _transcript_index_for_token(
        self, window_run: WindowRun, token: Token
    ) -> int | None:
        projected = self._projected_for_token(window_run, token)
        return projected.transcript_index if projected is not None else None

    def _projected_for_token(
        self, window_run: WindowRun, token: Token
    ) -> ProjectedToken | None:
        for projected in window_run.projected_tokens:
            if self._same_token(projected.token, token):
                return projected
        return None

    def _same_token(self, left: Token, right: Token) -> bool:
        return (
            left.text == right.text
            and left.start_time == right.start_time
            and left.end_time == right.end_time
            and left.unit == right.unit
            and left.language == right.language
        )

    def _window_span(self, window: AlignmentWindow) -> WindowSpan:
        return WindowSpan(
            core_start=window.core_start,
            core_end=window.core_end,
            context_start=window.context_start,
            context_end=window.context_end,
        )

    def _build_window_diagnostic(self, window_run: WindowRun) -> dict[str, Any]:
        diagnostic = {
            "index": window_run.window.index,
            "status": "failed" if window_run.error is not None else "completed",
            "core_start": window_run.window.core_start,
            "core_end": window_run.window.core_end,
            "context_start": window_run.window.context_start,
            "context_end": window_run.window.context_end,
            "token_count": len(window_run.tokens),
            "timing_source_counts": dict(window_run.timing_source_counts),
            "has_timing_anchor": window_run.has_timing_anchor,
        }
        if window_run.window.alignment_unit_index is not None:
            diagnostic["alignment_unit_index"] = window_run.window.alignment_unit_index
        if window_run.display_bounds is not None:
            diagnostic["display_start"] = window_run.display_bounds.start_time
            diagnostic["display_end"] = window_run.display_bounds.end_time
        if window_run.error is not None:
            diagnostic["error"] = window_run.error
            return diagnostic

        diagnostic["quality"] = {
            "passed": window_run.quality.passed if window_run.quality is not None else False,
            "monotonic_timestamp_ratio": (
                window_run.quality.monotonic_timestamp_ratio
                if window_run.quality is not None
                else 0.0
            ),
            "zero_or_flat_timestamp_ratio": (
                window_run.quality.zero_or_flat_timestamp_ratio
                if window_run.quality is not None
                else 1.0
            ),
            "boundary_disagreement_score": (
                window_run.quality.boundary_disagreement_score
                if window_run.quality is not None
                else 1.0
            ),
            "core_context_text_divergence": (
                window_run.quality.core_context_text_divergence
                if window_run.quality is not None
                else 1.0
            ),
        }
        if window_run.quality is not None:
            diagnostic["quality"]["estimated_token_ratio"] = (
                window_run.quality.estimated_token_ratio
            )
            diagnostic["quality"]["unresolved_token_ratio"] = (
                window_run.quality.unresolved_token_ratio
            )
        return diagnostic

    def _fallback_segments_from_windows(self, window_runs: List[WindowRun]) -> List[Segment]:
        segments: List[Segment] = []
        for window_run in window_runs:
            if window_run.error is not None or not window_run.text.strip():
                continue
            start_time = self._fallback_start_time(window_run)
            end_time = self._fallback_end_time(window_run, start_time)
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1}",
                    text=window_run.text,
                    start_time=start_time,
                    end_time=end_time,
                    language=window_run.language,
                    tokens=[],
                )
            )
        return segments

    def _unresolved_fallback_segments_from_windows(
        self,
        window_runs: List[WindowRun],
    ) -> List[Segment]:
        segments: List[Segment] = []
        for window_run in window_runs:
            if self._needs_text_fallback(window_run):
                continue
            segments = self._append_segments(
                segments,
                self._unresolved_fallback_segments_for_run(window_run),
            )
        return segments

    def _unresolved_fallback_segments_for_run(
        self,
        window_run: WindowRun,
    ) -> List[Segment]:
        if (
            window_run.error is not None
            or not window_run.text.strip()
            or not window_run.has_timing_anchor
        ):
            return []

        segments: List[Segment] = []
        projected_tokens = list(window_run.projected_tokens)
        index = 0
        while index < len(projected_tokens):
            projected = projected_tokens[index]
            if projected.timing_source != "unresolved":
                index += 1
                continue

            group_start = index
            group: List[ProjectedToken] = []
            while (
                index < len(projected_tokens)
                and projected_tokens[index].timing_source == "unresolved"
            ):
                group.append(projected_tokens[index])
                index += 1

            segment = self._unresolved_group_to_segment(
                window_run,
                projected_tokens,
                group,
                group_start,
                index,
            )
            if segment is not None:
                segments.append(segment)

        for segment_index, segment in enumerate(segments, start=1):
            segment.id = f"seg-{segment_index}"
        return segments

    def _unresolved_group_to_segment(
        self,
        window_run: WindowRun,
        projected_tokens: List[ProjectedToken],
        group: List[ProjectedToken],
        group_start: int,
        group_end: int,
    ) -> Segment | None:
        previous = self._nearest_timed_projected_token(
            projected_tokens,
            start_index=group_start - 1,
            step=-1,
        )

        next_projected = self._nearest_timed_projected_token(
            projected_tokens,
            start_index=group_end,
            step=1,
        )
        text = self._join_tokens(projected.token for projected in group)
        if not text:
            return None

        duration = self._estimate_local_fallback_text_duration(
            text,
            window_run.language,
        )
        previous_overlaps_core = previous is not None and token_overlaps_core(
            previous.token,
            core_start=window_run.window.core_start,
            core_end=window_run.window.core_end,
        )
        next_overlaps_core = next_projected is not None and token_overlaps_core(
            next_projected.token,
            core_start=window_run.window.core_start,
            core_end=window_run.window.core_end,
        )

        if previous_overlaps_core and self._group_is_transcript_adjacent_to_previous(
            group,
            previous,
        ):
            start_time = previous.token.end_time
            if next_projected is not None:
                end_time = min(next_projected.token.start_time, start_time + duration)
            else:
                end_time = start_time + duration
        elif next_overlaps_core and self._group_is_transcript_adjacent_to_next(
            group,
            next_projected,
        ):
            end_time = next_projected.token.start_time
            start_time = end_time - duration
        else:
            return None

        start_time = max(start_time, window_run.window.core_start)
        end_time = min(end_time, window_run.window.core_end)
        if window_run.display_bounds is not None:
            start_time = max(start_time, window_run.display_bounds.start_time)
            end_time = min(end_time, window_run.display_bounds.end_time)
        if end_time <= start_time:
            return None

        return Segment(
            id="seg-0",
            text=text,
            start_time=start_time,
            end_time=end_time,
            language=window_run.language,
            tokens=[],
        )

    def _nearest_timed_projected_token(
        self,
        projected_tokens: List[ProjectedToken],
        *,
        start_index: int,
        step: int,
    ) -> ProjectedToken | None:
        index = start_index
        while 0 <= index < len(projected_tokens):
            projected = projected_tokens[index]
            if projected.timing_source in {"aligner", "estimated"}:
                return projected
            index += step
        return None

    def _group_is_transcript_adjacent_to_previous(
        self,
        group: List[ProjectedToken],
        previous: ProjectedToken,
    ) -> bool:
        if not group:
            return False
        group_index = group[0].transcript_index
        previous_index = previous.transcript_index
        if group_index is None or previous_index is None:
            return True
        return group_index == previous_index + 1

    def _group_is_transcript_adjacent_to_next(
        self,
        group: List[ProjectedToken],
        next_projected: ProjectedToken,
    ) -> bool:
        if not group:
            return False
        group_index = group[-1].transcript_index
        next_index = next_projected.transcript_index
        if group_index is None or next_index is None:
            return True
        return group_index + 1 == next_index

    def _estimate_local_fallback_text_duration(
        self,
        text: str,
        language: Optional[str],
    ) -> float:
        normalized = (language or "").lower()
        if (
            normalized.startswith("zh")
            or "chinese" in normalized
            or self._contains_cjk(text)
        ):
            char_count = sum(1 for char in text if not char.isspace())
            return min(6.0, max(0.12, char_count * 0.12))
        word_count = len([piece for piece in text.split() if piece])
        return min(6.0, max(0.35, word_count * 0.35))

    def _fallback_start_time(self, window_run: WindowRun) -> float:
        if window_run.display_bounds is not None:
            return max(window_run.display_bounds.start_time, window_run.window.core_start)
        return window_run.window.core_start

    def _fallback_end_time(self, window_run: WindowRun, start_time: float) -> float:
        max_duration = 6.0
        estimated_duration = min(
            max_duration,
            self._estimate_fallback_text_duration(window_run.text, window_run.language),
        )
        end_time = start_time + estimated_duration
        if window_run.display_bounds is not None:
            end_time = min(end_time, window_run.display_bounds.end_time)
        else:
            end_time = min(end_time, window_run.window.core_end)
        return max(start_time, end_time)

    def _estimate_fallback_text_duration(
        self, text: str, language: Optional[str]
    ) -> float:
        normalized = (language or "").lower()
        if normalized.startswith("zh") or "chinese" in normalized or self._contains_cjk(text):
            char_count = sum(1 for char in text if not char.isspace())
            return max(1.0, char_count * 0.12)
        word_count = len([piece for piece in text.split() if piece])
        return max(1.2, word_count * 0.35)

    def _stabilize_segment_boundaries(
        self,
        segments: List[Segment],
        *,
        total_duration_sec: float,
        display_bounds: Iterable[WindowDisplayBounds] | None = None,
        tail_padding_sec: float = 0.12,
        target_max_segment_duration_sec: float = 8.0,
    ) -> List[Segment]:
        if not segments:
            return []

        stabilized = [
            Segment(
                id=segment.id,
                text=segment.text,
                start_time=segment.start_time,
                end_time=segment.end_time,
                language=segment.language,
                tokens=list(segment.tokens),
                speaker=segment.speaker,
            )
            for segment in segments
        ]

        previous_end = 0.0
        for segment in stabilized:
            segment.start_time = max(0.0, segment.start_time)
            segment.end_time = max(segment.start_time, segment.end_time)
            if segment.start_time < previous_end:
                segment.start_time = previous_end
                segment.end_time = max(segment.end_time, segment.start_time)
            previous_end = segment.end_time

        for index, segment in enumerate(stabilized):
            next_start = (
                stabilized[index + 1].start_time
                if index + 1 < len(stabilized)
                else total_duration_sec
            )
            padded_end = segment.end_time + tail_padding_sec
            segment.end_time = min(total_duration_sec, max(segment.end_time, min(padded_end, next_start)))
            segment.end_time = max(segment.start_time, segment.end_time)

        for index in range(len(stabilized) - 1):
            if stabilized[index].end_time > stabilized[index + 1].start_time:
                stabilized[index].end_time = stabilized[index + 1].start_time
                stabilized[index].end_time = max(
                    stabilized[index].start_time,
                    stabilized[index].end_time,
                )

        bounds = list(display_bounds or [])
        for segment in stabilized:
            bound = self._nearest_display_bound(segment, bounds)
            if bound is None:
                continue
            segment.start_time = max(segment.start_time, bound.start_time)
            segment.end_time = min(segment.end_time, bound.end_time)
            segment.end_time = max(segment.start_time, segment.end_time)

        for index in range(len(stabilized) - 1):
            if stabilized[index].end_time > stabilized[index + 1].start_time:
                stabilized[index].end_time = stabilized[index + 1].start_time
                stabilized[index].end_time = max(
                    stabilized[index].start_time,
                    stabilized[index].end_time,
                )

        return stabilized

    def _nearest_display_bound(
        self,
        segment: Segment,
        bounds: List[WindowDisplayBounds],
    ) -> WindowDisplayBounds | None:
        overlapping = [
            bound
            for bound in bounds
            if segment.end_time >= bound.start_time
            and segment.start_time <= bound.end_time
        ]
        if not overlapping:
            return None
        return max(
            overlapping,
            key=lambda bound: self._display_bound_overlap_duration(segment, bound),
        )

    def _display_bound_overlap_duration(
        self,
        segment: Segment,
        bound: WindowDisplayBounds,
    ) -> float:
        return max(
            0.0,
            min(segment.end_time, bound.end_time)
            - max(segment.start_time, bound.start_time),
        )

    def _item_to_token(self, item: Any, language: Optional[str]) -> Token:
        text = str(getattr(item, "text", "")).strip()
        return Token(
            text=text,
            start_time=float(getattr(item, "start_time", 0.0)),
            end_time=float(getattr(item, "end_time", 0.0)),
            unit=self._infer_unit(text=text, language=language),
            language=language,
        )

    def _infer_unit(self, *, text: str, language: Optional[str]) -> str:
        normalized = (language or "").lower()
        if normalized.startswith("zh") or "chinese" in normalized or self._contains_cjk(text):
            return "char"
        return "word"

    def _contains_cjk(self, text: str) -> bool:
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _normalize_language(self, language: Optional[str]) -> Optional[str]:
        if language is None:
            return None
        normalized = str(language).strip()
        return normalized or None

    def _tokens_to_segments(
        self,
        tokens: Iterable[Token],
        *,
        target_max_segment_duration_sec: float = 8.0,
    ) -> List[Segment]:
        segments: List[Segment] = []
        current_tokens: List[Token] = []
        previous_end: Optional[float] = None

        for token in tokens:
            should_break = False
            if current_tokens:
                if previous_end is not None and token.start_time - previous_end >= 1.0:
                    should_break = True
                if self._ends_segment(current_tokens[-1].text):
                    should_break = True
                if (
                    current_tokens
                    and token.end_time - current_tokens[0].start_time
                    > target_max_segment_duration_sec
                ):
                    should_break = True
            if should_break:
                segments.append(self._build_segment(len(segments) + 1, current_tokens))
                current_tokens = []
            current_tokens.append(token)
            previous_end = token.end_time

        if current_tokens:
            segments.append(self._build_segment(len(segments) + 1, current_tokens))

        return segments

    def _build_segment(self, index: int, tokens: List[Token]) -> Segment:
        language = tokens[0].language if tokens else None
        text = self._join_tokens(tokens)
        return Segment(
            id=f"seg-{index}",
            text=text,
            start_time=tokens[0].start_time if tokens else 0.0,
            end_time=tokens[-1].end_time if tokens else 0.0,
            language=language,
            tokens=list(tokens),
        )

    def _join_tokens(self, tokens: Iterable[Token]) -> str:
        pieces: List[str] = []
        previous_unit: Optional[str] = None
        for token in tokens:
            if pieces and token.unit == "word" and previous_unit == "word":
                pieces.append(" ")
            pieces.append(token.text)
            previous_unit = token.unit
        return "".join(pieces).strip()

    def _ends_segment(self, text: str) -> bool:
        return text.endswith((".", "!", "?", "。", "！", "？", ";", "；"))
