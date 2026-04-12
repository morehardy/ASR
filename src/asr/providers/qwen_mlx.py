"""MLX provider for the planned Qwen3 ASR + ForcedAligner backend."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

from asr.models import Segment, Token, TranscriptionDocument
from asr.observability.observer import Observer
from asr.observability.timing import observe_step
from asr.providers.authority import (
    build_transcript_tokens,
    project_timing_onto_transcript,
)
from asr.providers.media_probe import parse_silence_anchors, probe_duration_sec
from asr.providers.quality import QualityResult, QualityThresholds, evaluate_quality
from asr.providers.window_merge import WindowSpan, merge_adjacent_windows
from asr.providers.windowing import AlignmentWindow, WindowBudgetConfig, WindowPlanner


DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"


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


@dataclass
class QwenMlxProvider:
    """Provider that wraps the planned MLX Qwen3 model pair."""

    asr_model_id: str = DEFAULT_ASR_MODEL
    aligner_model_id: str = DEFAULT_ALIGNER_MODEL
    name: str = "qwen-mlx"
    window_config: WindowBudgetConfig = field(default_factory=WindowBudgetConfig)
    quality_thresholds: QualityThresholds = field(default_factory=QualityThresholds)

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

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
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
            self._active_audio_path = audio_path
            try:
                windows = self._plan_windows(total_duration_sec)
            finally:
                self._active_audio_path = None

        self._begin_context_windowing()
        try:
            if not windows:
                return self._build_document(
                    audio_path=audio_path,
                    total_duration_sec=total_duration_sec,
                    windows=windows,
                    window_runs=[],
                    segments=[],
                )

            window_runs: List[WindowRun] = []
            for index, window in enumerate(windows, start=1):
                window_runs.append(
                    self._execute_window(
                        audio_path,
                        window,
                        window_index=index,
                        window_count=len(windows),
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
                if not segments:
                    segments = self._fallback_segments_from_windows(window_runs)
                segments = self._stabilize_segment_boundaries(
                    segments,
                    total_duration_sec=total_duration_sec,
                )

            return self._build_document(
                audio_path=audio_path,
                total_duration_sec=total_duration_sec,
                windows=windows,
                window_runs=window_runs,
                segments=segments,
            )
        finally:
            self._cleanup_context_windowing()

    def _plan_windows(self, total_duration_sec: float) -> List[AlignmentWindow]:
        planner = WindowPlanner(
            self.window_config,
            anchor_resolver=self._resolve_silence_anchor,
        )
        return planner.plan(total_duration_sec)

    def _transcribe_window(self, audio_path: Path, window: AlignmentWindow) -> WindowRun:
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
        projected_tokens = project_timing_onto_transcript(
            transcript_tokens,
            aligner_tokens,
        )
        global_tokens = self._offset_tokens(projected_tokens, window.context_start)

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
        )

    def _execute_window(
        self,
        audio_path: Path,
        window: AlignmentWindow,
        *,
        window_index: int,
        window_count: int,
    ) -> WindowRun:
        try:
            with observe_step(
                self._observer,
                run_id=self._observer_run_id,
                file_id=self._observer_file_id,
                source_path=self._observer_source_path,
                step="provider_window",
                meta={"window_index": window_index, "window_count": window_count},
            ):
                return self._transcribe_window(audio_path, window)
        except Exception as exc:
            return WindowRun(
                window=window,
                error=str(exc),
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

        return candidate

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
                "Install the optional dependency set with `uv sync --extra mlx`."
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

    def _split_window_tokens(
        self,
        tokens: Iterable[Token],
        window: AlignmentWindow,
    ) -> tuple[List[Token], List[Token], List[Token]]:
        left_overlap: List[Token] = []
        core_tokens: List[Token] = []
        right_overlap: List[Token] = []

        for token in tokens:
            if token.start_time < window.core_start:
                left_overlap.append(token)
            elif token.start_time < window.core_end:
                core_tokens.append(token)
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

            if window_run.quality is not None and window_run.quality.passed:
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
                window_run.window.core_start <= token.start_time < window_run.window.core_end
                for window_run in window_runs
            )
        ]
        return owned_tokens

    def _fallback_tokens_for_run(self, window_run: WindowRun) -> List[Token]:
        return self._preferred_tokens_for_window(window_run)

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
        document.ensure_source_media()["provider_metadata"] = {
            "processing_strategy": "windowed_bounded_alignment",
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
        return document

    def _preferred_tokens_for_window(self, window_run: WindowRun) -> List[Token]:
        if window_run.core_tokens:
            return list(window_run.core_tokens)
        return list(window_run.tokens)

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
        }
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
        return diagnostic

    def _fallback_segments_from_windows(self, window_runs: List[WindowRun]) -> List[Segment]:
        segments: List[Segment] = []
        for window_run in window_runs:
            if not window_run.text:
                continue
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1}",
                    text=window_run.text,
                    start_time=window_run.window.core_start,
                    end_time=window_run.window.core_end,
                    language=window_run.language,
                    tokens=[],
                )
            )
        return segments

    def _stabilize_segment_boundaries(
        self,
        segments: List[Segment],
        *,
        total_duration_sec: float,
        tail_padding_sec: float = 0.12,
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

        return stabilized

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

    def _tokens_to_segments(self, tokens: Iterable[Token]) -> List[Segment]:
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
