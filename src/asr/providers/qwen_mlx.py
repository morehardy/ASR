"""MLX provider for the planned Qwen3 ASR + ForcedAligner backend."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional

from asr.models import Segment, Token, TranscriptionDocument
from asr.providers.authority import (
    build_transcript_tokens,
    project_timing_onto_transcript,
)
from asr.providers.media_probe import probe_duration_sec
from asr.providers.quality import QualityResult, QualityThresholds, evaluate_quality
from asr.providers.window_merge import WindowSpan, merge_adjacent_windows
from asr.providers.windowing import AlignmentWindow, WindowBudgetConfig, WindowPlanner


DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"


@dataclass(slots=True)
class WindowRun:
    window: AlignmentWindow
    text: str
    language: Optional[str]
    tokens: List[Token]
    core_tokens: List[Token]
    quality: QualityResult


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

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        load = self._load_backend()
        self._asr_model = self._asr_model or load(self.asr_model_id)
        self._aligner_model = self._aligner_model or load(self.aligner_model_id)

        total_duration_sec = self._probe_duration_sec(audio_path)
        windows = self._plan_windows(total_duration_sec)
        window_runs = [self._transcribe_window(audio_path, window) for window in windows]

        merged_tokens = self._merge_window_runs(window_runs)
        segments = self._tokens_to_segments(merged_tokens)
        if not segments:
            segments = self._fallback_segments_from_windows(window_runs)

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
            "quality_pass_count": sum(1 for run in window_runs if run.quality.passed),
            "window_diagnostics": [
                self._build_window_diagnostic(run) for run in window_runs
            ],
        }
        return document

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
        quality = evaluate_quality(
            tokens=global_tokens,
            left_overlap_tokens=left_overlap_tokens,
            right_overlap_tokens=right_overlap_tokens,
            core_text=core_text,
            context_text=text,
            thresholds=self.quality_thresholds,
        )

        return WindowRun(
            window=window,
            text=text,
            language=language,
            tokens=global_tokens,
            core_tokens=core_tokens,
            quality=quality,
        )

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
        del target_split_sec, search_start_sec, search_end_sec
        return None

    def _context_input_path(self, audio_path: Path, window: AlignmentWindow) -> str:
        return (
            f"{audio_path}#t={window.context_start:.3f},{window.context_end:.3f}"
        )

    def _context_generate_kwargs(self, window: AlignmentWindow) -> dict[str, float]:
        return {
            "start_time": window.context_start,
            "end_time": window.context_end,
        }

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
        tokenized_runs = [window_run for window_run in window_runs if window_run.tokens]
        if not tokenized_runs:
            return []

        if not all(window_run.quality.passed for window_run in tokenized_runs):
            return self._quality_fallback_tokens(tokenized_runs)

        return self._merge_window_runs_with_overlap_resolution(tokenized_runs)

    def _merge_window_runs_with_overlap_resolution(
        self, window_runs: List[WindowRun]
    ) -> List[Token]:
        merged_tokens: List[Token] = []
        current_span: Optional[WindowSpan] = None

        for window_run in window_runs:
            next_span = self._window_span(window_run.window)
            if not merged_tokens or current_span is None:
                merged_tokens = list(window_run.tokens)
                current_span = next_span
                continue

            merged_tokens = merge_adjacent_windows(
                merged_tokens,
                window_run.tokens,
                current_span,
                next_span,
            )
            current_span = next_span

        return merged_tokens

    def _quality_fallback_tokens(self, window_runs: List[WindowRun]) -> List[Token]:
        merged_tokens: List[Token] = []

        for window_run in window_runs:
            preferred_tokens = self._preferred_tokens_for_window(window_run)
            for token in preferred_tokens:
                if merged_tokens and self._same_token(merged_tokens[-1], token):
                    continue
                merged_tokens.append(token)

        return merged_tokens

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
        return {
            "index": window_run.window.index,
            "core_start": window_run.window.core_start,
            "core_end": window_run.window.core_end,
            "context_start": window_run.window.context_start,
            "context_end": window_run.window.context_end,
            "token_count": len(window_run.tokens),
            "quality": {
                "passed": window_run.quality.passed,
                "monotonic_timestamp_ratio": (
                    window_run.quality.monotonic_timestamp_ratio
                ),
                "zero_or_flat_timestamp_ratio": (
                    window_run.quality.zero_or_flat_timestamp_ratio
                ),
                "boundary_disagreement_score": (
                    window_run.quality.boundary_disagreement_score
                ),
                "core_context_text_divergence": (
                    window_run.quality.core_context_text_divergence
                ),
            },
        }

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
