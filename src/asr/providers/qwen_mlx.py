"""MLX provider for the planned Qwen3 ASR + ForcedAligner backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from asr.models import Segment, Token, TranscriptionDocument


DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-bf16"


@dataclass
class QwenMlxProvider:
    """Provider that wraps the planned MLX Qwen3 model pair."""

    asr_model_id: str = DEFAULT_ASR_MODEL
    aligner_model_id: str = DEFAULT_ALIGNER_MODEL
    name: str = "qwen-mlx"

    def __post_init__(self) -> None:
        self._asr_model: Optional[Any] = None
        self._aligner_model: Optional[Any] = None

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        load = self._load_backend()
        self._asr_model = self._asr_model or load(self.asr_model_id)
        self._aligner_model = self._aligner_model or load(self.aligner_model_id)

        transcription = self._asr_model.generate(str(audio_path))
        text = getattr(transcription, "text", "").strip()
        language = self._normalize_language(getattr(transcription, "language", None))

        align_kwargs = {"text": text}
        if language:
            align_kwargs["language"] = language

        aligned_items = list(self._aligner_model.generate(str(audio_path), **align_kwargs))
        tokens = [self._item_to_token(item, language=language) for item in aligned_items if getattr(item, "text", "").strip()]
        segments = self._tokens_to_segments(tokens)

        if not segments and text:
            segments = [
                Segment(
                    id="seg-1",
                    text=text,
                    start_time=0.0,
                    end_time=0.0,
                    language=language,
                    tokens=[],
                )
            ]

        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            detected_language=language,
            segments=segments,
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
