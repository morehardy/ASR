"""Helpers for keeping transcript text authoritative while borrowing timing."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from difflib import SequenceMatcher
from typing import List, Literal, Optional

from asr.models import Token

TimingSource = Literal["aligner", "estimated", "unresolved"]


@dataclass(frozen=True, slots=True)
class ProjectedToken:
    token: Token
    timing_source: TimingSource
    aligner_index: int | None = None
    transcript_index: int | None = None

    @property
    def start_time(self) -> float:
        return self.token.start_time

    @property
    def end_time(self) -> float:
        return self.token.end_time


def build_transcript_tokens(text: str, language: Optional[str]) -> List[Token]:
    stripped = text.strip()
    if not stripped:
        return []

    normalized_language = _normalize_language(language)
    if _is_zh_language(normalized_language):
        units = [char for char in stripped if not char.isspace()]
    else:
        units = stripped.split()

    return [
        Token(
            text=unit,
            start_time=0.0,
            end_time=0.0,
            unit="token",
            language=normalized_language,
        )
        for unit in units
    ]


def project_timing_onto_transcript(
    transcript_tokens: List[Token], aligner_tokens: List[Token]
) -> List[Token]:
    return [
        projected.token
        for projected in project_timing_onto_transcript_detailed(
            transcript_tokens,
            aligner_tokens,
        )
    ]


def project_timing_onto_transcript_detailed(
    transcript_tokens: List[Token], aligner_tokens: List[Token]
) -> List[ProjectedToken]:
    projected: List[ProjectedToken] = []
    aligner_index = 0

    for transcript_index, transcript_token in enumerate(transcript_tokens):
        match_index = _find_forward_match(transcript_token.text, aligner_tokens, aligner_index)
        if match_index is None:
            projected.append(
                ProjectedToken(
                    _clone_token(transcript_token),
                    "unresolved",
                    None,
                    transcript_index,
                )
            )
            continue

        aligner_token = aligner_tokens[match_index]
        aligner_index = match_index + 1

        if not _valid_token_timing(aligner_token):
            projected.append(
                ProjectedToken(
                    _clone_token(transcript_token),
                    "unresolved",
                    None,
                    transcript_index,
                )
            )
            continue

        projected.append(
            ProjectedToken(
                Token(
                    text=transcript_token.text,
                    start_time=aligner_token.start_time,
                    end_time=aligner_token.end_time,
                    unit=transcript_token.unit,
                    language=transcript_token.language,
                ),
                "aligner",
                match_index,
                transcript_index,
            )
        )

    return projected


def repair_unmatched_timings(
    projected_tokens: List[ProjectedToken],
    *,
    clip_duration_sec: float | None = None,
    max_estimated_token_duration_sec: float = 0.32,
    prefer_next_anchor_indexes: set[int] | None = None,
) -> List[ProjectedToken]:
    if not projected_tokens:
        return []

    prefer_next_anchor_indexes = prefer_next_anchor_indexes or set()
    anchor_indexes = [
        index
        for index, projected in enumerate(projected_tokens)
        if projected.timing_source == "aligner" and _valid_token_timing(projected.token)
    ]
    if not anchor_indexes:
        return list(projected_tokens)

    repaired = list(projected_tokens)
    first_anchor = anchor_indexes[0]
    if first_anchor > 0:
        _repair_leading_tokens(
            repaired,
            start_index=0,
            end_index=first_anchor,
            next_anchor=repaired[first_anchor].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )

    for left_anchor, right_anchor in zip(anchor_indexes, anchor_indexes[1:]):
        if right_anchor - left_anchor > 1:
            _repair_middle_tokens(
                repaired,
                start_index=left_anchor + 1,
                end_index=right_anchor,
                previous_anchor=repaired[left_anchor].token,
                next_anchor=repaired[right_anchor].token,
                max_estimated_token_duration_sec=max_estimated_token_duration_sec,
                prefer_next_anchor_indexes=prefer_next_anchor_indexes,
            )

    last_anchor = anchor_indexes[-1]
    if last_anchor < len(repaired) - 1:
        _repair_trailing_tokens(
            repaired,
            start_index=last_anchor + 1,
            previous_anchor=repaired[last_anchor].token,
            clip_duration_sec=clip_duration_sec,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )

    return repaired


_ESTIMATED_TOKEN_GAP_SEC = 0.02


def _repair_leading_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    end_index: int,
    next_anchor: Token,
    max_estimated_token_duration_sec: float,
) -> None:
    cursor = next_anchor.start_time
    for index in range(end_index - 1, start_index - 1, -1):
        duration = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        end_time = max(0.0, cursor - _ESTIMATED_TOKEN_GAP_SEC)
        start_time = max(0.0, end_time - duration)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = start_time


def _repair_middle_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    end_index: int,
    previous_anchor: Token,
    next_anchor: Token,
    max_estimated_token_duration_sec: float,
    prefer_next_anchor_indexes: set[int],
) -> None:
    count = end_index - start_index
    if count <= 0:
        return

    raw_start = previous_anchor.end_time
    raw_end = next_anchor.start_time
    raw_available = raw_end - raw_start
    if raw_available <= 0.0:
        return

    gapped_start = raw_start + _ESTIMATED_TOKEN_GAP_SEC
    gapped_end = raw_end - _ESTIMATED_TOKEN_GAP_SEC
    if gapped_end > gapped_start:
        available_start = gapped_start
        available_end = gapped_end
    else:
        available_start = raw_start
        available_end = raw_end

    available = available_end - available_start
    if available <= 0.0:
        return

    slot = available / count
    cursor = available_start
    for index in range(start_index, end_index):
        requested = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        duration = min(requested, max(0.0, slot - _ESTIMATED_TOKEN_GAP_SEC))
        start_time = cursor
        end_time = min(available_end, start_time + duration)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = min(available_end, start_time + slot)

    _repair_selected_middle_tokens_from_next_anchor(
        repaired,
        start_index=start_index,
        end_index=end_index,
        next_anchor=next_anchor,
        max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        prefer_next_anchor_indexes=prefer_next_anchor_indexes,
    )


def _repair_selected_middle_tokens_from_next_anchor(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    end_index: int,
    next_anchor: Token,
    max_estimated_token_duration_sec: float,
    prefer_next_anchor_indexes: set[int],
) -> None:
    if not prefer_next_anchor_indexes:
        return

    cursor = next_anchor.start_time
    for index in range(end_index - 1, start_index - 1, -1):
        if index + 1 < end_index:
            cursor = min(cursor, repaired[index + 1].token.start_time)

        projected = repaired[index]
        if projected.transcript_index not in prefer_next_anchor_indexes:
            continue

        duration = _estimated_token_duration(
            projected.token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        end_time = max(0.0, cursor - _ESTIMATED_TOKEN_GAP_SEC)
        start_time = max(0.0, end_time - duration)
        repaired[index] = _with_estimated_timing(projected, start_time, end_time)
        cursor = start_time


def _repair_trailing_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    previous_anchor: Token,
    clip_duration_sec: float | None,
    max_estimated_token_duration_sec: float,
) -> None:
    cursor = previous_anchor.end_time
    clip_end = clip_duration_sec if clip_duration_sec is not None and math.isfinite(clip_duration_sec) else None
    for index in range(start_index, len(repaired)):
        duration = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        start_time = cursor + _ESTIMATED_TOKEN_GAP_SEC
        if clip_end is not None and start_time > clip_end:
            return
        end_time = start_time + duration
        if clip_end is not None:
            end_time = min(end_time, clip_end)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = end_time


def _with_estimated_timing(projected: ProjectedToken, start_time: float, end_time: float) -> ProjectedToken:
    token = projected.token
    return ProjectedToken(
        Token(
            text=token.text,
            start_time=max(0.0, start_time),
            end_time=max(max(0.0, start_time), end_time),
            unit=token.unit,
            language=token.language,
        ),
        "estimated",
        projected.aligner_index,
        projected.transcript_index,
    )


def _estimated_token_duration(
    token: Token,
    *,
    max_estimated_token_duration_sec: float,
) -> float:
    text = token.text.strip()
    if token.unit == "char" or _contains_cjk(text):
        return min(0.10, max_estimated_token_duration_sec)
    if len(text) <= 3:
        return min(0.10, max_estimated_token_duration_sec)
    return min(0.18, max_estimated_token_duration_sec)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _find_forward_match(
    transcript_text: str, aligner_tokens: List[Token], start_index: int
) -> Optional[int]:
    transcript_raw = transcript_text.lower()
    for index in range(start_index, len(aligner_tokens)):
        aligner_text = aligner_tokens[index].text
        if (
            SequenceMatcher(
                None,
                transcript_raw,
                aligner_text.lower(),
            ).ratio()
            >= 0.9
        ):
            return index

    # Conservative fallback: only strip sentence punctuation at token edges and require exact match.
    transcript_fallback = _strip_edge_sentence_punctuation(transcript_raw)
    if not transcript_fallback:
        return None

    for index in range(start_index, len(aligner_tokens)):
        aligner_text = aligner_tokens[index].text
        aligner_fallback = _strip_edge_sentence_punctuation(aligner_text.lower())
        if transcript_fallback == aligner_fallback:
            return index

    return None


def _clone_token(token: Token) -> Token:
    return Token(
        text=token.text,
        start_time=token.start_time,
        end_time=token.end_time,
        unit=token.unit,
        language=token.language,
    )


def _valid_token_timing(token: Token) -> bool:
    return (
        math.isfinite(token.start_time)
        and math.isfinite(token.end_time)
        and token.end_time >= token.start_time
    )


def _normalize_language(language: Optional[str]) -> Optional[str]:
    if language is None:
        return None
    normalized = str(language).strip()
    return normalized or None


def _is_zh_language(language: Optional[str]) -> bool:
    if language is None:
        return False
    normalized = language.lower()
    return normalized.startswith("zh")


_EDGE_SENTENCE_PUNCTUATION_RE = re.compile(
    r"^[\s\"'`“”‘’.,!?;:，。！？；：、()\[\]{}<>《》「」『』…]+|[\s\"'`“”‘’.,!?;:，。！？；：、()\[\]{}<>《》「」『』…]+$",
    flags=re.UNICODE,
)


def _strip_edge_sentence_punctuation(text: str) -> str:
    stripped = _EDGE_SENTENCE_PUNCTUATION_RE.sub("", text)
    return stripped or text
