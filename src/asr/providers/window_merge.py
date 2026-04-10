"""Ownership-aware merging for adjacent provider windows."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from asr.models import Token


@dataclass(frozen=True, slots=True)
class WindowSpan:
    core_start: float
    core_end: float
    context_start: float
    context_end: float


def _match(a: Token, b: Token, max_time_delta: float) -> bool:
    return (
        abs(a.start_time - b.start_time) <= max_time_delta
        and SequenceMatcher(None, a.text, b.text).ratio() >= 0.85
    )


def _dp_pairs(left: list[Token], right: list[Token], max_time_delta: float) -> list[tuple[int, int]]:
    left_len = len(left)
    right_len = len(right)
    scores = [[0] * (right_len + 1) for _ in range(left_len + 1)]

    for i in range(1, left_len + 1):
        for j in range(1, right_len + 1):
            if _match(left[i - 1], right[j - 1], max_time_delta):
                scores[i][j] = scores[i - 1][j - 1] + 1
            else:
                scores[i][j] = max(scores[i - 1][j], scores[i][j - 1])

    pairs: list[tuple[int, int]] = []
    i = left_len
    j = right_len
    while i > 0 and j > 0:
        if _match(left[i - 1], right[j - 1], max_time_delta):
            if scores[i][j] == scores[i - 1][j - 1] + 1:
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if scores[i - 1][j] >= scores[i][j - 1]:
            i -= 1
        else:
            j -= 1

    pairs.reverse()
    return pairs


def merge_adjacent_windows(
    left_tokens: list[Token],
    right_tokens: list[Token],
    left_span: WindowSpan,
    right_span: WindowSpan,
    max_time_delta: float = 0.25,
) -> list[Token]:
    matched_right_indexes = {right_index for _, right_index in _dp_pairs(left_tokens, right_tokens, max_time_delta)}

    merged: list[Token] = list(left_tokens)
    for idx, token in enumerate(right_tokens):
        in_right_core = right_span.core_start <= token.start_time < right_span.core_end
        if idx in matched_right_indexes and not in_right_core:
            continue
        if idx in matched_right_indexes:
            continue
        merged.append(token)

    merged.sort(key=lambda token: (token.start_time, token.end_time))
    return merged
