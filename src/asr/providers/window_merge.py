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


def _in_core(token: Token, span: WindowSpan) -> bool:
    return span.core_start <= token.start_time < span.core_end


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
    matched_pairs = _dp_pairs(left_tokens, right_tokens, max_time_delta)
    matched_right_indexes = {right_index for _, right_index in matched_pairs}
    matched_right_by_left = {left_index: right_index for left_index, right_index in matched_pairs}

    merged: list[Token] = []
    for left_index, left_token in enumerate(left_tokens):
        right_index = matched_right_by_left.get(left_index)
        if right_index is None:
            merged.append(left_token)
            continue

        right_token = right_tokens[right_index]
        left_in_core = _in_core(left_token, left_span)
        right_in_core = _in_core(right_token, right_span)
        if right_in_core:
            merged.append(right_token)
        elif left_in_core:
            merged.append(left_token)
        else:
            merged.append(left_token)

    for right_index, right_token in enumerate(right_tokens):
        if right_index in matched_right_indexes:
            continue
        merged.append(right_token)

    merged.sort(key=lambda token: (token.start_time, token.end_time))
    return merged
