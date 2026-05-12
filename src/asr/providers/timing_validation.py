"""Validation helpers for suspicious token timing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from asr.models import Token
from asr.vad import SpeechSpan


@dataclass(frozen=True, slots=True)
class TimingValidationPolicy:
    short_word_max_chars: int = 5
    short_word_max_duration_sec: float = 1.0
    max_word_duration_sec: float = 2.5
    max_duration_per_char_sec: float = 0.35


_EDGE_PUNCTUATION_RE = re.compile(r"^[\W_]+|[\W_]+$", flags=re.UNICODE)


def token_has_suspicious_duration(
    token: Token,
    policy: TimingValidationPolicy = TimingValidationPolicy(),
) -> bool:
    duration = token.end_time - token.start_time
    if duration <= 0.0:
        return False

    text = _normalized_text(token.text)
    if not text:
        return False

    if token.unit == "char":
        return duration > policy.max_duration_per_char_sec

    if len(text) <= policy.short_word_max_chars:
        return duration > policy.short_word_max_duration_sec

    allowed = max(
        policy.max_word_duration_sec,
        len(text) * policy.max_duration_per_char_sec,
    )
    return duration > allowed


def token_crosses_non_speech_gap(
    token: Token,
    speech_spans: Sequence[SpeechSpan],
    *,
    tolerance_sec: float = 0.10,
) -> bool:
    if token.end_time <= token.start_time:
        return False

    spans = sorted(speech_spans, key=lambda span: (span.start, span.end))
    overlapping = [
        span
        for span in spans
        if span.end + tolerance_sec >= token.start_time
        and span.start - tolerance_sec <= token.end_time
    ]
    if len(overlapping) < 2:
        return False

    for left, right in zip(overlapping, overlapping[1:]):
        gap_start = left.end
        gap_end = right.start
        if gap_end - gap_start <= tolerance_sec:
            continue
        if (
            token.start_time < gap_start - tolerance_sec
            and token.end_time > gap_end + tolerance_sec
        ):
            return True
    return False


def _normalized_text(text: str) -> str:
    return _EDGE_PUNCTUATION_RE.sub("", text.strip())
