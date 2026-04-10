"""Quality gates for boundary-aware window alignment."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Sequence

from asr.models import Token


@dataclass(frozen=True, slots=True)
class QualityThresholds:
    monotonic_timestamp_ratio_min: float = 0.98
    zero_or_flat_timestamp_ratio_max: float = 0.05
    boundary_disagreement_score_max: float = 0.20
    core_context_text_divergence_max: float = 0.15


@dataclass(frozen=True, slots=True)
class QualityResult:
    passed: bool
    monotonic_timestamp_ratio: float
    zero_or_flat_timestamp_ratio: float
    boundary_disagreement_score: float
    core_context_text_divergence: float


def evaluate_quality(
    tokens: Sequence[Token],
    left_overlap_tokens: Sequence[Token],
    right_overlap_tokens: Sequence[Token],
    core_text: str,
    context_text: str,
    thresholds: QualityThresholds,
) -> QualityResult:
    if not tokens:
        return QualityResult(False, 0.0, 1.0, 1.0, 1.0)

    monotonic_timestamp_ratio = _monotonic_timestamp_ratio(tokens)
    zero_or_flat_timestamp_ratio = _zero_or_flat_timestamp_ratio(tokens)
    boundary_disagreement_score = 1.0 - SequenceMatcher(
        None,
        _joined_token_text(left_overlap_tokens),
        _joined_token_text(right_overlap_tokens),
    ).ratio()
    core_context_text_divergence = 1.0 - SequenceMatcher(
        None,
        core_text,
        context_text,
    ).ratio()

    passed = (
        monotonic_timestamp_ratio >= thresholds.monotonic_timestamp_ratio_min
        and zero_or_flat_timestamp_ratio <= thresholds.zero_or_flat_timestamp_ratio_max
        and boundary_disagreement_score <= thresholds.boundary_disagreement_score_max
        and core_context_text_divergence <= thresholds.core_context_text_divergence_max
    )

    return QualityResult(
        passed=passed,
        monotonic_timestamp_ratio=monotonic_timestamp_ratio,
        zero_or_flat_timestamp_ratio=zero_or_flat_timestamp_ratio,
        boundary_disagreement_score=boundary_disagreement_score,
        core_context_text_divergence=core_context_text_divergence,
    )


def _monotonic_timestamp_ratio(tokens: Sequence[Token]) -> float:
    if len(tokens) < 2:
        return 1.0

    monotonic_steps = 0
    total_steps = len(tokens) - 1
    previous_start_time = tokens[0].start_time

    for token in tokens[1:]:
        if token.start_time >= previous_start_time:
            monotonic_steps += 1
        previous_start_time = token.start_time

    return monotonic_steps / total_steps


def _zero_or_flat_timestamp_ratio(tokens: Sequence[Token]) -> float:
    if not tokens:
        return 0.0

    flat_or_zero_count = sum(
        1 for token in tokens if token.start_time == token.end_time
    )
    return flat_or_zero_count / len(tokens)


def _joined_token_text(tokens: Sequence[Token]) -> str:
    return " ".join(token.text for token in tokens)

