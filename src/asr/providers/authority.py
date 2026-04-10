"""Helpers for keeping transcript text authoritative while borrowing timing."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Optional

from asr.models import Token


def build_transcript_tokens(text: str, language: Optional[str]) -> List[Token]:
    stripped = text.strip()
    if not stripped:
        return []

    normalized_language = _normalize_language(language)
    if _is_zh_language(normalized_language):
        units = list(stripped)
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
    projected: List[Token] = []
    aligner_index = 0

    for transcript_token in transcript_tokens:
        match_index = _find_forward_match(transcript_token.text, aligner_tokens, aligner_index)
        if match_index is None:
            projected.append(_clone_token(transcript_token))
            continue

        aligner_token = aligner_tokens[match_index]
        aligner_index = match_index + 1

        if aligner_token.end_time < aligner_token.start_time:
            projected.append(_clone_token(transcript_token))
            continue

        projected.append(
            Token(
                text=transcript_token.text,
                start_time=aligner_token.start_time,
                end_time=aligner_token.end_time,
                unit=transcript_token.unit,
                language=transcript_token.language,
            )
        )

    return projected


def _find_forward_match(
    transcript_text: str, aligner_tokens: List[Token], start_index: int
) -> Optional[int]:
    best_index: Optional[int] = None
    best_ratio = 0.0

    for index in range(start_index, len(aligner_tokens)):
        aligner_text = aligner_tokens[index].text
        ratio = SequenceMatcher(
            None, transcript_text.lower(), aligner_text.lower()
        ).ratio()
        if ratio >= 0.9 and ratio > best_ratio:
            best_index = index
            best_ratio = ratio

    return best_index


def _clone_token(token: Token) -> Token:
    return Token(
        text=token.text,
        start_time=token.start_time,
        end_time=token.end_time,
        unit=token.unit,
        language=token.language,
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

