"""Render canonical transcriptions into public output formats."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from asr.models import Segment, Token, TranscriptionDocument


def format_timestamp(seconds: float, decimal_separator: str = ",") -> str:
    total_milliseconds = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(total_milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, milliseconds = divmod(remainder, 1000)
    return (
        f"{hours:02}:{minutes:02}:{secs:02}"
        f"{decimal_separator}{milliseconds:03}"
    )


def _entry_from_segment(segment: Segment) -> Dict[str, Any]:
    return {
        "id": segment.id,
        "text": segment.text,
        "start_time": segment.start_time,
        "end_time": segment.end_time,
        "language": segment.language,
    }


def _entry_from_token(segment: Segment, token: Token, index: int) -> Dict[str, Any]:
    return {
        "id": f"{segment.id}-tok-{index}",
        "text": token.text,
        "start_time": token.start_time,
        "end_time": token.end_time,
        "language": token.language or segment.language,
    }


def render_items(document: TranscriptionDocument, granularity: str = "sentence") -> List[Dict[str, Any]]:
    if granularity == "token":
        items: List[Dict[str, Any]] = []
        for segment in document.segments:
            for index, token in enumerate(segment.tokens, start=1):
                items.append(_entry_from_token(segment, token, index))
        return items
    return [_entry_from_segment(segment) for segment in document.segments]


def render_srt(document: TranscriptionDocument, granularity: str = "sentence") -> str:
    entries = render_items(document, granularity=granularity)
    blocks = []
    for index, entry in enumerate(entries, start=1):
        blocks.append(
            "\n".join(
                [
                    str(index),
                    (
                        f"{format_timestamp(entry['start_time'])} --> "
                        f"{format_timestamp(entry['end_time'])}"
                    ),
                    entry["text"],
                ]
            )
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(document: TranscriptionDocument, granularity: str = "sentence") -> str:
    entries = render_items(document, granularity=granularity)
    blocks = ["WEBVTT"]
    for entry in entries:
        blocks.append(
            "\n".join(
                [
                    (
                        f"{format_timestamp(entry['start_time'], decimal_separator='.')}"
                        f" --> {format_timestamp(entry['end_time'], decimal_separator='.')}"
                    ),
                    entry["text"],
                ]
            )
        )
    return "\n\n".join(blocks) + "\n"


def render_json(document: TranscriptionDocument, granularity: str = "sentence") -> str:
    payload = document.to_dict()
    payload["granularity"] = granularity
    payload["items"] = render_items(document, granularity=granularity)
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
