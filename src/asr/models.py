"""Canonical data structures for transcriptions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Token:
    text: str
    start_time: float
    end_time: float
    unit: str
    language: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Segment:
    id: str
    text: str
    start_time: float
    end_time: float
    language: Optional[str]
    tokens: List[Token] = field(default_factory=list)
    speaker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tokens"] = [token.to_dict() for token in self.tokens]
        return payload


@dataclass(slots=True)
class TranscriptionDocument:
    source_path: str
    provider_name: str
    segments: List[Segment]
    source_media: Optional[Dict[str, Any]] = None
    detected_language: Optional[str] = None

    def ensure_source_media(self) -> Dict[str, Any]:
        if self.source_media is None:
            self.source_media = {}
        return self.source_media

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "provider_name": self.provider_name,
            "source_media": self.source_media,
            "detected_language": self.detected_language,
            "segments": [segment.to_dict() for segment in self.segments],
        }
