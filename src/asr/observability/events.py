"""Observability event models."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(slots=True)
class ObservabilityEvent:
    """One emitted observability event."""

    event_type: str
    run_id: str
    file_id: Optional[str] = None
    source_path: Optional[str] = None
    step: Optional[str] = None
    timestamp: str = field(default_factory=_utc_now_iso)
    perf_counter: float = field(default_factory=time.perf_counter)
    meta: Dict[str, Any] = field(default_factory=dict)
