"""Structured metrics collector for verbose observability output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from asr.observability.events import ObservabilityEvent


@dataclass(slots=True)
class MetricsCollectorObserver:
    """Collect step timing events and emit per-file metrics JSON."""

    _files: Dict[str, dict] = field(default_factory=dict)
    _step_starts: Dict[Tuple[str, str], List[ObservabilityEvent]] = field(default_factory=dict)

    def on_event(self, event: ObservabilityEvent) -> None:
        if event.file_id is None:
            return

        file_record = self._files.setdefault(
            event.file_id,
            {
                "run_id": event.run_id,
                "source_path": event.source_path,
                "status": "unknown",
                "steps": [],
            },
        )
        if file_record.get("source_path") is None and event.source_path is not None:
            file_record["source_path"] = event.source_path

        if event.event_type == "file_end":
            file_record["status"] = str(event.meta.get("status", "unknown"))
            return

        if event.step is None:
            return

        key = (event.file_id, event.step)
        if event.event_type == "step_start":
            self._step_starts.setdefault(key, []).append(event)
            return

        if event.event_type in {"step_end", "step_error"}:
            starts = self._step_starts.get(key)
            if not starts:
                return
            start = starts.pop()
            if not starts:
                self._step_starts.pop(key, None)

            duration_ms = (event.perf_counter - start.perf_counter) * 1000.0
            file_record["steps"].append(
                {
                    "name": event.step,
                    "status": "ok" if event.event_type == "step_end" else "failed",
                    "started_at": start.timestamp,
                    "ended_at": event.timestamp,
                    "duration_ms": duration_ms,
                    "error": event.meta.get("error") if event.event_type == "step_error" else None,
                }
            )

    def write_file_metrics(self, *, file_id: str, target_path: Path) -> None:
        file_record = self._files[file_id]
        payload = {
            "schema_version": 1,
            "run": {"run_id": file_record["run_id"]},
            "file": {
                "file_id": file_id,
                "source_path": file_record["source_path"],
                "status": file_record["status"],
            },
            "steps": file_record["steps"],
        }
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def close(self) -> None:
        return None
