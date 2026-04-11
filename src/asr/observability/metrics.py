"""Structured metrics collector for verbose observability output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from asr.observability.events import ObservabilityEvent

_RUN_SCOPE_ID = "__run__"


@dataclass(slots=True)
class MetricsCollectorObserver:
    """Collect step timing events and emit per-file metrics JSON."""

    _runs: Dict[str, dict] = field(default_factory=dict)
    _files: Dict[str, dict] = field(default_factory=dict)
    _step_starts: Dict[Tuple[str, str, str], List[ObservabilityEvent]] = field(
        default_factory=dict
    )

    def on_event(self, event: ObservabilityEvent) -> None:
        run_record = self._runs.setdefault(
            event.run_id,
            {
                "run_id": event.run_id,
                "started_at": None,
                "ended_at": None,
                "duration_ms": None,
                "steps": [],
            },
        )
        if event.event_type == "run_start":
            run_record["started_at"] = event.timestamp
            run_record["_start_perf_counter"] = event.perf_counter
            return
        if event.event_type == "run_end":
            run_record["ended_at"] = event.timestamp
            started = run_record.get("_start_perf_counter")
            if isinstance(started, (float, int)):
                run_record["duration_ms"] = (event.perf_counter - float(started)) * 1000.0
            return

        file_record = None
        scope_id = _RUN_SCOPE_ID
        if event.file_id is not None:
            scope_id = event.file_id
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

        if event.event_type == "file_end" and file_record is not None:
            file_record["status"] = str(event.meta.get("status", "unknown"))
            return

        if event.step is None:
            return

        key = (event.run_id, scope_id, event.step)
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
            merged_meta: Dict[str, Any] = dict(start.meta)
            merged_meta.update(event.meta)
            step_record = {
                "name": event.step,
                "status": "ok" if event.event_type == "step_end" else "failed",
                "started_at": start.timestamp,
                "ended_at": event.timestamp,
                "duration_ms": duration_ms,
                "error": event.meta.get("error") if event.event_type == "step_error" else None,
                "meta": merged_meta,
            }
            if file_record is not None:
                file_record["steps"].append(step_record)
            else:
                run_record["steps"].append(step_record)

    def write_file_metrics(self, *, file_id: str, target_path: Path) -> None:
        file_record = self._files[file_id]
        run_record = self._runs[file_record["run_id"]]
        payload = {
            "schema_version": 1,
            "run": {
                "run_id": run_record["run_id"],
                "started_at": run_record["started_at"],
                "ended_at": run_record["ended_at"],
                "duration_ms": run_record["duration_ms"],
                "steps": [self._serialize_step(step) for step in run_record["steps"]],
            },
            "file": {
                "file_id": file_id,
                "source_path": file_record["source_path"],
                "status": file_record["status"],
            },
            "steps": [self._serialize_step(step) for step in file_record["steps"]],
            "provider": self._build_provider_summary(file_record["steps"]),
        }
        target_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def close(self) -> None:
        return None

    def _serialize_step(self, step: dict) -> dict:
        return {
            "name": step["name"],
            "status": step["status"],
            "started_at": step["started_at"],
            "ended_at": step["ended_at"],
            "duration_ms": step["duration_ms"],
            "error": step["error"],
        }

    def _build_provider_summary(self, steps: List[dict]) -> dict:
        provider_steps = [step for step in steps if step["name"].startswith("provider_")]
        merge_step = next(
            (step for step in provider_steps if step["name"] == "provider_merge"),
            None,
        )
        window_steps = [step for step in provider_steps if step["name"] == "provider_window"]

        summary_windows = []
        max_window_count = 0
        for step in window_steps:
            meta = step.get("meta", {})
            window_count = meta.get("window_count")
            if isinstance(window_count, int):
                max_window_count = max(max_window_count, window_count)
            summary_windows.append(
                {
                    "window_index": meta.get("window_index"),
                    "window_count": window_count,
                    "status": step["status"],
                    "duration_ms": step["duration_ms"],
                    "error": step["error"],
                }
            )

        return {
            "window_count": max_window_count or len(summary_windows),
            "window_steps": summary_windows,
            "merge_duration_ms": None if merge_step is None else merge_step["duration_ms"],
        }
