"""Console progress observer."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from asr.observability.events import ObservabilityEvent


@dataclass(slots=True)
class ConsoleProgressObserver:
    """Render file/step progress in a single terminal line."""

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    is_tty: bool | None = None
    _current_index: int = field(default=0, init=False, repr=False)
    _current_total: int = field(default=0, init=False, repr=False)
    _current_name: str = field(default="", init=False, repr=False)
    _file_start_perf: float | None = field(default=None, init=False, repr=False)
    _last_width: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.is_tty is None:
            self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def on_event(self, event: ObservabilityEvent) -> None:
        if event.event_type == "file_start":
            self._current_index = int(event.meta.get("index", 0))
            self._current_total = int(event.meta.get("total", 0))
            self._current_name = Path(event.source_path or "").name
            self._file_start_perf = event.perf_counter
            self._write_line(self._with_elapsed("discover", event.perf_counter))
            return

        if event.event_type == "step_start":
            self._write_line(self._with_elapsed(self._display_step(event), event.perf_counter))
            return

        if event.event_type == "file_end":
            status = str(event.meta.get("status", "ok"))
            self._write_line(
                self._with_elapsed(status, event.perf_counter),
                finalize=True,
            )

    def close(self) -> None:
        return None

    def _display_step(self, event: ObservabilityEvent) -> str:
        if event.step == "provider_window":
            window_index = event.meta.get("window_index")
            window_count = event.meta.get("window_count")
            if window_index is not None and window_count is not None:
                return f"transcribe (window {window_index}/{window_count})"
        return event.step or "step"

    def _with_elapsed(self, label: str, perf_counter: float) -> str:
        if self._file_start_perf is None:
            return label
        elapsed = max(0.0, perf_counter - self._file_start_perf)
        return f"{label} | {elapsed:.1f}s"

    def _write_line(self, step: str, *, finalize: bool = False) -> None:
        line = f"[{self._current_index}/{self._current_total}] {self._current_name} | {step}"
        if self.is_tty:
            tail = "\n" if finalize else ""
            padded = line.ljust(self._last_width)
            self._last_width = max(self._last_width, len(line))
            self.stream.write(f"\r{padded}{tail}")
        else:
            self.stream.write(line + "\n")
        self.stream.flush()
