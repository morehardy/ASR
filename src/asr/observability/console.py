"""Console progress observer."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.progress import Progress, ProgressColumn, Task, TextColumn
from rich.text import Text

from asr.observability.events import ObservabilityEvent

_PROGRESS_BAR_WIDTH = 20


class _ThinProgressBarColumn(ProgressColumn):
    """Render a full-length thin progress track."""

    def render(self, task: Task) -> Text:
        total = max(1.0, float(task.total or 1.0))
        completed = max(0.0, min(float(task.completed), total))
        filled = int((completed / total) * _PROGRESS_BAR_WIDTH + 0.5)
        empty = _PROGRESS_BAR_WIDTH - filled
        bar = Text()
        bar.append("━" * filled, style="cyan")
        bar.append("─" * empty, style="dim white")
        return bar


@dataclass(slots=True)
class ConsoleProgressObserver:
    """Render file/step progress in a single terminal line."""

    stream: TextIO = field(default_factory=lambda: sys.stdout)
    warning_stream: TextIO = field(default_factory=lambda: sys.stderr)
    is_tty: bool | None = None
    _current_index: int = field(default=0, init=False, repr=False)
    _current_total: int = field(default=0, init=False, repr=False)
    _current_name: str = field(default="", init=False, repr=False)
    _current_window_index: int = field(default=0, init=False, repr=False)
    _current_window_count: int = field(default=0, init=False, repr=False)
    _file_start_perf: float | None = field(default=None, init=False, repr=False)
    _last_width: int = field(default=0, init=False, repr=False)
    _reported_warning_codes: set[str] = field(default_factory=set, init=False, repr=False)
    _console: Console | None = field(default=None, init=False, repr=False)
    _progress: Progress | None = field(default=None, init=False, repr=False)
    _progress_task_id: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.is_tty is None:
            self.is_tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def on_event(self, event: ObservabilityEvent) -> None:
        if event.event_type == "file_start":
            self._stop_progress()
            self._current_index = int(event.meta.get("index", 0))
            self._current_total = int(event.meta.get("total", 0))
            self._current_name = Path(event.source_path or "").name
            self._current_window_index = 0
            self._current_window_count = 0
            self._file_start_perf = event.perf_counter
            self._write_line(self._with_elapsed("discover", event.perf_counter))
            return

        if event.event_type == "step_start":
            if event.step == "provider_window":
                self._record_window_progress(event)
                self._write_window_progress(event.perf_counter)
                return
            self._stop_progress()
            self._write_line(self._with_elapsed(self._display_step(event), event.perf_counter))
            return

        if event.event_type == "step_error":
            self._write_step_error_warning(event)
            return

        if event.event_type == "file_end":
            status = str(event.meta.get("status", "ok"))
            if self._current_window_count > 0:
                if status == "ok":
                    self._current_window_index = self._current_window_count
                else:
                    self._stop_progress()
                    self._write_line(
                        f"{status} | {self._window_progress_line(event.perf_counter)}",
                        finalize=True,
                    )
                    return
                self._write_window_progress(event.perf_counter)
                self._stop_progress()
                return
            self._stop_progress()
            self._write_line(
                self._with_elapsed(status, event.perf_counter),
                finalize=True,
            )

    def close(self) -> None:
        self._stop_progress()

    def _display_step(self, event: ObservabilityEvent) -> str:
        return event.step or "step"

    def _record_window_progress(self, event: ObservabilityEvent) -> None:
        window_index = event.meta.get("window_index")
        window_count = event.meta.get("window_count")
        if isinstance(window_index, int):
            self._current_window_index = max(0, window_index)
        if isinstance(window_count, int):
            self._current_window_count = max(0, window_count)

    def _window_progress_line(self, perf_counter: float) -> str:
        count = self._current_window_count
        index = min(self._current_window_index, count) if count else self._current_window_index
        return (
            f"{self._progress_bar(index, count)} | "
            f"{self._progress_percent(index, count)} | "
            f"{self._progress_elapsed(perf_counter)}"
        )

    def _write_window_progress(self, perf_counter: float) -> None:
        if not self.is_tty:
            self._write_line(self._window_progress_line(perf_counter))
            return

        progress = self._ensure_progress()
        count = self._current_window_count
        total = max(1, count)
        completed = min(self._current_window_index, total)
        elapsed = self._progress_elapsed(perf_counter)
        percent = self._progress_percent(self._current_window_index, count)
        if self._progress_task_id is None:
            self._progress_task_id = progress.add_task(
                self._progress_description(),
                total=total,
                completed=completed,
                elapsed=elapsed,
                percent=percent,
            )
        else:
            progress.update(
                self._progress_task_id,
                total=total,
                completed=completed,
                elapsed=elapsed,
                percent=percent,
            )
        progress.refresh()

    def _ensure_progress(self) -> Progress:
        if self._progress is None:
            if self._last_width > 0:
                self.stream.write("\n")
                self.stream.flush()
                self._last_width = 0
            self._console = Console(
                file=self.stream,
                force_terminal=True,
            )
            self._progress = Progress(
                TextColumn("{task.description}"),
                _ThinProgressBarColumn(),
                TextColumn("{task.fields[percent]}"),
                TextColumn("{task.fields[elapsed]}"),
                console=self._console,
                auto_refresh=False,
                redirect_stdout=False,
                redirect_stderr=False,
                transient=False,
            )
            self._progress.start()
            self._progress_task_id = None
        return self._progress

    def _stop_progress(self) -> None:
        if self._progress is None:
            return
        self._progress.stop()
        if self.is_tty:
            if not self._stream_ends_with_newline():
                self.stream.write("\n")
                self.stream.flush()
            self._last_width = 0
        self._progress = None
        self._progress_task_id = None

    def _progress_description(self) -> str:
        return f"[{self._current_index}/{self._current_total}] {self._current_name}"

    def _stream_ends_with_newline(self) -> bool:
        getvalue = getattr(self.stream, "getvalue", None)
        if not callable(getvalue):
            return False
        value = str(getvalue())
        return value.endswith("\n")

    def _progress_bar(self, index: int, count: int) -> str:
        if count <= 0:
            return "─" * _PROGRESS_BAR_WIDTH
        filled = int((max(0, min(index, count)) / count) * _PROGRESS_BAR_WIDTH + 0.5)
        return ("━" * filled) + ("─" * (_PROGRESS_BAR_WIDTH - filled))

    def _progress_percent(self, index: int, count: int) -> str:
        if count <= 0:
            return "0%"
        bounded = max(0, min(index, count))
        percent = int((bounded / count) * 100 + 0.5)
        return f"{percent}%"

    def _progress_elapsed(self, perf_counter: float) -> str:
        if self._file_start_perf is None:
            return "00:00"
        elapsed = int(max(0.0, perf_counter - self._file_start_perf))
        minutes, seconds = divmod(elapsed, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _elapsed(self, perf_counter: float) -> str:
        if self._file_start_perf is None:
            return "0.0s"
        elapsed = max(0.0, perf_counter - self._file_start_perf)
        return f"{elapsed:.1f}s"

    def _with_elapsed(self, label: str, perf_counter: float) -> str:
        return f"{label} | {self._elapsed(perf_counter)}"

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

    def _write_step_error_warning(self, event: ObservabilityEvent) -> None:
        if event.step != "preprocess_vad":
            return
        error_code = str(event.meta.get("error_code", ""))
        if error_code != "vad_dependency_missing":
            return
        if error_code in self._reported_warning_codes:
            return
        self._reported_warning_codes.add(error_code)

        install_hint = str(event.meta.get("install_hint", "")).strip()
        message = (
            "[easr] warning: VAD preprocessing is unavailable because dependencies "
            "are missing (silero-vad or torchcodec); continuing with full-duration transcription."
        )
        if install_hint:
            message = f"{message} Install with: {install_hint}"
        if self.is_tty:
            self.stream.write("\n")
            self.stream.flush()
        self.warning_stream.write(message + "\n")
        self.warning_stream.flush()
