"""Observer interfaces and fan-out dispatcher."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

from asr.observability.events import ObservabilityEvent


class Observer(Protocol):
    """Sink contract for observability events."""

    def on_event(self, event: ObservabilityEvent) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class NoopObserver:
    """Observer that drops all events."""

    def on_event(self, event: ObservabilityEvent) -> None:
        _ = event

    def close(self) -> None:
        return None


@dataclass(slots=True)
class ObserverMux:
    """Fan-out dispatcher that isolates observer failures."""

    observers: Iterable[Observer] = field(default_factory=list)
    warning_sink: Callable[[str], None] = print

    def on_event(self, event: ObservabilityEvent) -> None:
        for observer in list(self.observers):
            try:
                observer.on_event(event)
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.warning_sink(
                    f"[easr] warning: observer {type(observer).__name__} failed: {exc}"
                )

    def close(self) -> None:
        for observer in list(self.observers):
            try:
                observer.close()
            except Exception as exc:  # pragma: no cover - defensive fallback
                self.warning_sink(
                    f"[easr] warning: observer {type(observer).__name__} close failed: {exc}"
                )
