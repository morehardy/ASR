"""Anchor-aware window planning for provider alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional


@dataclass(frozen=True)
class WindowBudgetConfig:
    """Budget constraints for alignment windows."""

    max_alignment_window_sec: float = 180.0
    target_core_window_sec: float = 150.0
    min_core_window_sec: float = 120.0
    context_margin_sec: float = 15.0
    max_context_margin_sec: float = 30.0
    anchor_search_radius_sec: float = 12.0


@dataclass(frozen=True)
class AlignmentWindow:
    """A planned alignment window and its surrounding context."""

    index: int
    core_start: float
    core_end: float
    context_start: float
    context_end: float


AnchorResolver = Callable[[float, float, float], Optional[float]]


class WindowPlanner:
    """Plan bounded alignment windows across a source duration."""

    def __init__(
        self,
        config: WindowBudgetConfig,
        anchor_resolver: AnchorResolver | None,
    ) -> None:
        self._config = config
        self._anchor_resolver = anchor_resolver

    def plan(self, total_duration_sec: float) -> list[AlignmentWindow]:
        if total_duration_sec <= 0.0:
            return []

        windows: list[AlignmentWindow] = []
        cursor = 0.0
        index = 0

        while cursor < total_duration_sec:
            core_end = self._choose_core_end(cursor, total_duration_sec)
            if core_end <= cursor:
                core_end = total_duration_sec

            context_start, context_end = self._apply_context_budget(
                cursor, core_end, total_duration_sec
            )
            windows.append(
                AlignmentWindow(
                    index=index,
                    core_start=cursor,
                    core_end=core_end,
                    context_start=context_start,
                    context_end=context_end,
                )
            )

            if core_end >= total_duration_sec:
                break

            cursor = core_end
            index += 1

        return windows

    def _choose_core_end(self, core_start: float, total_duration_sec: float) -> float:
        cfg = self._config
        remaining = total_duration_sec - core_start

        if remaining <= cfg.target_core_window_sec:
            return total_duration_sec

        candidate_end = core_start + cfg.target_core_window_sec
        tail_after_candidate = total_duration_sec - candidate_end
        if tail_after_candidate < cfg.min_core_window_sec:
            candidate_end = max(core_start + cfg.min_core_window_sec, total_duration_sec - cfg.min_core_window_sec)

        candidate_end = min(candidate_end, total_duration_sec)
        candidate_end = max(candidate_end, core_start)

        if self._anchor_resolver is None:
            return candidate_end

        search_start = max(
            core_start + cfg.min_core_window_sec,
            candidate_end - cfg.anchor_search_radius_sec,
        )
        search_end = min(
            total_duration_sec - cfg.min_core_window_sec,
            candidate_end + cfg.anchor_search_radius_sec,
        )

        if search_end < search_start:
            return candidate_end

        resolved = self._anchor_resolver(candidate_end, search_start, search_end)
        if resolved is None:
            return candidate_end

        if resolved <= core_start:
            return candidate_end
        if resolved > total_duration_sec:
            return candidate_end
        if resolved < search_start or resolved > search_end:
            return candidate_end

        return resolved

    def _apply_context_budget(
        self, core_start: float, core_end: float, total_duration_sec: float
    ) -> tuple[float, float]:
        cfg = self._config
        core_length = core_end - core_start
        if core_length > cfg.max_alignment_window_sec:
            raise ValueError("core window exceeds hard context limit")

        margin_budget = min(cfg.context_margin_sec, cfg.max_context_margin_sec)
        max_symmetric_margin = max(0.0, (cfg.max_alignment_window_sec - core_length) / 2.0)
        margin = min(margin_budget, max_symmetric_margin)

        context_start = max(0.0, core_start - margin)
        context_end = min(total_duration_sec, core_end + margin)

        if context_end - context_start > cfg.max_alignment_window_sec + 1e-9:
            raise ValueError("unable to fit window within hard context limit")

        return context_start, context_end
