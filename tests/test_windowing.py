import unittest

from asr.providers.windowing import AlignmentWindow, WindowBudgetConfig, WindowPlanner


class WindowPlannerTest(unittest.TestCase):
    def test_all_windows_obey_hard_context_limit(self) -> None:
        cfg = WindowBudgetConfig(
            max_alignment_window_sec=180.0,
            target_core_window_sec=150.0,
            min_core_window_sec=120.0,
            context_margin_sec=20.0,
            max_context_margin_sec=30.0,
        )
        planner = WindowPlanner(cfg, anchor_resolver=None)

        windows = planner.plan(total_duration_sec=620.0)

        self.assertGreaterEqual(len(windows), 4)
        for win in windows:
            self.assertLessEqual(win.context_end - win.context_start, 180.0)

    def test_anchor_resolver_changes_split_when_available(self) -> None:
        calls: list[tuple[float, float, float]] = []

        def resolver(
            target_split_sec: float, search_start_sec: float, search_end_sec: float
        ) -> float:
            calls.append((target_split_sec, search_start_sec, search_end_sec))
            return 141.25

        cfg = WindowBudgetConfig()
        planner = WindowPlanner(cfg, anchor_resolver=resolver)

        windows = planner.plan(total_duration_sec=320.0)

        self.assertGreaterEqual(len(calls), 1)
        self.assertAlmostEqual(windows[0].core_end, 141.25)

    def test_tail_shortfall_never_mutates_previous_window_into_invalid_context(self) -> None:
        cfg = WindowBudgetConfig(
            max_alignment_window_sec=180.0,
            target_core_window_sec=150.0,
            min_core_window_sec=120.0,
            context_margin_sec=15.0,
            max_context_margin_sec=30.0,
        )
        planner = WindowPlanner(cfg, anchor_resolver=None)

        windows = planner.plan(total_duration_sec=260.0)

        self.assertGreaterEqual(len(windows), 2)
        for win in windows:
            self.assertLessEqual(win.context_end - win.context_start, 180.0)
