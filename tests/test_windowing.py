import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_WINDOWING_PATH = Path(__file__).resolve().parents[1] / "src" / "asr" / "providers" / "windowing.py"
_WINDOWING_SPEC = spec_from_file_location("windowing_under_test", _WINDOWING_PATH)
assert _WINDOWING_SPEC is not None
assert _WINDOWING_SPEC.loader is not None
_WINDOWING_MODULE = module_from_spec(_WINDOWING_SPEC)
sys.modules[_WINDOWING_SPEC.name] = _WINDOWING_MODULE
_WINDOWING_SPEC.loader.exec_module(_WINDOWING_MODULE)

WindowBudgetConfig = _WINDOWING_MODULE.WindowBudgetConfig
WindowPlanner = _WINDOWING_MODULE.WindowPlanner


class WindowPlannerTest(unittest.TestCase):
    def test_non_finite_total_duration_is_rejected(self) -> None:
        planner = WindowPlanner(WindowBudgetConfig(), anchor_resolver=None)

        with self.assertRaises(ValueError):
            planner.plan(total_duration_sec=float("nan"))

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
            self.assertLessEqual(
                win.context_end - win.context_start, cfg.max_alignment_window_sec
            )

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
        self.assertAlmostEqual(windows[0].core_end, 140.0)
        self.assertAlmostEqual(windows[1].core_start, 140.0)
        self.assertAlmostEqual(windows[1].core_end, 260.0)
        for win in windows:
            self.assertLessEqual(
                win.context_end - win.context_start, cfg.max_alignment_window_sec
            )

    def test_non_finite_anchor_output_falls_back_safely(self) -> None:
        def resolver(
            target_split_sec: float, search_start_sec: float, search_end_sec: float
        ) -> float:
            return float("nan")

        cfg = WindowBudgetConfig()
        planner = WindowPlanner(cfg, anchor_resolver=resolver)

        windows = planner.plan(total_duration_sec=320.0)

        self.assertAlmostEqual(windows[0].core_end, cfg.target_core_window_sec)
