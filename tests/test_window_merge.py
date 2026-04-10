import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from asr.models import Token

_WINDOW_MERGE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "asr" / "providers" / "window_merge.py"
)
_WINDOW_MERGE_SPEC = spec_from_file_location("window_merge_under_test", _WINDOW_MERGE_PATH)
assert _WINDOW_MERGE_SPEC is not None
assert _WINDOW_MERGE_SPEC.loader is not None
_WINDOW_MERGE_MODULE = module_from_spec(_WINDOW_MERGE_SPEC)
sys.modules[_WINDOW_MERGE_SPEC.name] = _WINDOW_MERGE_MODULE
_WINDOW_MERGE_SPEC.loader.exec_module(_WINDOW_MERGE_MODULE)

WindowSpan = _WINDOW_MERGE_MODULE.WindowSpan
merge_adjacent_windows = _WINDOW_MERGE_MODULE.merge_adjacent_windows


class WindowMergeTest(unittest.TestCase):
    def test_merge_prefers_right_core_token_when_duplicate_is_in_right_core(self) -> None:
        left_tokens = [
            Token(text="hello", start_time=0.0, end_time=0.4, unit="word"),
            Token(text="world", start_time=0.61, end_time=0.79, unit="word"),
        ]
        right_tokens = [
            Token(text="world", start_time=0.8, end_time=1.2, unit="word"),
            Token(text="again", start_time=1.21, end_time=1.5, unit="word"),
        ]
        left_span = WindowSpan(
            core_start=0.0,
            core_end=0.8,
            context_start=0.0,
            context_end=1.0,
        )
        right_span = WindowSpan(
            core_start=0.8,
            core_end=1.6,
            context_start=0.6,
            context_end=1.6,
        )

        merged = merge_adjacent_windows(
            left_tokens,
            right_tokens,
            left_span,
            right_span,
            max_time_delta=0.20,
        )

        self.assertEqual([token.text for token in merged], ["hello", "world", "again"])
        self.assertEqual([token.start_time for token in merged], [0.0, 0.8, 1.21])

    def test_merge_keeps_left_context_duplicate_and_drops_right_duplicate(self) -> None:
        left_tokens = [
            Token(text="hello", start_time=0.0, end_time=0.4, unit="word"),
            Token(text="world", start_time=0.4, end_time=0.8, unit="word"),
        ]
        right_tokens = [
            Token(text="world", start_time=0.41, end_time=0.81, unit="word"),
            Token(text="again", start_time=0.82, end_time=1.1, unit="word"),
        ]
        left_span = WindowSpan(
            core_start=0.0,
            core_end=0.8,
            context_start=0.0,
            context_end=1.0,
        )
        right_span = WindowSpan(
            core_start=0.8,
            core_end=1.4,
            context_start=0.6,
            context_end=1.4,
        )

        merged = merge_adjacent_windows(
            left_tokens,
            right_tokens,
            left_span,
            right_span,
            max_time_delta=0.20,
        )

        self.assertEqual([token.text for token in merged], ["hello", "world", "again"])

    def test_merge_keeps_boundary_token_on_right_core_start(self) -> None:
        left_tokens = [
            Token(text="alpha", start_time=0.0, end_time=0.3, unit="word"),
            Token(text="beta", start_time=0.61, end_time=0.79, unit="word"),
        ]
        right_tokens = [
            Token(text="beta", start_time=0.8, end_time=1.0, unit="word"),
        ]
        left_span = WindowSpan(
            core_start=0.0,
            core_end=0.8,
            context_start=0.0,
            context_end=1.0,
        )
        right_span = WindowSpan(
            core_start=0.8,
            core_end=1.2,
            context_start=0.6,
            context_end=1.2,
        )

        merged = merge_adjacent_windows(
            left_tokens,
            right_tokens,
            left_span,
            right_span,
            max_time_delta=0.20,
        )

        self.assertEqual([token.start_time for token in merged], [0.0, 0.8])
        self.assertEqual([token.text for token in merged], ["alpha", "beta"])

    def test_merge_respects_core_ownership_and_drops_context_only_duplicates(self) -> None:
        left_tokens = [
            Token(text="hello", start_time=0.0, end_time=0.4, unit="word"),
            Token(text="world", start_time=0.4, end_time=0.8, unit="word"),
        ]
        right_tokens = [
            Token(text="world", start_time=0.41, end_time=0.81, unit="word"),
            Token(text="again", start_time=0.82, end_time=1.1, unit="word"),
        ]
        left_span = WindowSpan(
            core_start=0.0,
            core_end=0.8,
            context_start=0.0,
            context_end=1.0,
        )
        right_span = WindowSpan(
            core_start=0.8,
            core_end=1.4,
            context_start=0.6,
            context_end=1.4,
        )

        merged = merge_adjacent_windows(
            left_tokens,
            right_tokens,
            left_span,
            right_span,
            max_time_delta=0.20,
        )

        self.assertEqual([token.text for token in merged], ["hello", "world", "again"])
