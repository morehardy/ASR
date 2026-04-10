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
