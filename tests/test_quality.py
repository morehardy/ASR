import unittest

from asr.models import Token
from asr.providers.quality import QualityThresholds, evaluate_quality


class QualityGateTest(unittest.TestCase):
    def test_boundary_disagreement_uses_adjacent_window_overlap_tokens(self) -> None:
        left_overlap = [
            Token(text="we", start_time=10.0, end_time=10.2, unit="token"),
            Token(text="agree", start_time=10.2, end_time=10.5, unit="token"),
        ]
        right_overlap = [
            Token(text="we", start_time=10.0, end_time=10.2, unit="token"),
            Token(text="argue", start_time=10.2, end_time=10.5, unit="token"),
        ]
        all_tokens = left_overlap + right_overlap

        result = evaluate_quality(
            tokens=all_tokens,
            left_overlap_tokens=left_overlap,
            right_overlap_tokens=right_overlap,
            core_text="we agree",
            context_text="we argue",
            thresholds=QualityThresholds(),
        )

        self.assertGreater(result.boundary_disagreement_score, 0.0)
        self.assertGreater(result.core_context_text_divergence, 0.0)

