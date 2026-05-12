import unittest

from asr.models import Token
from asr.providers.timing_validation import (
    TimingValidationPolicy,
    token_crosses_non_speech_gap,
    token_has_suspicious_duration,
)
from asr.providers.quality import QualityThresholds, evaluate_quality
from asr.vad import SpeechSpan


class QualityGateTest(unittest.TestCase):
    def test_empty_tokens_fall_back_to_failure_defaults(self) -> None:
        result = evaluate_quality(
            tokens=[],
            left_overlap_tokens=[],
            right_overlap_tokens=[],
            core_text="",
            context_text="",
            thresholds=QualityThresholds(),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.monotonic_timestamp_ratio, 0.0)
        self.assertEqual(result.zero_or_flat_timestamp_ratio, 1.0)
        self.assertEqual(result.boundary_disagreement_score, 1.0)
        self.assertEqual(result.core_context_text_divergence, 1.0)

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

    def test_monotonic_ratio_reflects_start_time_progression(self) -> None:
        tokens = [
            Token(text="one", start_time=1.0, end_time=1.2, unit="token"),
            Token(text="two", start_time=1.1, end_time=1.3, unit="token"),
            Token(text="three", start_time=1.05, end_time=1.4, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="one two three",
            context_text="one two three",
            thresholds=QualityThresholds(),
        )

        self.assertAlmostEqual(result.monotonic_timestamp_ratio, 0.5)

    def test_zero_or_flat_ratio_counts_non_positive_durations(self) -> None:
        tokens = [
            Token(text="ok", start_time=1.0, end_time=1.2, unit="token"),
            Token(text="flat", start_time=1.3, end_time=1.3, unit="token"),
            Token(text="reversed", start_time=1.5, end_time=1.4, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="ok flat reversed",
            context_text="ok flat reversed",
            thresholds=QualityThresholds(),
        )

        self.assertAlmostEqual(result.zero_or_flat_timestamp_ratio, 2 / 3)

    def test_thresholds_drive_pass_and_fail_decisions(self) -> None:
        passing_tokens = [
            Token(text="a", start_time=0.0, end_time=0.1, unit="token"),
            Token(text="b", start_time=0.1, end_time=0.2, unit="token"),
        ]
        passing_result = evaluate_quality(
            tokens=passing_tokens,
            left_overlap_tokens=passing_tokens,
            right_overlap_tokens=passing_tokens,
            core_text="a b",
            context_text="a b",
            thresholds=QualityThresholds(),
        )

        failing_result = evaluate_quality(
            tokens=passing_tokens,
            left_overlap_tokens=passing_tokens,
            right_overlap_tokens=passing_tokens,
            core_text="a b",
            context_text="x y",
            thresholds=QualityThresholds(core_context_text_divergence_max=0.0),
        )

        self.assertTrue(passing_result.passed)
        self.assertFalse(failing_result.passed)

    def test_unresolved_tokens_cannot_pass_quality(self) -> None:
        tokens = [
            Token(text="hello", start_time=1.0, end_time=1.2, unit="token"),
            Token(text="world", start_time=1.3, end_time=1.5, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="hello world",
            context_text="hello world",
            thresholds=QualityThresholds(),
            timing_source_counts={"aligner": 1, "estimated": 0, "unresolved": 1},
            has_timing_anchor=True,
        )

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.unresolved_token_ratio, 0.5)

    def test_high_estimated_ratio_downgrades_merge_confidence(self) -> None:
        tokens = [
            Token(text="I", start_time=1.0, end_time=1.1, unit="token"),
            Token(text="have", start_time=1.2, end_time=1.5, unit="token"),
            Token(text="one", start_time=1.6, end_time=1.8, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="I have one",
            context_text="I have one",
            thresholds=QualityThresholds(),
            timing_source_counts={"aligner": 2, "estimated": 1, "unresolved": 0},
            has_timing_anchor=True,
        )

        self.assertFalse(result.passed)
        self.assertGreater(result.estimated_token_ratio, 0.30)

    def test_short_word_with_long_duration_is_suspicious(self) -> None:
        policy = TimingValidationPolicy()
        token = Token("You'd", start_time=10484.52, end_time=10498.92, unit="word")

        self.assertTrue(token_has_suspicious_duration(token, policy))

    def test_normal_word_duration_is_not_suspicious(self) -> None:
        policy = TimingValidationPolicy()
        token = Token("better", start_time=10498.92, end_time=10499.08, unit="word")

        self.assertFalse(token_has_suspicious_duration(token, policy))

    def test_token_crossing_vad_non_speech_gap_is_suspicious(self) -> None:
        token = Token("You'd", start_time=10.5, end_time=14.2, unit="word")
        speech_spans = [
            SpeechSpan(start=10.0, end=11.0),
            SpeechSpan(start=14.0, end=15.0),
        ]

        self.assertTrue(
            token_crosses_non_speech_gap(token, speech_spans, tolerance_sec=0.10)
        )
