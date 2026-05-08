import unittest

from asr.models import Token
from asr.providers.authority import (
    ProjectedToken,
    build_transcript_tokens,
    project_timing_onto_transcript,
    project_timing_onto_transcript_detailed,
    repair_unmatched_timings,
)


class AuthorityTest(unittest.TestCase):
    def test_transcript_tokens_come_from_asr_text_not_aligner_items(self) -> None:
        asr_text = "I agree"
        aligner_tokens = [
            Token("I", 0.0, 0.2, unit="token"),
            Token("agreed", 0.2, 0.6, unit="token"),
        ]

        transcript_tokens = build_transcript_tokens(asr_text, language="en")
        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertEqual([token.text for token in transcript_tokens], ["I", "agree"])
        self.assertEqual([token.text for token in projected], ["I", "agree"])

    def test_projection_preserves_forward_order_and_matching_timings(self) -> None:
        transcript_tokens = build_transcript_tokens("abcdefghij abcdefghij", language="en")
        aligner_tokens = [
            Token("xbcdefghij", 0.0, 0.1, unit="token"),
            Token("abcdefghij", 0.1, 0.2, unit="token"),
        ]

        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertEqual([token.text for token in projected], ["abcdefghij", "abcdefghij"])
        self.assertEqual(
            [(token.start_time, token.end_time) for token in projected],
            [(0.0, 0.1), (0.1, 0.2)],
        )
        self.assertLess(projected[0].start_time, projected[1].start_time)
        self.assertGreater(projected[1].end_time, projected[1].start_time)

    def test_zh_tokenization_skips_whitespace(self) -> None:
        tokens = build_transcript_tokens("你 好", language="zh")

        self.assertEqual([token.text for token in tokens], ["你", "好"])

    def test_projection_matches_tokens_when_transcript_has_trailing_punctuation(self) -> None:
        transcript_tokens = build_transcript_tokens("as a developer too.", language="en")
        aligner_tokens = [
            Token("as", 16.16, 16.40, unit="token"),
            Token("a", 16.40, 16.48, unit="token"),
            Token("developer", 16.48, 17.12, unit="token"),
            Token("too", 17.12, 17.44, unit="token"),
        ]

        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertEqual([token.text for token in projected], ["as", "a", "developer", "too."])
        self.assertEqual(projected[-1].start_time, 17.12)
        self.assertEqual(projected[-1].end_time, 17.44)
        self.assertGreater(projected[-1].end_time, projected[-1].start_time)

    def test_projection_fallback_does_not_strip_programming_symbols(self) -> None:
        transcript_tokens = build_transcript_tokens("C++ C#", language="en")
        aligner_tokens = [
            Token("C", 1.00, 1.10, unit="token"),
            Token("C", 1.10, 1.20, unit="token"),
        ]

        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertEqual([token.text for token in projected], ["C++", "C#"])
        self.assertEqual(projected[0].start_time, 0.0)
        self.assertEqual(projected[0].end_time, 0.0)
        self.assertEqual(projected[1].start_time, 0.0)
        self.assertEqual(projected[1].end_time, 0.0)

    def test_detailed_projection_marks_unmatched_tokens_unresolved(self) -> None:
        transcript_tokens = build_transcript_tokens("I have", language="en")
        aligner_tokens = [
            Token("have", 5.20, 5.50, unit="token"),
        ]

        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            aligner_tokens,
        )

        self.assertIsInstance(projected[0], ProjectedToken)
        self.assertEqual([item.token.text for item in projected], ["I", "have"])
        self.assertEqual([item.timing_source for item in projected], ["unresolved", "aligner"])
        self.assertIsNone(projected[0].aligner_index)
        self.assertEqual(projected[1].aligner_index, 0)
        self.assertEqual((projected[1].token.start_time, projected[1].token.end_time), (5.20, 5.50))

    def test_existing_projection_wrapper_preserves_public_token_return_type(self) -> None:
        transcript_tokens = build_transcript_tokens("hello", language="en")
        aligner_tokens = [Token("hello", 1.0, 1.2, unit="token")]

        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertIsInstance(projected[0], Token)
        self.assertEqual(
            [(token.text, token.start_time, token.end_time) for token in projected],
            [("hello", 1.0, 1.2)],
        )

    def test_unmatched_leading_short_word_gets_interpolated(self) -> None:
        transcript_tokens = build_transcript_tokens("I have", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [Token("have", 5.20, 5.50, unit="token")],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=10.0)

        self.assertEqual([item.token.text for item in repaired], ["I", "have"])
        self.assertEqual([item.timing_source for item in repaired], ["estimated", "aligner"])
        self.assertGreaterEqual(repaired[0].token.start_time, 5.0)
        self.assertLess(repaired[0].token.end_time, repaired[1].token.start_time)
        self.assertEqual((repaired[1].token.start_time, repaired[1].token.end_time), (5.20, 5.50))

    def test_unmatched_middle_token_is_interpolated(self) -> None:
        transcript_tokens = build_transcript_tokens("we really have", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("we", 2.00, 2.12, unit="token"),
                Token("have", 2.80, 3.10, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=5.0)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "estimated", "aligner"])
        self.assertGreaterEqual(repaired[1].token.start_time, repaired[0].token.end_time)
        self.assertLessEqual(repaired[1].token.end_time, repaired[2].token.start_time)

    def test_unmatched_trailing_token_is_estimated_after_last_match(self) -> None:
        transcript_tokens = build_transcript_tokens("hello there", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [Token("hello", 1.00, 1.30, unit="token")],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=1.50)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "estimated"])
        self.assertGreaterEqual(repaired[1].token.start_time, repaired[0].token.end_time)
        self.assertLessEqual(repaired[1].token.end_time, 1.50)

    def test_fully_unmatched_tokens_remain_unresolved_for_provider_fallback(self) -> None:
        transcript_tokens = build_transcript_tokens("C++ C#", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("C", 1.00, 1.10, unit="token"),
                Token("C", 1.10, 1.20, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=2.0)

        self.assertEqual([item.token.text for item in repaired], ["C++", "C#"])
        self.assertEqual([item.timing_source for item in repaired], ["unresolved", "unresolved"])
