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
        self.assertGreaterEqual(repaired[1].token.start_time, repaired[0].token.end_time + 0.02)
        self.assertLessEqual(repaired[1].token.end_time, repaired[2].token.start_time)

    def test_unmatched_middle_token_remains_unresolved_when_anchors_overlap(self) -> None:
        transcript_tokens = build_transcript_tokens("a x b", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("a", 1.0, 2.0, unit="token"),
                Token("b", 1.5, 1.8, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=3.0)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "unresolved", "aligner"])
        self.assertEqual((repaired[1].token.start_time, repaired[1].token.end_time), (0.0, 0.0))

    def test_unmatched_middle_token_uses_tight_positive_anchor_gap(self) -> None:
        transcript_tokens = build_transcript_tokens("a x b", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("a", 1.00, 1.10, unit="token"),
                Token("b", 1.13, 1.40, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=2.0)

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

    def test_trailing_tight_clip_does_not_move_estimate_before_last_match(self) -> None:
        transcript_tokens = build_transcript_tokens("hello there", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [Token("hello", 1.00, 1.30, unit="token")],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=1.20)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "unresolved"])
        self.assertEqual((repaired[1].token.start_time, repaired[1].token.end_time), (0.0, 0.0))

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

    def test_prefer_next_anchor_indexes_repairs_only_selected_middle_tokens_against_next_anchor(self) -> None:
        projected = [
            ProjectedToken(
                Token("battlefield.", 21.0, 21.5, unit="word"),
                "aligner",
                aligner_index=0,
                transcript_index=0,
            ),
            ProjectedToken(
                Token("You'd", 25.0, 39.4, unit="word"),
                "unresolved",
                transcript_index=1,
            ),
            ProjectedToken(
                Token("better", 39.4, 39.6, unit="word"),
                "aligner",
                aligner_index=2,
                transcript_index=2,
            ),
        ]

        repaired = repair_unmatched_timings(
            projected,
            clip_duration_sec=45.0,
            prefer_next_anchor_indexes={1},
        )

        self.assertEqual(repaired[1].timing_source, "estimated")
        self.assertGreaterEqual(repaired[1].start_time, 39.0)
        self.assertLessEqual(repaired[1].end_time, 39.38)

    def test_prefer_next_anchor_indexes_preserves_unselected_middle_estimates(self) -> None:
        selected_index = 2
        projected = [
            ProjectedToken(
                Token("left", 1.0, 1.4, unit="word"),
                "aligner",
                aligner_index=0,
                transcript_index=0,
            ),
            ProjectedToken(
                Token("plain", 0.0, 0.0, unit="word"),
                "unresolved",
                transcript_index=1,
            ),
            ProjectedToken(
                Token("You'd", 0.0, 0.0, unit="word"),
                "unresolved",
                transcript_index=selected_index,
            ),
            ProjectedToken(
                Token("better", 10.0, 10.2, unit="word"),
                "aligner",
                aligner_index=3,
                transcript_index=3,
            ),
        ]

        normally_repaired = repair_unmatched_timings(
            projected,
            clip_duration_sec=12.0,
        )
        preferentially_repaired = repair_unmatched_timings(
            projected,
            clip_duration_sec=12.0,
            prefer_next_anchor_indexes={selected_index},
        )

        self.assertEqual(
            (
                preferentially_repaired[1].start_time,
                preferentially_repaired[1].end_time,
            ),
            (
                normally_repaired[1].start_time,
                normally_repaired[1].end_time,
            ),
        )
        self.assertGreaterEqual(preferentially_repaired[2].start_time, 9.0)
        self.assertLessEqual(preferentially_repaired[2].end_time, 9.98)

    def test_prefer_next_anchor_indexes_does_not_overlap_previous_anchor_in_tight_gap(self) -> None:
        selected_index = 1
        projected = [
            ProjectedToken(
                Token("left", 1.00, 1.10, unit="word"),
                "aligner",
                aligner_index=0,
                transcript_index=0,
            ),
            ProjectedToken(
                Token("You'd", 0.0, 0.0, unit="word"),
                "unresolved",
                transcript_index=selected_index,
            ),
            ProjectedToken(
                Token("better", 1.15, 1.35, unit="word"),
                "aligner",
                aligner_index=2,
                transcript_index=2,
            ),
        ]

        repaired = repair_unmatched_timings(
            projected,
            clip_duration_sec=2.0,
            prefer_next_anchor_indexes={selected_index},
        )

        self.assertEqual(repaired[1].timing_source, "estimated")
        self.assertGreaterEqual(repaired[1].start_time, repaired[0].end_time)
        self.assertLessEqual(repaired[1].end_time, repaired[2].start_time)
        self.assertLessEqual(repaired[0].end_time, repaired[1].start_time)
