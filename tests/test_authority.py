import unittest

from asr.models import Token
from asr.providers.authority import (
    build_transcript_tokens,
    project_timing_onto_transcript,
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
