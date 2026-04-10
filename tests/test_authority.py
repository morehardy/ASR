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

