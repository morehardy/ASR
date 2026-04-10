import unittest

from asr.providers.media_probe import parse_silence_anchors


class MediaProbeTest(unittest.TestCase):
    def test_parse_silence_anchors_returns_empty_list_for_empty_stderr(self) -> None:
        self.assertEqual(parse_silence_anchors(""), [])

    def test_parse_silence_anchors_reads_midpoints(self) -> None:
        stderr = """
[silencedetect @ 0x0] silence_start: 120.0
[silencedetect @ 0x0] silence_end: 121.2 | silence_duration: 1.2
[silencedetect @ 0x0] silence_start: 300.0
[silencedetect @ 0x0] silence_end: 301.0 | silence_duration: 1.0
"""

        self.assertEqual(parse_silence_anchors(stderr), [120.6, 300.5])

    def test_parse_silence_anchors_truncates_unmatched_markers(self) -> None:
        stderr = """
[silencedetect @ 0x0] silence_start: 10.0
[silencedetect @ 0x0] silence_end: 12.0 | silence_duration: 2.0
[silencedetect @ 0x0] silence_start: 30.0
"""

        self.assertEqual(parse_silence_anchors(stderr), [11.0])
