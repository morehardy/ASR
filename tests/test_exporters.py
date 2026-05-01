import json
import unittest

from asr.exporters import render_json, render_srt, render_vtt
from asr.models import Segment, Token, TranscriptionDocument


class ExporterTest(unittest.TestCase):
    def test_renderers_emit_expected_formats(self) -> None:
        document = TranscriptionDocument(
            source_path="demo.wav",
            provider_name="fake",
            segments=[
                Segment(
                    id="seg-1",
                    text="Hello world",
                    start_time=0.0,
                    end_time=1.5,
                    language="en",
                    tokens=[
                        Token(text="Hello", start_time=0.0, end_time=0.7, unit="word", language="en"),
                        Token(text="world", start_time=0.8, end_time=1.5, unit="word", language="en"),
                    ],
                )
            ],
        )

        srt_text = render_srt(document)
        vtt_text = render_vtt(document)
        payload = json.loads(render_json(document))

        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello world", srt_text)
        self.assertTrue(vtt_text.startswith("WEBVTT"))
        self.assertEqual(payload["segments"][0]["tokens"][0]["text"], "Hello")

    def test_token_granularity_uses_token_level_entries(self) -> None:
        document = TranscriptionDocument(
            source_path="demo.wav",
            provider_name="fake",
            segments=[
                Segment(
                    id="seg-1",
                    text="你好",
                    start_time=0.0,
                    end_time=0.6,
                    language="zh",
                    tokens=[
                        Token(text="你", start_time=0.0, end_time=0.2, unit="char", language="zh"),
                        Token(text="好", start_time=0.3, end_time=0.6, unit="char", language="zh"),
                    ],
                )
            ],
        )

        srt_text = render_srt(document, granularity="token")
        payload = json.loads(render_json(document, granularity="token"))

        self.assertIn("1\n00:00:00,000 --> 00:00:00,200\n你", srt_text)
        self.assertIn("2\n00:00:00,300 --> 00:00:00,600\n好", srt_text)
        self.assertEqual(payload["granularity"], "token")
        self.assertEqual(payload["items"][1]["text"], "好")
        self.assertIn("source_media", payload)

    def test_json_preserves_vad_metadata_without_affecting_subtitles(self) -> None:
        vad_metadata = {
            "enabled": True,
            "status": "ok",
            "duration_sec": 60.0,
            "raw_span_count": 0,
            "super_chunk_count": 0,
            "config": {"threshold": 0.25},
            "super_chunks": [],
        }
        document = TranscriptionDocument(
            source_path="quiet.wav",
            provider_name="fake",
            source_media={"vad": vad_metadata},
            segments=[],
        )

        srt_text = render_srt(document)
        vtt_text = render_vtt(document)
        payload = json.loads(render_json(document))

        self.assertEqual(srt_text, "")
        self.assertEqual(vtt_text, "WEBVTT\n")
        self.assertEqual(payload["source_media"]["vad"], vad_metadata)
