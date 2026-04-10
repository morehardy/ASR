import unittest
from pathlib import Path

from asr.models import Segment, Token, TranscriptionDocument
from asr.pipeline import process_media_file
from asr.providers.base import Provider


class FakeProvider(Provider):
    name = "fake"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[
                Segment(
                    id="seg-1",
                    text="hello",
                    start_time=0.0,
                    end_time=0.4,
                    language="en",
                    tokens=[
                        Token(
                            text="hello",
                            start_time=0.0,
                            end_time=0.4,
                            unit="word",
                            language="en",
                        )
                    ],
                )
            ],
        )


class FakeMediaPreparer:
    def prepare(self, source_path: Path) -> Path:
        return source_path.with_suffix(".wav")


class PipelineTest(unittest.TestCase):
    def test_pipeline_uses_media_preparer_before_provider(self) -> None:
        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=FakeProvider(),
            media_preparer=FakeMediaPreparer(),
        )

        self.assertEqual(document.source_path, "demo.mp4")
        self.assertEqual(document.source_media["prepared_audio_path"], "demo.wav")
        self.assertEqual(document.provider_name, "fake")
