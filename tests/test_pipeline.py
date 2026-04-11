import unittest
from pathlib import Path

from asr.models import Segment, Token, TranscriptionDocument
from asr.observability.events import ObservabilityEvent
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


class ContractProvider(Provider):
    name = "contract"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            source_media={
                "provider_metadata": {
                    "processing_strategy": "windowed_bounded_alignment",
                    "window_count": 3,
                }
            },
            segments=[
                Segment(
                    id="seg-1",
                    text="ok",
                    start_time=0.0,
                    end_time=0.3,
                    language="en",
                    tokens=[
                        Token(
                            text="ok",
                            start_time=0.0,
                            end_time=0.3,
                            unit="token",
                            language="en",
                        )
                    ],
                )
            ],
        )


class IdentityPreparer:
    def prepare(self, source_path: Path) -> Path:
        return source_path


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

    def test_pipeline_preserves_provider_metadata_and_full_media_contract(self) -> None:
        document = process_media_file(
            source_path=Path("clip.wav"),
            provider=ContractProvider(),
            media_preparer=IdentityPreparer(),
        )

        self.assertEqual(document.source_path, "clip.wav")
        self.assertEqual(document.source_media["provider_metadata"]["window_count"], 3)
        self.assertEqual(document.source_media["prepared_audio_path"], "clip.wav")


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def on_event(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class PipelineObservabilityTest(unittest.TestCase):
    def test_pipeline_emits_prepare_then_transcribe_steps(self) -> None:
        observer = RecordingObserver()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=FakeProvider(),
            media_preparer=FakeMediaPreparer(),
            observer=observer,
            run_id="run-1",
            file_id="file-1",
        )

        self.assertEqual(document.provider_name, "fake")
        events = [
            (event.event_type, event.step)
            for event in observer.events
            if event.event_type.startswith("step_")
        ]
        self.assertEqual(
            events,
            [
                ("step_start", "prepare"),
                ("step_end", "prepare"),
                ("step_start", "transcribe"),
                ("step_end", "transcribe"),
            ],
        )
