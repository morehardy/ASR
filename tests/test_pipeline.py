import unittest
from pathlib import Path

from asr.models import Segment, Token, TranscriptionDocument
from asr.observability.events import ObservabilityEvent
from asr.pipeline import _provider_accepts_speech_plan, process_media_file
from asr.providers.base import Provider
from asr.vad import DEFAULT_VAD_CONFIG, SpeechPlan, SpeechSpan, SuperChunk, failed_speech_plan


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


class FakeVadPreprocessor:
    def __init__(self, plan: SpeechPlan) -> None:
        self.plan = plan
        self.calls: list[Path] = []

    def build_plan(self, audio_path: Path) -> SpeechPlan:
        self.calls.append(audio_path)
        return self.plan


class ExplodingVadPreprocessor:
    def build_plan(self, audio_path: Path) -> SpeechPlan:
        raise RuntimeError(f"vad unavailable for {audio_path}")


class PlanAwareProvider:
    name = "plan-aware"

    def __init__(self) -> None:
        self.received_plan: SpeechPlan | None = None
        self.calls: list[Path] = []

    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
    ) -> TranscriptionDocument:
        self.calls.append(audio_path)
        self.received_plan = speech_plan
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            source_media={"provider_metadata": {"processing_strategy": "fake"}},
            segments=[],
        )


class RequiredPlanProvider:
    name = "required-plan"

    def transcribe(
        self,
        audio_path: Path,
        *,
        speech_plan: SpeechPlan,
    ) -> TranscriptionDocument:
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[],
        )


class KwargsProvider:
    name = "kwargs"

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.received_kwargs: dict[str, object] | None = None

    def transcribe(self, audio_path: Path, **kwargs: object) -> TranscriptionDocument:
        self.calls.append(audio_path)
        self.received_kwargs = kwargs
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[],
        )


class PositionalOnlyPlanProvider:
    name = "positional-only"

    def __init__(self) -> None:
        self.calls: list[Path] = []
        self.received_plan: SpeechPlan | None = None

    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
        /,
    ) -> TranscriptionDocument:
        self.calls.append(audio_path)
        self.received_plan = speech_plan
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[],
        )


class LegacyProvider:
    name = "legacy"

    def __init__(self) -> None:
        self.calls: list[Path] = []

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        self.calls.append(audio_path)
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[],
        )


class ExplodingProvider:
    name = "explode"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        raise AssertionError("provider should not be called when VAD found no speech")


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

    def test_pipeline_passes_ok_speech_plan_to_plan_aware_provider(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=100.0,
            raw_spans=[SpeechSpan(start=10.0, end=12.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=10.0,
                    speech_end=12.0,
                    chunk_start=6.0,
                    chunk_end=16.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )
        provider = PlanAwareProvider()
        vad = FakeVadPreprocessor(speech_plan)

        document = process_media_file(
            source_path=Path("clip.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=vad,
        )

        self.assertIs(provider.received_plan, speech_plan)
        self.assertEqual(vad.calls, [Path("clip.wav")])
        self.assertEqual(document.source_media["vad"]["status"], "ok")
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 1)
        self.assertEqual(document.source_media["prepared_audio_path"], "clip.wav")

    def test_required_speech_plan_parameter_is_not_optional_provider_extension(self) -> None:
        self.assertFalse(_provider_accepts_speech_plan(RequiredPlanProvider()))

    def test_pipeline_does_not_pass_speech_plan_to_kwargs_only_provider(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=100.0,
            raw_spans=[SpeechSpan(start=10.0, end=12.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=10.0,
                    speech_end=12.0,
                    chunk_start=6.0,
                    chunk_end=16.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )
        provider = KwargsProvider()

        document = process_media_file(
            source_path=Path("clip.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
        )

        self.assertEqual(provider.calls, [Path("clip.wav")])
        self.assertEqual(provider.received_kwargs, {})
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 1)

    def test_pipeline_does_not_keyword_pass_speech_plan_to_positional_only_provider(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=100.0,
            raw_spans=[SpeechSpan(start=10.0, end=12.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=10.0,
                    speech_end=12.0,
                    chunk_start=6.0,
                    chunk_end=16.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )
        provider = PositionalOnlyPlanProvider()

        document = process_media_file(
            source_path=Path("clip.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
        )

        self.assertEqual(provider.calls, [Path("clip.wav")])
        self.assertIsNone(provider.received_plan)
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 1)

    def test_pipeline_skips_provider_when_vad_finds_no_speech(self) -> None:
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=90.0,
            raw_spans=[],
            super_chunks=[],
            config=DEFAULT_VAD_CONFIG,
        )

        document = process_media_file(
            source_path=Path("quiet.mp4"),
            provider=ExplodingProvider(),
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
        )

        self.assertEqual(document.provider_name, "explode")
        self.assertEqual(document.segments, [])
        self.assertEqual(document.source_path, "quiet.mp4")
        self.assertEqual(document.source_media["vad"]["status"], "ok")
        self.assertEqual(document.source_media["vad"]["super_chunk_count"], 0)

    def test_pipeline_falls_back_without_speech_plan_when_vad_fails(self) -> None:
        failed_plan = failed_speech_plan(
            duration_sec=5.0,
            error="backend unavailable",
            config=DEFAULT_VAD_CONFIG,
        )
        observer = RecordingObserver()
        provider = LegacyProvider()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=FakeVadPreprocessor(failed_plan),
            observer=observer,
            run_id="run-1",
            file_id="file-1",
        )

        self.assertEqual(provider.calls, [Path("demo.wav")])
        self.assertEqual(document.provider_name, "legacy")
        self.assertEqual(document.source_media["vad"]["status"], "failed")
        self.assertIn("backend unavailable", document.source_media["vad"]["error"])
        vad_events = [
            event
            for event in observer.events
            if event.step == "preprocess_vad"
        ]
        self.assertEqual(
            [(event.event_type, event.step) for event in vad_events],
            [
                ("step_start", "preprocess_vad"),
                ("step_error", "preprocess_vad"),
            ],
        )
        self.assertEqual(vad_events[-1].meta["status"], "failed")
        self.assertIn("backend unavailable", vad_events[-1].meta["error"])

    def test_pipeline_converts_vad_exception_to_failed_plan(self) -> None:
        observer = RecordingObserver()
        provider = LegacyProvider()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_preprocessor=ExplodingVadPreprocessor(),
            observer=observer,
            run_id="run-1",
            file_id="file-1",
        )

        self.assertEqual(provider.calls, [Path("demo.wav")])
        self.assertEqual(document.source_media["vad"]["status"], "failed")
        self.assertIn("vad unavailable for demo.wav", document.source_media["vad"]["error"])
        vad_error = next(
            event
            for event in observer.events
            if event.event_type == "step_error" and event.step == "preprocess_vad"
        )
        self.assertEqual(vad_error.meta["status"], "failed")
        self.assertIn("vad unavailable for demo.wav", vad_error.meta["error"])

    def test_pipeline_can_disable_vad_explicitly(self) -> None:
        provider = LegacyProvider()

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=provider,
            media_preparer=FakeMediaPreparer(),
            vad_enabled=False,
        )

        self.assertEqual(provider.calls, [Path("demo.wav")])
        self.assertEqual(document.source_media["vad"]["status"], "disabled")
        self.assertFalse(document.source_media["vad"]["enabled"])


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
        speech_plan = SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=30.0,
            raw_spans=[SpeechSpan(start=1.0, end=2.0)],
            super_chunks=[
                SuperChunk(
                    index=0,
                    speech_start=1.0,
                    speech_end=2.0,
                    chunk_start=0.0,
                    chunk_end=6.0,
                    source_span_count=1,
                )
            ],
            config=DEFAULT_VAD_CONFIG,
        )

        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=FakeProvider(),
            media_preparer=FakeMediaPreparer(),
            observer=observer,
            run_id="run-1",
            file_id="file-1",
            vad_preprocessor=FakeVadPreprocessor(speech_plan),
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
                ("step_start", "preprocess_vad"),
                ("step_end", "preprocess_vad"),
                ("step_start", "transcribe"),
                ("step_end", "transcribe"),
            ],
        )
        vad_end = next(
            event
            for event in observer.events
            if event.event_type == "step_end" and event.step == "preprocess_vad"
        )
        self.assertEqual(vad_end.meta["status"], "ok")
        self.assertEqual(vad_end.meta["raw_span_count"], 1)
        self.assertEqual(vad_end.meta["super_chunk_count"], 1)
