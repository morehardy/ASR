import math
import os
import unittest

from asr.vad import (
    DEFAULT_VAD_CONFIG,
    SpeechSpan,
    VadConfig,
    build_speech_plan,
    disabled_speech_plan,
    failed_speech_plan,
    speech_plan_metadata,
)


class VadPlanningTest(unittest.TestCase):
    def test_build_speech_plan_sanitizes_pads_merges_and_clamps(self) -> None:
        config = VadConfig(
            threshold=0.25,
            min_speech_duration_ms=80,
            min_silence_duration_ms=300,
            speech_pad_ms=1200,
            merge_gap_sec=12.0,
            chunk_padding_sec=4.0,
        )
        raw_spans = [
            SpeechSpan(start=10.0, end=12.0),
            SpeechSpan(start=20.0, end=20.0),
            SpeechSpan(start=math.nan, end=22.0),
            SpeechSpan(start=-2.0, end=1.0, confidence=math.inf),
            SpeechSpan(start=24.0, end=25.0),
            SpeechSpan(start=50.0, end=51.0),
            SpeechSpan(start=70.0, end=75.0),
            SpeechSpan(start=58.0, end=65.0, confidence=0.9),
        ]

        plan = build_speech_plan(
            duration_sec=60.0,
            raw_spans=raw_spans,
            config=config,
        )

        self.assertEqual(plan.status, "ok")
        self.assertEqual(
            [(span.start, span.end) for span in plan.raw_spans],
            [(0.0, 1.0), (10.0, 12.0), (24.0, 25.0), (50.0, 51.0), (58.0, 60.0)],
        )
        self.assertIsNone(plan.raw_spans[0].confidence)
        self.assertEqual(plan.raw_spans[-1].confidence, 0.9)
        self.assertEqual(len(plan.super_chunks), 2)
        first = plan.super_chunks[0]
        second = plan.super_chunks[1]
        self.assertEqual(first.index, 0)
        self.assertEqual(first.source_span_count, 3)
        self.assertEqual((first.speech_start, first.speech_end), (0.0, 25.0))
        self.assertEqual((first.chunk_start, first.chunk_end), (0.0, 29.0))
        self.assertEqual(second.index, 1)
        self.assertEqual(second.source_span_count, 2)
        self.assertEqual((second.speech_start, second.speech_end), (50.0, 60.0))
        self.assertEqual((second.chunk_start, second.chunk_end), (46.0, 60.0))
        self.assertEqual(
            speech_plan_metadata(plan)["super_chunks"],
            [
                {
                    "index": 0,
                    "speech_start": 0.0,
                    "speech_end": 25.0,
                    "chunk_start": 0.0,
                    "chunk_end": 29.0,
                    "source_span_count": 3,
                },
                {
                    "index": 1,
                    "speech_start": 50.0,
                    "speech_end": 60.0,
                    "chunk_start": 46.0,
                    "chunk_end": 60.0,
                    "source_span_count": 2,
                },
            ],
        )

    def test_build_speech_plan_sorts_sanitized_spans(self) -> None:
        plan = build_speech_plan(
            duration_sec=20.0,
            raw_spans=[
                SpeechSpan(start=12.0, end=13.0),
                SpeechSpan(start=2.0, end=3.0),
                SpeechSpan(start=8.0, end=9.0),
            ],
            config=DEFAULT_VAD_CONFIG,
        )

        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(2.0, 3.0), (8.0, 9.0), (12.0, 13.0)])

    def test_build_speech_plan_returns_ok_empty_plan_for_no_speech(self) -> None:
        plan = build_speech_plan(
            duration_sec=120.0,
            raw_spans=[],
            config=DEFAULT_VAD_CONFIG,
        )

        self.assertTrue(plan.enabled)
        self.assertEqual(plan.status, "ok")
        self.assertEqual(plan.raw_spans, [])
        self.assertEqual(plan.super_chunks, [])

    def test_disabled_and_failed_plans_serialize_to_metadata(self) -> None:
        disabled = disabled_speech_plan(config=DEFAULT_VAD_CONFIG)
        failed = failed_speech_plan(
            duration_sec=12.5,
            error="silero import failed",
            config=DEFAULT_VAD_CONFIG,
        )
        disabled_with_infinite_duration = disabled_speech_plan(
            duration_sec=math.inf,
            config=DEFAULT_VAD_CONFIG,
        )
        failed_with_infinite_duration = failed_speech_plan(
            duration_sec=math.inf,
            error="duration probe failed",
            config=DEFAULT_VAD_CONFIG,
        )

        disabled_meta = speech_plan_metadata(disabled)
        failed_meta = speech_plan_metadata(failed)
        disabled_infinite_meta = speech_plan_metadata(disabled_with_infinite_duration)
        failed_infinite_meta = speech_plan_metadata(failed_with_infinite_duration)

        self.assertEqual(disabled.status, "disabled")
        self.assertFalse(disabled_meta["enabled"])
        self.assertEqual(disabled_meta["status"], "disabled")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed_meta["status"], "failed")
        self.assertEqual(failed_meta["duration_sec"], 12.5)
        self.assertIn("silero import failed", failed_meta["error"])
        self.assertEqual(failed_meta["config"]["threshold"], 0.25)
        self.assertEqual(disabled_infinite_meta["duration_sec"], 0.0)
        self.assertEqual(failed_infinite_meta["duration_sec"], 0.0)


class SileroVadPreprocessorTest(unittest.TestCase):
    def test_silero_preprocessor_requests_second_timestamps(self) -> None:
        from asr.vad import SileroVadPreprocessor

        seen_kwargs: dict[str, object] = {}

        def timestamp_getter(wav: object, model: object, **kwargs: object) -> list[dict[str, float]]:
            seen_kwargs.update(kwargs)
            return [
                {"start": 1.0, "end": 2.0},
                {"start": 3.0, "end": 3.5},
            ]

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=lambda path, sampling_rate: [0.0] * sampling_rate,
            timestamp_getter=timestamp_getter,
            duration_probe=lambda path: 4.0,
        )

        plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(1.0, 2.0), (3.0, 3.5)])
        self.assertTrue(seen_kwargs["return_seconds"])
        self.assertEqual(plan.config.threshold, 0.25)
        self.assertEqual(len(plan.super_chunks), 1)

    def test_silero_preprocessor_accepts_second_timestamps(self) -> None:
        from asr.vad import SileroVadPreprocessor

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=lambda path, sampling_rate: [0.0] * sampling_rate,
            timestamp_getter=lambda wav, model, **kwargs: [
                {"start": 1.25, "end": 2.5},
            ],
            duration_probe=lambda path: 5.0,
        )

        plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(1.25, 2.5)])

    def test_silero_preprocessor_returns_failed_plan_when_duration_probe_raises(self) -> None:
        from asr.vad import SileroVadPreprocessor

        def unavailable_duration(path: object) -> float:
            raise RuntimeError("ffprobe unavailable")

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=lambda path, sampling_rate: [0.0] * sampling_rate,
            timestamp_getter=lambda wav, model, **kwargs: [
                {"start": 1.0, "end": 2.0},
            ],
            duration_probe=unavailable_duration,
        )

        with self.assertLogs("asr.vad", level="DEBUG") as logs:
            plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "failed")
        self.assertEqual(plan.duration_sec, 0.0)
        self.assertIn("duration probe failed", plan.error or "")
        self.assertIn("ffprobe unavailable", plan.error or "")
        self.assertTrue(
            any("VAD duration probe failed for demo.wav" in message for message in logs.output)
        )

    def test_silero_preprocessor_normalizes_pathlike_before_reading_audio(self) -> None:
        from asr.vad import SileroVadPreprocessor

        class DemoPath(os.PathLike[str]):
            def __fspath__(self) -> str:
                return "demo.wav"

            def __str__(self) -> str:
                return "not-the-filesystem-path"

        seen_paths: list[str] = []

        def audio_reader(path: str, sampling_rate: int) -> list[float]:
            seen_paths.append(path)
            return [0.0] * sampling_rate

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: "model",
            audio_reader=audio_reader,
            timestamp_getter=lambda wav, model, **kwargs: [{"start": 0.0, "end": 1.0}],
            duration_probe=lambda path: 1.0,
        )

        plan = preprocessor.build_plan(DemoPath())

        self.assertEqual(plan.status, "ok")
        self.assertEqual(seen_paths, ["demo.wav"])

    def test_silero_preprocessor_returns_failed_plan_when_backend_raises(self) -> None:
        from asr.vad import SileroVadPreprocessor

        def explode() -> object:
            raise RuntimeError("backend unavailable")

        preprocessor = SileroVadPreprocessor(
            model_loader=explode,
            duration_probe=lambda path: 9.0,
        )

        with self.assertLogs("asr.vad", level="DEBUG") as logs:
            plan = preprocessor.build_plan("demo.wav")

        self.assertEqual(plan.status, "failed")
        self.assertEqual(plan.duration_sec, 9.0)
        self.assertIn("backend unavailable", plan.error or "")
        self.assertTrue(
            any("Silero VAD preprocessing failed for demo.wav" in message for message in logs.output)
        )

    def test_silero_preprocessor_marks_missing_dependency_with_install_hint(self) -> None:
        from asr.vad import SileroVadPreprocessor, speech_plan_metadata

        def missing_silero() -> object:
            raise ModuleNotFoundError("No module named 'silero_vad'", name="silero_vad")

        preprocessor = SileroVadPreprocessor(
            model_loader=missing_silero,
            duration_probe=lambda path: 9.0,
        )

        plan = preprocessor.build_plan("demo.wav")
        metadata = speech_plan_metadata(plan)

        self.assertEqual(plan.status, "failed")
        self.assertEqual(plan.error_code, "vad_dependency_missing")
        self.assertIn("VAD dependencies are missing", plan.error or "")
        self.assertIn("silero-vad", plan.error or "")
        self.assertIn("echoalign-asr-mlx[mlx]", plan.install_hint or "")
        self.assertEqual(metadata["error_code"], "vad_dependency_missing")
        self.assertIn("echoalign-asr-mlx[mlx]", metadata["install_hint"])

    def test_silero_preprocessor_marks_missing_torchcodec_with_install_hint(self) -> None:
        from asr.vad import SileroVadPreprocessor, speech_plan_metadata

        def missing_torchcodec(*args: object, **kwargs: object) -> object:
            raise RuntimeError(
                "torchaudio version 2.11.0 requires torchcodec for audio I/O. "
                "Install torchcodec or pin torchaudio < 2.9"
            )

        preprocessor = SileroVadPreprocessor(
            model_loader=lambda: object(),
            audio_reader=missing_torchcodec,
            duration_probe=lambda path: 9.0,
        )

        plan = preprocessor.build_plan("demo.wav")
        metadata = speech_plan_metadata(plan)

        self.assertEqual(plan.status, "failed")
        self.assertEqual(plan.error_code, "vad_dependency_missing")
        self.assertIn("VAD dependencies are missing", plan.error or "")
        self.assertIn("torchcodec", plan.error or "")
        self.assertIn("echoalign-asr-mlx[mlx]", plan.install_hint or "")
        self.assertEqual(metadata["error_code"], "vad_dependency_missing")
        self.assertIn("echoalign-asr-mlx[mlx]", metadata["install_hint"])


if __name__ == "__main__":
    unittest.main()
