import math
import os
import unittest

from asr.vad import (
    DEFAULT_VAD_CONFIG,
    AlignmentUnit,
    SpeechSpan,
    VadConfig,
    build_alignment_units,
    build_speech_plan,
    disabled_speech_plan,
    failed_speech_plan,
    speech_plan_metadata,
)


class VadPlanningTest(unittest.TestCase):
    def test_build_speech_plan_sanitizes_pads_merges_and_clamps_alignment_units(self) -> None:
        config = VadConfig(
            threshold=0.25,
            min_speech_duration_ms=80,
            min_silence_duration_ms=300,
            speech_pad_ms=1200,
            merge_gap_sec=3.0,
            input_padding_sec=0.8,
            max_alignment_unit_sec=180.0,
        )
        raw_spans = [
            SpeechSpan(start=10.0, end=12.0),
            SpeechSpan(start=14.5, end=15.0),
            SpeechSpan(start=20.0, end=20.0),
            SpeechSpan(start=math.nan, end=22.0),
            SpeechSpan(start=-2.0, end=1.0, confidence=math.inf),
            SpeechSpan(start=19.0, end=20.0, confidence=0.9),
            SpeechSpan(start=27.0, end=29.0),
        ]

        plan = build_speech_plan(
            duration_sec=30.0,
            raw_spans=raw_spans,
            config=config,
        )

        self.assertEqual(plan.status, "ok")
        self.assertEqual(
            [(span.start, span.end) for span in plan.raw_spans],
            [(0.0, 1.0), (10.0, 12.0), (14.5, 15.0), (19.0, 20.0), (27.0, 29.0)],
        )
        self.assertIsNone(plan.raw_spans[0].confidence)
        self.assertEqual(plan.raw_spans[3].confidence, 0.9)
        self.assertEqual(
            [
                (
                    unit.index,
                    unit.speech_start,
                    unit.speech_end,
                    unit.input_start,
                    unit.input_end,
                    unit.source_span_count,
                )
                for unit in plan.alignment_units
            ],
            [
                (0, 0.0, 1.0, 0.0, 1.8, 1),
                (1, 10.0, 15.0, 9.2, 15.8, 2),
                (2, 19.0, 20.0, 18.2, 20.8, 1),
                (3, 27.0, 29.0, 26.2, 29.8, 1),
            ],
        )
        metadata = speech_plan_metadata(plan)
        self.assertEqual(metadata["alignment_unit_count"], 4)
        self.assertNotIn("super_chunk_count", metadata)
        self.assertNotIn("super_chunks", metadata)

    def test_alignment_units_merge_spans_within_three_seconds(self) -> None:
        config = VadConfig(merge_gap_sec=3.0, input_padding_sec=0.8)

        units = build_alignment_units(
            [
                SpeechSpan(10.0, 11.0),
                SpeechSpan(13.5, 14.0),
                SpeechSpan(18.0, 19.0),
            ],
            duration_sec=30.0,
            config=config,
        )

        self.assertEqual(
            [(unit.speech_start, unit.speech_end, unit.source_span_count) for unit in units],
            [(10.0, 14.0, 2), (18.0, 19.0, 1)],
        )

    def test_alignment_unit_input_padding_overlap_is_trimmed_at_speech_midpoint(self) -> None:
        config = VadConfig(
            merge_gap_sec=3.0,
            input_padding_sec=2.0,
            max_alignment_unit_sec=180.0,
        )

        units = build_alignment_units(
            [
                SpeechSpan(10.0, 11.0),
                SpeechSpan(14.2, 15.0),
            ],
            duration_sec=30.0,
            config=config,
        )

        self.assertEqual(len(units), 2)
        self.assertEqual(units[0].input_end, 12.6)
        self.assertEqual(units[1].input_start, 12.6)

    def test_alignment_unit_splits_before_exceeding_hard_ceiling(self) -> None:
        config = VadConfig(
            merge_gap_sec=3.0,
            input_padding_sec=0.8,
            max_alignment_unit_sec=180.0,
        )

        units = build_alignment_units(
            [
                SpeechSpan(0.0, 100.0),
                SpeechSpan(102.0, 181.0),
                SpeechSpan(183.0, 185.0),
            ],
            duration_sec=200.0,
            config=config,
        )

        self.assertEqual(
            [(unit.speech_start, unit.speech_end, unit.source_span_count) for unit in units],
            [(0.0, 100.0, 1), (102.0, 185.0, 2)],
        )

    def test_alignment_unit_splits_overlapping_spans_at_hard_ceiling(self) -> None:
        config = VadConfig(
            merge_gap_sec=3.0,
            input_padding_sec=0.8,
            max_alignment_unit_sec=180.0,
        )

        units = build_alignment_units(
            [
                SpeechSpan(0.0, 179.0),
                SpeechSpan(100.0, 181.0),
            ],
            duration_sec=200.0,
            config=config,
        )

        self.assertEqual(
            [(unit.speech_start, unit.speech_end, unit.source_span_count) for unit in units],
            [(0.0, 179.0, 2), (179.0, 181.0, 1)],
        )
        for unit in units:
            self.assertLessEqual(unit.input_start, unit.speech_start)
            self.assertGreaterEqual(unit.input_end, unit.speech_end)

    def test_alignment_unit_caps_long_overlapping_chain(self) -> None:
        config = VadConfig(
            merge_gap_sec=3.0,
            input_padding_sec=0.8,
            max_alignment_unit_sec=180.0,
        )

        units = build_alignment_units(
            [
                SpeechSpan(0.0, 120.0),
                SpeechSpan(100.0, 240.0),
                SpeechSpan(220.0, 360.0),
            ],
            duration_sec=400.0,
            config=config,
        )

        self.assertGreater(len(units), 1)
        self.assertEqual(
            [(unit.speech_start, unit.speech_end) for unit in units],
            [(0.0, 120.0), (120.0, 240.0), (240.0, 360.0)],
        )
        for unit in units:
            self.assertLessEqual(unit.speech_end - unit.speech_start, 180.0)
            self.assertLessEqual(unit.input_start, unit.speech_start)
            self.assertGreaterEqual(unit.input_end, unit.speech_end)

    def test_alignment_unit_prefers_raw_gap_boundary_when_capping_chain(self) -> None:
        config = VadConfig(
            merge_gap_sec=3.0,
            input_padding_sec=0.8,
            max_alignment_unit_sec=180.0,
        )

        units = build_alignment_units(
            [
                SpeechSpan(0.0, 80.0),
                SpeechSpan(82.0, 100.0),
                SpeechSpan(90.0, 240.0),
            ],
            duration_sec=300.0,
            config=config,
        )

        self.assertEqual(
            [(unit.speech_start, unit.speech_end, unit.source_span_count) for unit in units],
            [(0.0, 80.0, 1), (82.0, 240.0, 2)],
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
        self.assertEqual(plan.alignment_units, [])

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
        self.assertEqual(len(plan.alignment_units), 1)

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
