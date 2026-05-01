import math
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
            SpeechSpan(start=24.0, end=25.0),
            SpeechSpan(start=50.0, end=51.0),
            SpeechSpan(start=70.0, end=75.0),
        ]

        plan = build_speech_plan(
            duration_sec=60.0,
            raw_spans=raw_spans,
            config=config,
        )

        self.assertEqual(plan.status, "ok")
        self.assertEqual([(span.start, span.end) for span in plan.raw_spans], [(10.0, 12.0), (24.0, 25.0), (50.0, 51.0)])
        self.assertEqual(len(plan.super_chunks), 2)
        first = plan.super_chunks[0]
        second = plan.super_chunks[1]
        self.assertEqual(first.index, 0)
        self.assertEqual(first.source_span_count, 2)
        self.assertEqual((first.speech_start, first.speech_end), (10.0, 25.0))
        self.assertEqual((first.chunk_start, first.chunk_end), (6.0, 29.0))
        self.assertEqual(second.index, 1)
        self.assertEqual(second.source_span_count, 1)
        self.assertEqual((second.speech_start, second.speech_end), (50.0, 51.0))
        self.assertEqual((second.chunk_start, second.chunk_end), (46.0, 55.0))

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

        disabled_meta = speech_plan_metadata(disabled)
        failed_meta = speech_plan_metadata(failed)

        self.assertEqual(disabled.status, "disabled")
        self.assertFalse(disabled_meta["enabled"])
        self.assertEqual(disabled_meta["status"], "disabled")
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed_meta["status"], "failed")
        self.assertEqual(failed_meta["duration_sec"], 12.5)
        self.assertIn("silero import failed", failed_meta["error"])
        self.assertEqual(failed_meta["config"]["threshold"], 0.25)


if __name__ == "__main__":
    unittest.main()
