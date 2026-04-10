import unittest
from pathlib import Path
from unittest.mock import patch

from asr.providers.quality import QualityResult, QualityThresholds
from asr.providers.qwen_mlx import QwenMlxProvider


class FakeChunk:
    def __init__(
        self,
        text: str,
        language: str | None = None,
        start_time: float = 0.0,
        end_time: float = 0.0,
    ) -> None:
        self.text = text
        self.language = language
        self.start_time = start_time
        self.end_time = end_time


class FakeModel:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, dict[str, object]]] = []

    def generate(self, audio_input: str, **kwargs: object) -> object:
        self.calls.append((audio_input, dict(kwargs)))
        if not self._responses:
            raise AssertionError("FakeModel ran out of responses")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class QwenProviderWindowedTest(unittest.TestCase):
    def _build_provider_with_models(
        self,
        *,
        asr_responses: list[object],
        align_responses: list[object],
        quality_thresholds: QualityThresholds | None = None,
    ) -> tuple[QwenMlxProvider, FakeModel, FakeModel]:
        provider = QwenMlxProvider(
            quality_thresholds=quality_thresholds or QualityThresholds()
        )
        provider._probe_duration_sec = lambda _: 340.0
        provider._resolve_silence_anchor = lambda target, left, right: None

        asr_model = FakeModel(asr_responses)
        align_model = FakeModel(align_responses)
        provider._load_backend = (
            lambda: (
                lambda model_id: asr_model if "ASR" in model_id else align_model
            )
        )
        return provider, asr_model, align_model

    def test_provider_processes_all_windows_not_first_window_only(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("hello world", language="en"),
                FakeChunk("again now", language="en"),
                FakeChunk("tail done", language="en"),
            ],
            align_responses=[
                [
                    FakeChunk("hello", start_time=0.00, end_time=0.45),
                    FakeChunk("world", start_time=0.46, end_time=0.95),
                ],
                [
                    FakeChunk("again", start_time=0.00, end_time=0.35),
                    FakeChunk("now", start_time=0.36, end_time=0.72),
                ],
                [
                    FakeChunk("tail", start_time=0.00, end_time=0.30),
                    FakeChunk("done", start_time=0.31, end_time=0.68),
                ],
            ],
        )

        doc = provider.transcribe(Path("demo.wav"))
        metadata = doc.source_media["provider_metadata"]
        diagnostics = metadata["window_diagnostics"]

        self.assertEqual(len(asr_model.calls), metadata["window_count"])
        self.assertEqual(len(align_model.calls), metadata["window_count"])
        self.assertGreaterEqual(len(doc.segments), 2)
        self.assertEqual(metadata["processing_strategy"], "windowed_bounded_alignment")
        self.assertGreaterEqual(metadata["window_count"], 2)

        for (audio_input, kwargs), diagnostic in zip(asr_model.calls, diagnostics):
            self.assertIn("#t=", audio_input)
            self.assertEqual(kwargs["start_time"], diagnostic["context_start"])
            self.assertEqual(kwargs["end_time"], diagnostic["context_end"])

        for (audio_input, kwargs), diagnostic in zip(align_model.calls, diagnostics):
            self.assertIn("#t=", audio_input)
            self.assertEqual(kwargs["start_time"], diagnostic["context_start"])
            self.assertEqual(kwargs["end_time"], diagnostic["context_end"])
            self.assertIn("quality", diagnostic)

        self.assertTrue(diagnostics[0]["quality"]["passed"])
        self.assertTrue(diagnostics[-1]["quality"]["passed"])
        self.assertLess(diagnostics[0]["quality"]["boundary_disagreement_score"], 1.0)
        self.assertLess(diagnostics[-1]["quality"]["boundary_disagreement_score"], 1.0)

    def test_single_window_failure_does_not_abort_full_run_and_records_diagnostic(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("hello world", language="en"),
                RuntimeError("window exploded"),
                FakeChunk("tail done", language="en"),
            ],
            align_responses=[
                [
                    FakeChunk("hello", start_time=0.00, end_time=0.45),
                    FakeChunk("world", start_time=0.46, end_time=0.95),
                ],
                [
                    FakeChunk("tail", start_time=0.00, end_time=0.30),
                    FakeChunk("done", start_time=0.31, end_time=0.68),
                ],
            ],
        )

        doc = provider.transcribe(Path("demo.wav"))
        metadata = doc.source_media["provider_metadata"]
        diagnostics = metadata["window_diagnostics"]

        self.assertEqual(len(asr_model.calls), metadata["window_count"])
        self.assertEqual(len(align_model.calls), 2)
        self.assertEqual([segment.text for segment in doc.segments], ["hello world", "tail done"])
        self.assertEqual(metadata["quality_pass_count"], 2)
        self.assertEqual(diagnostics[1]["error"], "window exploded")
        self.assertEqual(diagnostics[1]["status"], "failed")

    def test_mixed_pass_fail_quality_still_merges_passing_windows(self) -> None:
        provider, _, _ = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("hello world", language="en"),
                FakeChunk("again now", language="en"),
                FakeChunk("tail done", language="en"),
            ],
            align_responses=[
                [
                    FakeChunk("hello", start_time=0.00, end_time=0.45),
                    FakeChunk("world", start_time=0.46, end_time=0.95),
                ],
                [
                    FakeChunk("again", start_time=0.00, end_time=0.35),
                    FakeChunk("now", start_time=0.36, end_time=0.72),
                ],
                [
                    FakeChunk("tail", start_time=0.00, end_time=0.30),
                    FakeChunk("done", start_time=0.31, end_time=0.68),
                ],
            ],
        )
        quality_results = [
            QualityResult(True, 1.0, 0.0, 0.0, 0.0),
            QualityResult(True, 1.0, 0.0, 0.0, 0.0),
            QualityResult(False, 1.0, 0.0, 1.0, 0.0),
        ]

        with patch(
            "asr.providers.qwen_mlx.evaluate_quality",
            side_effect=quality_results,
        ), patch(
            "asr.providers.qwen_mlx.merge_adjacent_windows",
            side_effect=lambda left, right, left_span, right_span, max_time_delta=0.25: left + right,
        ) as merge_mock:
            doc = provider.transcribe(Path("demo.wav"))

        self.assertEqual(merge_mock.call_count, 1)
        self.assertEqual(
            [segment.text for segment in doc.segments],
            ["hello world", "again now", "tail done"],
        )
        self.assertEqual(doc.source_media["provider_metadata"]["quality_pass_count"], 2)


if __name__ == "__main__":
    unittest.main()
