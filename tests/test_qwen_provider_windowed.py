import unittest
from pathlib import Path
from unittest.mock import patch

from asr.providers.quality import QualityThresholds
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
        return self._responses.pop(0)


class QwenProviderWindowedTest(unittest.TestCase):
    def test_provider_processes_all_windows_not_first_window_only(self) -> None:
        provider = QwenMlxProvider()
        provider._probe_duration_sec = lambda _: 340.0
        provider._resolve_silence_anchor = lambda target, left, right: None

        asr_model = FakeModel(
            [
                FakeChunk("hello world", language="en"),
                FakeChunk("again now", language="en"),
                FakeChunk("tail done", language="en"),
            ]
        )
        align_model = FakeModel(
            [
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
            ]
        )
        provider._load_backend = (
            lambda: (
                lambda model_id: asr_model if "ASR" in model_id else align_model
            )
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

    def test_quality_gate_skips_overlap_merge_and_uses_deterministic_fallback(self) -> None:
        provider = QwenMlxProvider(
            quality_thresholds=QualityThresholds(core_context_text_divergence_max=-1.0)
        )
        provider._probe_duration_sec = lambda _: 340.0
        provider._resolve_silence_anchor = lambda target, left, right: None

        asr_model = FakeModel(
            [
                FakeChunk("hello world", language="en"),
                FakeChunk("again now", language="en"),
                FakeChunk("tail done", language="en"),
            ]
        )
        align_model = FakeModel(
            [
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
            ]
        )
        provider._load_backend = (
            lambda: (
                lambda model_id: asr_model if "ASR" in model_id else align_model
            )
        )

        with patch(
            "asr.providers.qwen_mlx.merge_adjacent_windows",
            side_effect=AssertionError("quality-gated fallback should bypass merge"),
        ):
            doc = provider.transcribe(Path("demo.wav"))

        self.assertEqual(
            [segment.text for segment in doc.segments],
            ["hello world", "again now", "tail done"],
        )
        self.assertEqual(doc.source_media["provider_metadata"]["quality_pass_count"], 0)


if __name__ == "__main__":
    unittest.main()
