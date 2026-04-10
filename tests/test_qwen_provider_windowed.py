import unittest
from pathlib import Path

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

        self.assertGreaterEqual(len(asr_model.calls), 2)
        self.assertGreaterEqual(len(align_model.calls), 2)
        self.assertGreaterEqual(len(doc.segments), 2)
        self.assertEqual(
            doc.source_media["provider_metadata"]["processing_strategy"],
            "windowed_bounded_alignment",
        )
        self.assertGreaterEqual(
            doc.source_media["provider_metadata"]["window_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
