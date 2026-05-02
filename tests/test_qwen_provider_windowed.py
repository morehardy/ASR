import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from asr.models import Segment, Token
from asr.observability.events import ObservabilityEvent
from asr.providers.quality import QualityResult, QualityThresholds
from asr.providers.qwen_mlx import QwenMlxProvider, WindowRun
from asr.providers.windowing import AlignmentWindow
from asr.vad import DEFAULT_VAD_CONFIG, SpeechPlan, SpeechSpan, SuperChunk


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


class _ProviderRecordingObserver:
    def __init__(self) -> None:
        self.events: list[ObservabilityEvent] = []

    def on_event(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


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

    def _speech_plan(
        self,
        chunks: list[SuperChunk],
        duration_sec: float = 400.0,
    ) -> SpeechPlan:
        return SpeechPlan(
            enabled=True,
            status="ok",
            duration_sec=duration_sec,
            raw_spans=[
                SpeechSpan(start=chunk.speech_start, end=chunk.speech_end)
                for chunk in chunks
            ],
            super_chunks=chunks,
            config=DEFAULT_VAD_CONFIG,
        )

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
        self.assertNotIn("super_chunk_count", metadata)
        self.assertGreaterEqual(metadata["window_count"], 2)

        for (audio_input, kwargs), diagnostic in zip(asr_model.calls, diagnostics):
            self.assertNotIn("#t=", audio_input)
            self.assertNotIn("start_time", kwargs)
            self.assertNotIn("end_time", kwargs)

        for (audio_input, kwargs), diagnostic in zip(align_model.calls, diagnostics):
            self.assertNotIn("#t=", audio_input)
            self.assertNotIn("start_time", kwargs)
            self.assertNotIn("end_time", kwargs)
            self.assertIn("quality", diagnostic)

        self.assertTrue(diagnostics[0]["quality"]["passed"])
        self.assertTrue(diagnostics[-1]["quality"]["passed"])
        self.assertLess(diagnostics[0]["quality"]["boundary_disagreement_score"], 1.0)
        self.assertLess(diagnostics[-1]["quality"]["boundary_disagreement_score"], 1.0)

    def test_provider_processes_vad_super_chunks_on_global_timeline(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("first chunk", language="en"),
                FakeChunk("second chunk", language="en"),
            ],
            align_responses=[
                [
                    FakeChunk("first", start_time=0.00, end_time=0.40),
                    FakeChunk("chunk", start_time=0.41, end_time=0.90),
                ],
                [
                    FakeChunk("second", start_time=0.00, end_time=0.50),
                    FakeChunk("chunk", start_time=0.51, end_time=1.00),
                ],
            ],
        )
        plan = self._speech_plan(
            [
                SuperChunk(0, 105.0, 120.0, 100.0, 130.0, 1),
                SuperChunk(1, 285.0, 300.0, 280.0, 310.0, 1),
            ]
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)

        metadata = doc.source_media["provider_metadata"]
        diagnostics = metadata["window_diagnostics"]
        self.assertEqual(
            metadata["processing_strategy"],
            "vad_super_chunk_windowed_bounded_alignment",
        )
        self.assertEqual(metadata["super_chunk_count"], 2)
        self.assertEqual([item["super_chunk_index"] for item in diagnostics], [0, 1])
        self.assertEqual(len(asr_model.calls), 2)
        self.assertEqual(len(align_model.calls), 2)
        self.assertEqual(
            [
                round(token.start_time, 2)
                for segment in doc.segments
                for token in segment.tokens
            ],
            [100.0, 100.41, 280.0, 280.51],
        )

    def test_provider_splits_long_super_chunk_with_existing_hard_window_budget(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("alpha", language="en"),
                FakeChunk("beta", language="en"),
                FakeChunk("gamma", language="en"),
            ],
            align_responses=[
                [FakeChunk("alpha", start_time=0.0, end_time=0.4)],
                [FakeChunk("beta", start_time=0.0, end_time=0.4)],
                [FakeChunk("gamma", start_time=0.0, end_time=0.4)],
            ],
        )
        plan = self._speech_plan(
            [SuperChunk(0, 20.0, 330.0, 10.0, 350.0, 1)],
            duration_sec=400.0,
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)

        diagnostics = doc.source_media["provider_metadata"]["window_diagnostics"]
        self.assertGreater(len(diagnostics), 1)
        self.assertEqual(len(asr_model.calls), len(diagnostics))
        self.assertEqual(len(align_model.calls), len(diagnostics))
        for diagnostic in diagnostics:
            self.assertLessEqual(
                diagnostic["context_end"] - diagnostic["context_start"],
                provider.window_config.max_alignment_window_sec,
            )
            self.assertEqual(diagnostic["super_chunk_index"], 0)

    def test_vad_super_chunk_anchor_resolver_uses_global_timeline(self) -> None:
        provider = QwenMlxProvider()
        seen_calls: list[tuple[float, float, float]] = []

        def resolve_anchor(target: float, left: float, right: float) -> float:
            seen_calls.append((target, left, right))
            return 448.5

        provider._resolve_silence_anchor = resolve_anchor
        plan = self._speech_plan(
            [SuperChunk(0, 300.0, 610.0, 300.0, 620.0, 1)],
            duration_sec=700.0,
        )

        windows = provider._plan_windows(700.0, speech_plan=plan)

        self.assertGreaterEqual(len(windows), 2)
        self.assertEqual(seen_calls[0], (450.0, 438.0, 462.0))
        self.assertEqual(windows[0].core_end, 448.5)

    def test_provider_skips_invalid_vad_super_chunks_and_processes_valid_chunk(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                FakeChunk("valid", language="en"),
            ],
            align_responses=[
                [FakeChunk("valid", start_time=0.0, end_time=0.4)],
            ],
        )
        plan = self._speech_plan(
            [
                SuperChunk(0, 10.0, 20.0, 50.0, 40.0, 1),
                SuperChunk(1, 30.0, 40.0, float("nan"), 90.0, 1),
                SuperChunk(2, 320.0, 370.0, 310.0, 390.0, 1),
            ],
            duration_sec=400.0,
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)

        diagnostics = doc.source_media["provider_metadata"]["window_diagnostics"]
        self.assertEqual(len(asr_model.calls), 1)
        self.assertEqual(len(align_model.calls), 1)
        self.assertEqual([item["super_chunk_index"] for item in diagnostics], [2])
        self.assertEqual(diagnostics[0]["context_start"], 310.0)
        self.assertEqual(diagnostics[0]["context_end"], 340.0)

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

    def test_failed_gap_does_not_create_non_adjacent_quality_boundary_comparison(self) -> None:
        provider = QwenMlxProvider()
        left_boundary = [Token("left-edge", 10.0, 10.1, unit="word")]
        right_boundary = [Token("right-edge", 20.0, 20.1, unit="word")]
        window_runs = [
            WindowRun(
                window=AlignmentWindow(0, 0.0, 10.0, 0.0, 12.0),
                text="hello world",
                tokens=[Token("hello", 1.0, 1.2, unit="word")],
                right_overlap_tokens=left_boundary,
                core_text="hello world",
            ),
            WindowRun(
                window=AlignmentWindow(1, 10.0, 20.0, 8.0, 22.0),
                error="middle failed",
            ),
            WindowRun(
                window=AlignmentWindow(2, 20.0, 30.0, 18.0, 30.0),
                text="tail done",
                tokens=[Token("tail", 21.0, 21.2, unit="word")],
                left_overlap_tokens=right_boundary,
                core_text="tail done",
            ),
        ]

        captured_calls: list[tuple[list[object], list[object]]] = []

        def capture_quality(**kwargs: object) -> QualityResult:
            captured_calls.append(
                (
                    list(kwargs["left_overlap_tokens"]),
                    list(kwargs["right_overlap_tokens"]),
                )
            )
            return QualityResult(True, 1.0, 0.0, 0.0, 0.0)

        with patch(
            "asr.providers.qwen_mlx.evaluate_quality",
            side_effect=capture_quality,
        ):
            provider._evaluate_window_qualities(window_runs)

        self.assertEqual(len(captured_calls), 2)
        self.assertEqual(captured_calls[0], ([], []))
        self.assertEqual(captured_calls[1], ([], []))

    def test_super_chunk_gap_does_not_create_quality_boundary_comparison(self) -> None:
        provider = QwenMlxProvider()
        left_boundary = [Token("left-edge", 10.0, 10.1, unit="word")]
        right_boundary = [Token("right-edge", 20.0, 20.1, unit="word")]
        window_runs = [
            WindowRun(
                window=AlignmentWindow(0, 0.0, 10.0, 0.0, 10.0, super_chunk_index=0),
                text="left",
                tokens=[Token("left", 1.0, 1.2, unit="word")],
                right_overlap_tokens=left_boundary,
                core_text="left",
            ),
            WindowRun(
                window=AlignmentWindow(1, 20.0, 30.0, 20.0, 30.0, super_chunk_index=1),
                text="right",
                tokens=[Token("right", 21.0, 21.2, unit="word")],
                left_overlap_tokens=right_boundary,
                core_text="right",
            ),
        ]

        captured_calls: list[tuple[list[object], list[object]]] = []

        def capture_quality(**kwargs: object) -> QualityResult:
            captured_calls.append(
                (
                    list(kwargs["left_overlap_tokens"]),
                    list(kwargs["right_overlap_tokens"]),
                )
            )
            return QualityResult(True, 1.0, 0.0, 0.0, 0.0)

        with patch(
            "asr.providers.qwen_mlx.evaluate_quality",
            side_effect=capture_quality,
        ):
            provider._evaluate_window_qualities(window_runs)

        self.assertEqual(len(captured_calls), 2)
        self.assertEqual(captured_calls[0], ([], []))
        self.assertEqual(captured_calls[1], ([], []))

    def test_all_windows_fail_raises_explicit_error(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[
                RuntimeError("first failed"),
                RuntimeError("second failed"),
                RuntimeError("third failed"),
            ],
            align_responses=[],
        )

        with self.assertRaisesRegex(RuntimeError, "All transcription windows failed"):
            provider.transcribe(Path("demo.wav"))

        self.assertEqual(len(asr_model.calls), 3)
        self.assertEqual(len(align_model.calls), 0)

    def test_zero_duration_transcribe_returns_empty_document(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[],
            align_responses=[],
        )
        provider._probe_duration_sec = lambda _: 0.0

        doc = provider.transcribe(Path("demo.wav"))

        self.assertEqual(doc.segments, [])
        self.assertIsNone(doc.detected_language)
        metadata = doc.source_media["provider_metadata"]
        self.assertEqual(metadata["processing_strategy"], "windowed_bounded_alignment")
        self.assertEqual(metadata["window_count"], 0)
        self.assertEqual(metadata["duration_sec"], 0.0)
        self.assertEqual(metadata["quality_pass_count"], 0)
        self.assertEqual(metadata["failed_window_count"], 0)
        self.assertEqual(metadata["window_diagnostics"], [])
        self.assertEqual(asr_model.calls, [])
        self.assertEqual(align_model.calls, [])

    def test_fallback_merge_preserves_text_when_coercing_non_monotonic_tokens(self) -> None:
        provider = QwenMlxProvider()
        window_runs = [
            WindowRun(
                window=AlignmentWindow(0, 10.0, 11.0, 9.0, 12.0),
                text="alpha",
                tokens=[Token("alpha", 10.0, 10.2, unit="word")],
                core_tokens=[Token("alpha", 10.0, 10.2, unit="word")],
            ),
            WindowRun(
                window=AlignmentWindow(1, 11.0, 12.0, 9.0, 13.0),
                text="overlap alpha beta",
                tokens=[
                    Token("overlap", 9.5, 9.8, unit="word"),
                    Token("alpha", 10.0, 10.2, unit="word"),
                    Token("beta", 12.0, 12.2, unit="word"),
                ],
                core_tokens=[],
            ),
        ]

        merged_tokens = provider._merge_window_runs(window_runs)

        self.assertEqual(
            [token.start_time for token in merged_tokens],
            [10.0, 10.0, 10.0, 12.0],
        )
        self.assertEqual(
            [token.text for token in merged_tokens],
            ["alpha", "overlap", "alpha", "beta"],
        )
        self.assertTrue(
            all(
                merged_tokens[idx].start_time >= merged_tokens[idx - 1].start_time
                for idx in range(1, len(merged_tokens))
            )
        )

    def test_super_chunk_gap_does_not_merge_passing_blocks(self) -> None:
        provider = QwenMlxProvider()
        window_runs = [
            WindowRun(
                window=AlignmentWindow(0, 0.0, 10.0, 0.0, 10.0, super_chunk_index=0),
                text="left",
                tokens=[Token("left", 1.0, 1.2, unit="word")],
                core_tokens=[Token("left", 1.0, 1.2, unit="word")],
                quality=QualityResult(True, 1.0, 0.0, 0.0, 0.0),
            ),
            WindowRun(
                window=AlignmentWindow(1, 20.0, 30.0, 20.0, 30.0, super_chunk_index=1),
                text="right",
                tokens=[Token("right", 21.0, 21.2, unit="word")],
                core_tokens=[Token("right", 21.0, 21.2, unit="word")],
                quality=QualityResult(True, 1.0, 0.0, 0.0, 0.0),
            ),
        ]

        with patch("asr.providers.qwen_mlx.merge_adjacent_windows") as merge_mock:
            merged_tokens = provider._merge_window_runs(window_runs)

        self.assertEqual(merge_mock.call_count, 0)
        self.assertEqual([token.text for token in merged_tokens], ["left", "right"])

    def test_resolve_silence_anchor_uses_parsed_anchor_within_bounds(self) -> None:
        provider = QwenMlxProvider()
        provider._active_audio_path = Path("demo.wav")

        stderr = """
[silencedetect @ 0x0] silence_start: 143.0
[silencedetect @ 0x0] silence_end: 145.0 | silence_duration: 2.0
[silencedetect @ 0x0] silence_start: 148.0
[silencedetect @ 0x0] silence_end: 149.0 | silence_duration: 1.0
[silencedetect @ 0x0] silence_start: 170.0
[silencedetect @ 0x0] silence_end: 171.0 | silence_duration: 1.0
"""

        with patch(
            "asr.providers.qwen_mlx.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=["ffmpeg"],
                returncode=0,
                stdout="",
                stderr=stderr,
            ),
        ) as run_mock:
            anchor = provider._resolve_silence_anchor(147.4, 140.0, 150.0)

        self.assertEqual(anchor, 148.5)
        self.assertEqual(run_mock.call_count, 1)

    def test_stabilize_segments_removes_overlap_and_applies_small_tail_padding(self) -> None:
        provider = QwenMlxProvider()
        segments = [
            Segment(
                id="seg-1",
                text="hello",
                start_time=0.5,
                end_time=1.0,
                language="en",
                tokens=[],
            ),
            Segment(
                id="seg-2",
                text="world",
                start_time=0.8,
                end_time=1.4,
                language="en",
                tokens=[],
            ),
            Segment(
                id="seg-3",
                text="tail",
                start_time=2.0,
                end_time=2.1,
                language="en",
                tokens=[],
            ),
        ]

        stabilized = provider._stabilize_segment_boundaries(
            segments,
            total_duration_sec=2.2,
        )

        self.assertLessEqual(stabilized[0].end_time, stabilized[1].start_time)
        self.assertGreaterEqual(stabilized[0].end_time, 0.8)
        self.assertGreater(stabilized[2].end_time, 2.1)
        self.assertLessEqual(stabilized[2].end_time, 2.2)


class QwenProviderObservabilityTest(unittest.TestCase):
    def test_provider_emits_window_and_merge_steps(self) -> None:
        provider = QwenMlxProvider()
        provider._probe_duration_sec = lambda _: 140.0
        provider._resolve_silence_anchor = lambda target, left, right: None

        asr_model = FakeModel([FakeChunk("hello world", language="en")])
        align_model = FakeModel(
            [[FakeChunk("hello", start_time=0.0, end_time=0.4)]]
        )
        provider._load_backend = (
            lambda: (lambda model_id: asr_model if "ASR" in model_id else align_model)
        )

        observer = _ProviderRecordingObserver()
        provider.bind_observer(
            observer=observer,
            run_id="run-1",
            file_id="file-1",
            source_path="demo.wav",
        )

        provider.transcribe(Path("demo.wav"))

        step_events = [
            (event.event_type, event.step)
            for event in observer.events
            if event.event_type.startswith("step_")
        ]
        self.assertIn(("step_start", "provider_plan_windows"), step_events)
        self.assertIn(("step_end", "provider_merge"), step_events)
        self.assertTrue(any(step == "provider_window" for _, step in step_events))

    def test_provider_failed_window_emits_step_error_event(self) -> None:
        provider = QwenMlxProvider()
        provider._probe_duration_sec = lambda _: 340.0
        provider._resolve_silence_anchor = lambda target, left, right: None

        asr_model = FakeModel(
            [
                FakeChunk("hello world", language="en"),
                RuntimeError("window exploded"),
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
                    FakeChunk("tail", start_time=0.00, end_time=0.30),
                    FakeChunk("done", start_time=0.31, end_time=0.68),
                ],
            ]
        )
        provider._load_backend = (
            lambda: (lambda model_id: asr_model if "ASR" in model_id else align_model)
        )

        observer = _ProviderRecordingObserver()
        provider.bind_observer(
            observer=observer,
            run_id="run-1",
            file_id="file-1",
            source_path="demo.wav",
        )

        provider.transcribe(Path("demo.wav"))

        self.assertTrue(
            any(
                event.event_type == "step_error" and event.step == "provider_window"
                for event in observer.events
            )
        )


if __name__ == "__main__":
    unittest.main()
