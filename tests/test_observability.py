import io
import json
import tempfile
import unittest
from pathlib import Path

from asr.observability.console import ConsoleProgressObserver
from asr.observability.events import ObservabilityEvent
from asr.observability.metrics import MetricsCollectorObserver
from asr.observability.observer import ObserverMux
from asr.observability.timing import observe_step


class _RecordingObserver:
    def __init__(self) -> None:
        self.events = []

    def on_event(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class _FailingObserver:
    def on_event(self, event: ObservabilityEvent) -> None:
        raise RuntimeError("boom")

    def close(self) -> None:
        raise RuntimeError("close boom")


class _CaptureObserver:
    def __init__(self) -> None:
        self.events = []

    def on_event(self, event: ObservabilityEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class ObservabilityCoreTest(unittest.TestCase):
    def test_event_has_timestamp_and_perf_counter_defaults(self) -> None:
        event = ObservabilityEvent(event_type="run_start", run_id="run-1")
        self.assertEqual(event.event_type, "run_start")
        self.assertEqual(event.run_id, "run-1")
        self.assertIsInstance(event.timestamp, str)
        self.assertGreater(event.perf_counter, 0.0)

    def test_observer_mux_isolates_failures(self) -> None:
        recorded = _RecordingObserver()
        warnings = []
        mux = ObserverMux(
            observers=[_FailingObserver(), recorded],
            warning_sink=warnings.append,
        )

        mux.on_event(ObservabilityEvent(event_type="run_start", run_id="run-1"))
        mux.close()

        self.assertEqual(len(recorded.events), 1)
        self.assertTrue(any("observer" in warning for warning in warnings))


class ObservabilityTimingTest(unittest.TestCase):
    def test_observe_step_emits_start_and_end(self) -> None:
        observer = _CaptureObserver()
        with observe_step(
            observer,
            run_id="run-1",
            file_id="file-1",
            source_path="demo.wav",
            step="prepare",
        ):
            pass

        self.assertEqual(
            [event.event_type for event in observer.events],
            ["step_start", "step_end"],
        )
        self.assertEqual(observer.events[0].step, "prepare")
        self.assertEqual(observer.events[1].step, "prepare")

    def test_observe_step_emits_step_error(self) -> None:
        observer = _CaptureObserver()

        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
            with observe_step(
                observer,
                run_id="run-1",
                file_id="file-1",
                source_path="demo.wav",
                step="prepare",
            ):
                raise RuntimeError("prepare failed")

        self.assertEqual(
            [event.event_type for event in observer.events],
            ["step_start", "step_error"],
        )
        self.assertIn("prepare failed", observer.events[-1].meta.get("error", ""))


class ConsoleProgressObserverTest(unittest.TestCase):
    def test_non_tty_falls_back_to_plain_lines(self) -> None:
        stream = io.StringIO()
        observer = ConsoleProgressObserver(stream=stream, is_tty=False)

        observer.on_event(
            ObservabilityEvent(
                event_type="file_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"index": 1, "total": 2},
                perf_counter=10.0,
            )
        )
        observer.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="prepare",
                perf_counter=10.5,
            )
        )

        output = stream.getvalue()
        self.assertIn("[1/2]", output)
        self.assertIn("prepare", output)
        self.assertIn("0.5s", output)

    def test_provider_window_step_is_rendered_with_window_index(self) -> None:
        stream = io.StringIO()
        observer = ConsoleProgressObserver(stream=stream, is_tty=False)
        observer.on_event(
            ObservabilityEvent(
                event_type="file_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"index": 1, "total": 1},
            )
        )
        observer.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="provider_window",
                meta={"window_index": 2, "window_count": 8},
            )
        )

        output = stream.getvalue()
        self.assertIn("transcribe (window 2/8)", output)

    def test_vad_missing_dependency_warning_is_rendered_once(self) -> None:
        stream = io.StringIO()
        warning_stream = io.StringIO()
        observer = ConsoleProgressObserver(
            stream=stream,
            warning_stream=warning_stream,
            is_tty=False,
        )
        event = ObservabilityEvent(
            event_type="step_error",
            run_id="run-1",
            file_id="1",
            source_path="demo.wav",
            step="preprocess_vad",
            meta={
                "error": "silero-vad is not installed",
                "error_code": "vad_dependency_missing",
                "install_hint": 'pipx install --force --python python3.14 "echoalign-asr-mlx[mlx]"',
            },
        )

        observer.on_event(event)
        observer.on_event(event)

        warning_output = warning_stream.getvalue()
        self.assertEqual(warning_output.count("VAD preprocessing is unavailable"), 1)
        self.assertIn("continuing with full-duration transcription", warning_output)
        self.assertIn("echoalign-asr-mlx[mlx]", warning_output)


class MetricsCollectorObserverTest(unittest.TestCase):
    def test_writes_metrics_json_with_run_and_provider_summaries(self) -> None:
        collector = MetricsCollectorObserver()
        collector.on_event(
            ObservabilityEvent(
                event_type="run_start",
                run_id="run-1",
                perf_counter=0.0,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                step="preflight",
                perf_counter=1.0,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id="run-1",
                step="preflight",
                perf_counter=1.2,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="file_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"index": 1, "total": 1},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="prepare",
                perf_counter=10.0,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="prepare",
                perf_counter=10.2,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="provider_window",
                perf_counter=11.0,
                meta={"window_index": 1, "window_count": 3},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_error",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="provider_window",
                perf_counter=11.6,
                meta={"window_index": 1, "window_count": 3, "error": "window exploded"},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="provider_merge",
                perf_counter=12.0,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="provider_merge",
                perf_counter=12.4,
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="file_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"status": "ok"},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="run_end",
                run_id="run-1",
                perf_counter=15.0,
            )
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "demo.metrics.json"
            collector.write_file_metrics(file_id="1", target_path=target)
            payload = json.loads(target.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["file"]["status"], "ok")
        self.assertEqual(payload["steps"][0]["name"], "prepare")
        self.assertAlmostEqual(payload["steps"][0]["duration_ms"], 200.0, delta=0.001)
        self.assertEqual(payload["run"]["steps"][0]["name"], "preflight")
        self.assertAlmostEqual(payload["run"]["steps"][0]["duration_ms"], 200.0, delta=0.001)
        self.assertAlmostEqual(payload["run"]["duration_ms"], 15000.0, delta=0.001)
        self.assertEqual(payload["provider"]["window_count"], 3)
        self.assertEqual(payload["provider"]["window_steps"][0]["status"], "failed")
        self.assertAlmostEqual(payload["provider"]["merge_duration_ms"], 400.0, delta=0.001)

    def test_writes_vad_step_metadata(self) -> None:
        collector = MetricsCollectorObserver()
        collector.on_event(ObservabilityEvent(event_type="run_start", run_id="run-1", perf_counter=0.0))
        collector.on_event(
            ObservabilityEvent(
                event_type="file_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="preprocess_vad",
                perf_counter=1.0,
                meta={"enabled": True},
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="step_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="preprocess_vad",
                perf_counter=1.4,
                meta={
                    "status": "ok",
                    "raw_span_count": 2,
                    "super_chunk_count": 1,
                },
            )
        )
        collector.on_event(
            ObservabilityEvent(
                event_type="file_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"status": "ok"},
            )
        )
        collector.on_event(ObservabilityEvent(event_type="run_end", run_id="run-1", perf_counter=2.0))

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "demo.metrics.json"
            collector.write_file_metrics(file_id="1", target_path=target)
            payload = json.loads(target.read_text(encoding="utf-8"))

        vad_step = payload["steps"][0]
        self.assertEqual(vad_step["name"], "preprocess_vad")
        self.assertEqual(vad_step["meta"]["status"], "ok")
        self.assertEqual(vad_step["meta"]["raw_span_count"], 2)
        self.assertEqual(vad_step["meta"]["super_chunk_count"], 1)


if __name__ == "__main__":
    unittest.main()
