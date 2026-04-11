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
            )
        )
        observer.on_event(
            ObservabilityEvent(
                event_type="step_start",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                step="prepare",
            )
        )

        output = stream.getvalue()
        self.assertIn("[1/2]", output)
        self.assertIn("prepare", output)

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


class MetricsCollectorObserverTest(unittest.TestCase):
    def test_writes_metrics_json_with_step_durations(self) -> None:
        collector = MetricsCollectorObserver()
        collector.on_event(ObservabilityEvent(event_type="run_start", run_id="run-1"))
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
                event_type="file_end",
                run_id="run-1",
                file_id="1",
                source_path="demo.wav",
                meta={"status": "ok"},
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


if __name__ == "__main__":
    unittest.main()
