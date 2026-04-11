import io
import unittest

from asr.observability.console import ConsoleProgressObserver
from asr.observability.events import ObservabilityEvent
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


if __name__ == "__main__":
    unittest.main()
