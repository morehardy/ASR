# ASR Observability Design

Date: `2026-04-11`

## Goal

Design an observability layer for the `asr` package that improves runtime visibility and optimization workflow without breaking current CLI and export contracts.

Primary outcomes:

- default terminal progress via single-line TUI updates
- per-step timing for each media file
- structured timing export to `*.metrics.json` only when `--verbose` is enabled
- zero impact on existing transcript exports (`.srt`, `.vtt`, `.json`)

## Confirmed Decisions

- TUI progress is enabled by default (not gated behind `--verbose`).
- Existing transcript `.json` output remains unchanged and always generated.
- Additional observability JSON is exported only with `--verbose`.
- Metrics output file is independent from transcript JSON and should be named `*.metrics.json`.
- Progress style is single-line dynamic refresh (not multi-line panel).

## Non-Goals

Out of scope for this phase:

- full tracing platform integration (OpenTelemetry, Jaeger, etc.)
- token-level event streaming
- changing provider boundary or canonical transcript schema
- introducing concurrency in pipeline execution

## High-Level Approach

Use an event-driven observer architecture.

1. Add a dedicated `asr.observability` module.
2. Emit lifecycle events from CLI, pipeline, and provider steps.
3. Fan out events to multiple observers through one dispatcher.
4. Keep observer failures isolated from main transcription flow.

This provides a stable extension point for future diagnostics while preserving current behavior.

## Architecture

### Modules

- `asr/observability/events.py`
  - event dataclasses/types
  - step name constants
- `asr/observability/observer.py`
  - `Observer` protocol
  - `ObserverMux` fan-out dispatcher
- `asr/observability/timing.py`
  - step timing context helper based on `time.perf_counter()`
- `asr/observability/console.py`
  - `ConsoleProgressObserver` (single-line TUI)
- `asr/observability/metrics.py`
  - `MetricsCollectorObserver` (`--verbose` only)

### Integration Points

- `asr.cli`
  - initialize observer stack once per command run
  - emit run/file events
  - pass observer to pipeline/provider call path
- `asr.pipeline`
  - emit top-level step events: `prepare`, `transcribe`, render steps, and write step
- `asr.providers.qwen_mlx`
  - emit optional fine-grained steps: `provider_plan_windows`, per-window, and `provider_merge`

## Event Model

Each event contains:

- `event_type`: `run_start | run_end | file_start | file_end | step_start | step_end | step_error`
- `run_id`: unique ID per CLI invocation
- `file_id`: deterministic per-file identifier inside the run
- `source_path`: source media path when file-scoped
- `step`: logical step name when step-scoped
- `timestamp`: wall-clock ISO timestamp for operator correlation
- `perf_counter`: monotonic timestamp for elapsed calculation
- `meta`: small structured diagnostics payload

Notes:

- elapsed duration is computed as `step_end.perf_counter - step_start.perf_counter`
- event payloads should stay compact and avoid large blobs

## Step Taxonomy

### Run-Level

- `preflight`

### File-Level (pipeline)

- `prepare`
- `transcribe`
- `render_srt`
- `render_vtt`
- `render_json`
- `write_outputs`

### Provider-Level (optional detail)

- `provider_plan_windows`
- `provider_window` (with `meta.window_index`, `meta.window_count`)
- `provider_merge`

Provider-level events are supplemental diagnostics and must not change external contracts.

## Console Progress (Default)

### Display Rules

- Single active status line updated with carriage return in TTY environments.
- Includes file progress and current step.
- Example shape:
  - `[2/5] demo.mp4 | transcribe (window 3/8) | 42.3s`

### Refresh Policy

- immediate refresh on step boundary events
- optional refresh throttling for high-frequency provider events

### Completion Behavior

- print final line for each completed file (`done` or `failed`) before advancing
- preserve readable stderr failures from main flow

### Non-TTY Degradation

If output is not an interactive terminal, fall back to plain log lines (no in-place rewrite).

## Metrics JSON (Verbose-Only)

Generated only when `--verbose` is enabled.

### File Naming and Placement

- output alongside existing exports under the same output root
- name: `<media_basename>.metrics.json`

### Contract (v1)

Top-level fields:

- `schema_version`: `1`
- `run`: run-level summary
- `file`: current file summary
- `steps`: array of step timing records
- `provider`: optional provider diagnostic timings

`steps[]` record fields:

- `name`
- `status`: `ok | failed`
- `started_at`
- `ended_at`
- `duration_ms`
- `error` (optional)

`provider` section may include:

- `window_count`
- `window_steps[]` with per-window duration and status
- `merge_duration_ms`

The metrics schema is independent from transcript `.json` and versioned for forward evolution.

## Error Handling and Isolation

### Observer Failure Isolation

- any exception inside observer code is captured and downgraded to warning output
- observer failures never abort transcription flow

### File Failure Semantics

- keep current behavior: failed file reports error and batch continues
- emit `step_error` and `file_end(status=failed)` for diagnostics

### Metrics Write Failure

- if metrics file write fails under `--verbose`, print warning
- continue normal pipeline and transcript export

## Backward Compatibility

Guaranteed unchanged:

- CLI input discovery and file processing semantics
- exported transcript artifacts (`.srt`, `.vtt`, `.json`)
- current exit code behavior (`0` all success, `1` any failure/no files/preflight failure)

## Testing Strategy

### New Tests

- `tests/test_observability.py`
  - event timing pairing and elapsed math
  - observer mux isolation
  - non-TTY fallback rendering

### Updated CLI Tests

- default run: progress output is available without `--verbose`
- `--verbose` run: `*.metrics.json` is generated
- transcript `.json` still generated regardless of `--verbose`

### Updated Pipeline / Provider Tests

- verify expected step event order from pipeline
- verify provider window/merge timing events when windowed path executes

### Regression Guard

- existing exporter and contract tests must remain green
- no schema drift in transcript JSON

## Rollout Notes

Recommended implementation order:

1. event model + observer mux
2. console progress observer and CLI wiring
3. pipeline step instrumentation
4. provider detail instrumentation
5. verbose metrics writer
6. tests and documentation update

This ordering enables incremental validation and low-risk integration.

## Self-Review Checklist

- Placeholder scan: no TODO/TBD placeholders remain.
- Consistency scan: default TUI + verbose-only metrics behavior is consistent across sections.
- Scope scan: design remains focused on observability and does not include unrelated refactors.
- Ambiguity scan: transcript `.json` vs `*.metrics.json` responsibilities are explicitly separated.
