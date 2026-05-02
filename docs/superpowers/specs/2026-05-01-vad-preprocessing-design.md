# VAD Preprocessing Design

Date: `2026-05-01`

## Goal

Add a default VAD preprocessing stage before provider execution to address two
current long-video failure modes:

- subtitle end-time drift caused by long non-dialogue spans
- wasted ASR and forced-alignment computation over long silence or non-speech

The first implementation uses Silero VAD with a high-recall strategy. VAD is a
candidate-region detector only. It does not own final subtitle boundaries, does
not crop final output, and does not replace provider alignment.

## Core Decision

VAD is a public preprocessing capability outside providers.

- It runs after media normalization and before provider transcription.
- It is enabled by default.
- The CLI exposes only `--no-vad` in the first version.
- VAD output is a time plan, not audio clips.
- Providers may consume the time plan to restrict work to likely speech ranges.
- Exporters and the canonical transcription model remain backend-neutral.

This keeps VAD reusable for future providers while preserving provider ownership
of ASR, alignment, internal windowing, and final timestamp mapping.

## User-Facing Behavior

Default behavior:

- `easr input.mp4` runs media preparation, VAD, provider transcription, and
  exporters.
- If VAD finds speech, the provider processes only padded super-chunks.
- If VAD finds no speech, the pipeline returns a successful empty transcription
  document and writes empty subtitle outputs.
- If VAD fails, the run continues with the existing full-duration provider path.

Opt-out behavior:

- `easr --no-vad input.mp4` skips VAD and uses the current provider behavior.

No VAD tuning flags are exposed in the first version. Thresholds and padding are
observable in metadata but not part of the stable CLI contract yet.

## Architecture

The revised processing flow is:

1. Discover source media.
2. Normalize media to mono 16 kHz WAV with `MediaPreparer`.
3. Run `VadPreprocessor` unless disabled.
4. If VAD succeeds with no super-chunks, return an empty document.
5. If VAD succeeds with super-chunks, pass a `SpeechPlan` to the provider.
6. If VAD fails or is disabled, call the provider without a speech plan.
7. Render SRT, VTT, and JSON from the canonical document.

Pipeline-level VAD placement is intentional:

- `MediaPreparer` owns audio normalization.
- `VadPreprocessor` owns candidate speech detection.
- Provider owns ASR, forced alignment, provider-specific clipping, windowing,
  timestamp offsetting, quality gates, and merge behavior.
- Exporters remain unaware of VAD internals.

## Data Model

Add public preprocessing data structures in `src/asr/vad.py`.

```python
@dataclass(frozen=True, slots=True)
class VadConfig:
    threshold: float
    min_speech_duration_ms: int
    min_silence_duration_ms: int
    speech_pad_ms: int
    merge_gap_sec: float
    chunk_padding_sec: float


@dataclass(frozen=True, slots=True)
class SpeechSpan:
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SuperChunk:
    index: int
    speech_start: float
    speech_end: float
    chunk_start: float
    chunk_end: float
    source_span_count: int


@dataclass(frozen=True, slots=True)
class SpeechPlan:
    enabled: bool
    status: Literal["disabled", "ok", "failed"]
    duration_sec: float
    raw_spans: list[SpeechSpan]
    super_chunks: list[SuperChunk]
    config: VadConfig
    error: str | None = None
```

`raw_spans` are Silero-derived speech candidates on the prepared audio timeline.
`super_chunks` are padded, merged processing envelopes on the same timeline.

The model keeps both speech and processing boundaries:

- `speech_start` and `speech_end` describe VAD speech coverage.
- `chunk_start` and `chunk_end` describe the padded range handed to providers.

This distinction prevents VAD from becoming subtitle-boundary authority.

## VAD Strategy

The default parameters are intentionally high recall:

```text
threshold = 0.25
min_speech_duration_ms = 80
min_silence_duration_ms = 300
speech_pad_ms = 1200
merge_gap_sec = 12.0
chunk_padding_sec = 4.0
```

The strategy accepts false positives. Missing dialogue is more harmful than
processing a little extra non-speech.

Stage 1: lightweight speech probability scan

- Load Silero VAD from the `mlx` optional dependency set.
- Run against the prepared mono 16 kHz WAV.
- Use `get_speech_timestamps` to generate candidate speech intervals.
- Convert model timestamps to seconds on the prepared audio timeline.
- Sanitize intervals by dropping non-finite, reversed, or zero-length spans.

Stage 2: construct super-chunks

- Apply VAD speech padding from Silero and pipeline-level chunk padding.
- Merge adjacent spans when the gap is less than or equal to `merge_gap_sec`.
- Clamp merged chunks to `[0, duration_sec]`.
- Preserve raw spans and merged super-chunks for diagnostics.
- Do not enforce provider alignment-window limits in the VAD layer.

Stage 3: provider execution

- Providers consume super-chunks as processing envelopes.
- Providers keep their own alignment-window hard limits.

## Provider Consumption

`QwenMlxProvider` should support an optional speech plan while keeping backward
compatibility with the existing provider protocol.

Conceptual interface:

```python
class SpeechPlanAwareProvider(Protocol):
    name: str

    def transcribe(
        self,
        audio_path: Path,
        speech_plan: SpeechPlan | None = None,
    ) -> TranscriptionDocument:
        ...
```

The default Qwen provider behavior becomes:

- If no plan is provided, or the plan is disabled or failed, use the existing
  full-duration `WindowPlanner` path.
- If the plan is `ok` with super-chunks, plan bounded alignment windows inside
  each super-chunk.
- If the plan is `ok` with no super-chunks, the pipeline skips provider
  execution and returns an empty document.

For each super-chunk:

1. Treat `[chunk_start, chunk_end]` as the local planning duration envelope.
2. Run the existing `WindowPlanner` within that envelope.
3. Map each local window to global prepared-audio time.
4. Materialize provider window clips from the original prepared WAV using global
   `context_start` and `context_end`.
5. Run Qwen3-ASR and Qwen3-ForcedAligner on each clip.
6. Offset aligned tokens by global `context_start`.
7. Merge window results on the original prepared-audio timeline.

The existing hard invariant remains:

- `context_end - context_start <= max_alignment_window_sec`

Super-chunks do not bypass bounded alignment. A long continuous speech region
is still split into provider windows.

Super-chunks should not overlap after VAD merging. If overlap is observed anyway,
the provider should process in time order and preserve monotonic token output,
but overlap cleanup belongs primarily to VAD super-chunk construction.

## Metadata

VAD diagnostics should be recorded under `document.source_media["vad"]`, beside
provider metadata.

Example:

```json
{
  "prepared_audio_path": "...",
  "vad": {
    "enabled": true,
    "status": "ok",
    "duration_sec": 1234.56,
    "raw_span_count": 42,
    "super_chunk_count": 7,
    "config": {
      "threshold": 0.25,
      "min_speech_duration_ms": 80,
      "min_silence_duration_ms": 300,
      "speech_pad_ms": 1200,
      "merge_gap_sec": 12.0,
      "chunk_padding_sec": 4.0
    },
    "super_chunks": [
      {
        "index": 0,
        "speech_start": 103.2,
        "speech_end": 118.6,
        "chunk_start": 99.2,
        "chunk_end": 122.6,
        "source_span_count": 3
      }
    ]
  },
  "provider_metadata": {
    "processing_strategy": "vad_super_chunk_windowed_bounded_alignment",
    "window_count": 12
  }
}
```

Provider window diagnostics should include the super-chunk index when applicable.

Provider processing strategy values:

- `windowed_bounded_alignment` when VAD is disabled or unavailable
- `vad_super_chunk_windowed_bounded_alignment` when VAD super-chunks are used

Metadata is diagnostic. It must not change the public exporter shape beyond the
existing JSON document carrying additional `source_media` fields.

## Error Handling

VAD failure is non-fatal:

- Silero import failure
- model load failure
- audio read failure
- timestamp generation failure
- invalid scan output

All of these produce:

- `SpeechPlan(status="failed", error=...)`
- a warning through the existing observer or stderr path
- fallback to the full-duration provider flow

VAD success with no detected speech terminates provider execution but not the
file run:

- return a successful empty document
- record `vad.status="ok"` and `super_chunk_count=0`
- write empty subtitle outputs

Provider window failures remain window-local where possible. If every provider
window fails, keep the current explicit provider error behavior.

## Dependency Strategy

The `silero-vad` package belongs to the `mlx` optional dependency set for the
first version.

Rationale:

- the current runnable default stack already uses `.[mlx]`
- the base package remains lightweight
- users of the default runtime get VAD without discovering an extra extra
- VAD import failure still degrades safely

The implementation may choose the Silero backend mode that works best on the
target macOS and Apple Silicon environment. The public design does not require
exposing a Torch vs ONNX selection flag.

## Observability

Add a VAD step to the existing observer flow:

- `preprocess_vad`

Useful event metadata:

- enabled
- status
- duration_sec
- raw_span_count
- super_chunk_count
- elapsed time
- error, when failed

Verbose metrics should include VAD timing and counts. Default console output
should remain concise.

## Testing

Unit tests:

- span sanitization drops invalid intervals
- padding clamps to media duration
- merge gap combines nearby speech spans
- separated speech spans remain distinct super-chunks
- empty speech result creates an ok plan with no chunks
- VAD failure creates a failed plan with an error

Pipeline tests:

- default path runs VAD and passes a speech plan to a plan-aware provider
- `--no-vad` disables VAD and calls the provider without a plan
- VAD failure falls back to provider execution
- VAD success with zero super-chunks does not call the provider
- VAD metadata is present in the final document

Provider tests:

- Qwen provider plans windows inside super-chunk envelopes
- token timestamps are offset to the original prepared-audio timeline
- multiple super-chunks produce globally ordered output
- long super-chunks still respect `max_alignment_window_sec`
- diagnostics include super-chunk index when VAD is active

CLI tests:

- `--no-vad` parses successfully
- default parser behavior enables VAD

Exporter tests:

- JSON includes VAD metadata through `source_media`
- SRT and VTT rendering do not expose VAD internals

## Non-Goals

- No fine-grained public VAD parameter flags in the first version.
- No VAD-based final subtitle segmentation.
- No VAD-layer audio clipping as a public or shared responsibility.
- No provider selection redesign.
- No diarization or translation behavior changes.
- No guarantee that false positives are eliminated.

## Acceptance Criteria

- VAD runs by default before provider execution.
- `--no-vad` restores the current full-duration provider path.
- Successful VAD with no speech returns empty subtitles without running provider
  models.
- Successful VAD with speech limits provider work to padded super-chunks.
- Qwen provider still enforces bounded alignment windows inside each super-chunk.
- Final token and segment timestamps remain on the original media timeline.
- VAD failure never prevents transcription from proceeding with the old path.
- JSON output contains VAD metadata for observability and future tuning.
- Existing exporter contracts remain stable.

## Self-Review Checklist

- Placeholder scan: no placeholder requirements remain.
- Consistency scan: VAD is outside providers, but providers consume its time plan.
- Scope scan: the design covers one focused preprocessing feature.
- Ambiguity scan: no-speech, VAD failure, disabled VAD, and active VAD behavior are
  distinct and explicit.
- Boundary scan: VAD does not crop final output or own subtitle boundaries.
