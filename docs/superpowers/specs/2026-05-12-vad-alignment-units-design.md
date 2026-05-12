# VAD Alignment Units Design

Date: `2026-05-12`

## Purpose

Replace coarse VAD super-chunk provider input with shorter VAD-derived
alignment units. The goal is to reduce forced-alignment search space so short
words cannot smear across long silent or unrelated regions, while preserving the
high-recall VAD behavior that avoids dropping dialogue.

This design responds to a real failure where a one-word subtitle was emitted as:

```text
02:54:44.520 --> 02:54:58.920
You'd
```

The generated JSON showed the error already existed in canonical token timing:

```text
You'd = 10484.52 -> 10498.92
better = 10498.92 -> 10499.08
```

That token came from a VAD-derived provider window covering about 135 seconds of
audio. The root problem is not SRT rendering. It is that VAD spans are currently
merged into large processing envelopes before ASR and forced alignment, so the
aligner can attach a short transcript token to a much earlier region.

## Goals

- Make short VAD-derived alignment units the provider input authority.
- Keep VAD high-recall: false positives are acceptable; missed dialogue is not.
- Merge only nearby spans by default, using a small gap such as `3.0s`.
- Treat `max_alignment_unit_sec = 180.0` as a hard ceiling, not a target size.
- Preserve the separation between model input bounds and subtitle display
  bounds.
- Keep exporters unchanged; SRT, VTT, TXT, and public JSON continue to render
  canonical segments and tokens.
- Keep token-level timing validation as a safety net after alignment.
- Remove old super-chunk compatibility from the new provider path.

## Non-Goals

- Do not expose VAD tuning flags through the CLI in this design.
- Do not move timing correction into exporters.
- Do not make VAD the transcript authority.
- Do not require diarization, translation, or speaker-aware splitting.
- Do not optimize model call batching before fixing alignment correctness.
- Do not preserve a `super_chunks` fallback path in the Qwen provider.

## Current Behavior

The current VAD flow is:

```text
raw VAD spans
-> padded super_chunks with merge_gap_sec = 12.0 and chunk_padding_sec = 4.0
-> provider windows inside each super_chunk
-> ASR + forced aligner on each provider window
-> token merge
-> segment construction
-> exporters
```

This reduces work over long silence, but it does not constrain alignment tightly
enough. A super chunk may contain many raw speech spans. In the observed failure,
one super chunk contained 30 source spans and produced a provider window from
`10458.6` to `10593.6`. The aligner could search across that whole clip.

Quality gates then missed the issue because the bad token was monotonic and had
aligner provenance. It was not zero-duration, estimated, or unresolved.

## Proposed Behavior

The new flow is:

```text
raw VAD spans
-> alignment_units
-> provider windows inside each alignment unit when needed
-> ASR + forced aligner
-> token timing validation and repair
-> token merge
-> segment construction
-> exporters
```

`alignment_units` are the only VAD-derived provider processing units. The old
`super_chunks` model should be removed from the new design rather than kept as a
provider fallback.

### Alignment Unit Semantics

An alignment unit is a short, padded processing envelope around one or more
nearby VAD speech spans.

```python
@dataclass(frozen=True, slots=True)
class AlignmentUnit:
    index: int
    speech_start: float
    speech_end: float
    input_start: float
    input_end: float
    source_span_count: int
```

Field meaning:

- `speech_start` and `speech_end` are the union of the raw VAD speech spans.
- `input_start` and `input_end` add small model-input padding.
- `source_span_count` records how many raw spans contributed to the unit.
- `index` is stable time order after sanitation and merging.

`SpeechPlan` becomes:

```python
@dataclass(frozen=True, slots=True)
class SpeechPlan:
    enabled: bool
    status: Literal["disabled", "ok", "failed"]
    duration_sec: float
    raw_spans: list[SpeechSpan]
    alignment_units: list[AlignmentUnit]
    config: VadConfig
    error: str | None = None
    error_code: str | None = None
    install_hint: str | None = None
```

`super_chunks` should be removed from the new data model and tests.

## Default Parameters

Recommended internal defaults:

```text
threshold = 0.25
min_speech_duration_ms = 80
min_silence_duration_ms = 300
speech_pad_ms = 1200
merge_gap_sec = 3.0
input_padding_sec = 0.8
max_alignment_unit_sec = 180.0
```

Important interpretation:

- `merge_gap_sec = 3.0` is the ordinary grouping rule.
- `max_alignment_unit_sec = 180.0` is only a hard ceiling.
- The builder must not intentionally accumulate spans until an alignment unit is
  near 180 seconds.
- If ordinary 3-second merging creates a unit longer than 180 seconds, split it
  at the best available raw span boundary or silence anchor.

`chunk_padding_sec` should be renamed to `input_padding_sec` to describe its new
role. It pads model input, not a large chunk.

## Unit Construction Algorithm

1. Sanitize raw VAD spans on the prepared audio timeline.
2. Sort spans by `(start, end)`.
3. Start a new candidate unit with the first span.
4. For each next span:
   - Compute the speech gap from current `speech_end` to next `span.start`.
   - If the gap is `<= merge_gap_sec`, tentatively add the span.
   - If the gap is `> merge_gap_sec`, finalize the current unit and start a new
     one.
5. When a candidate unit would exceed `max_alignment_unit_sec`, split before
   adding the span when possible.
6. If a single raw span is longer than `max_alignment_unit_sec`, allow provider
   windowing to split it under the existing hard alignment-window budget.
7. Finalize each unit by applying `input_padding_sec` to speech bounds and
   clamping to `[0.0, duration_sec]`.

The final unit sequence must be time-ordered and non-overlapping after input
padding. If padding creates overlap between adjacent units, trim the touching
input bounds at the midpoint between their speech ranges rather than merging
them back together.

## Provider Consumption

The Qwen provider consumes `speech_plan.alignment_units` directly.

If VAD is disabled or failed, use the existing full-duration provider path.

If VAD is `ok` with no alignment units, the pipeline returns an empty document.

If VAD is `ok` with alignment units:

1. For each unit, treat `[input_start, input_end]` as the provider planning
   envelope.
2. If the unit fits within the provider alignment budget, create one provider
   window for that unit.
3. If the unit exceeds the provider budget, plan bounded provider windows inside
   the unit.
4. Materialize each provider window clip from the original prepared WAV.
5. Run ASR and forced alignment on that clip.
6. Offset local token times by `window.context_start`.
7. Use unit speech bounds as display and validation guardrails.
8. Merge token results on the global prepared-audio timeline.

Provider diagnostics should record both the provider window index and the
alignment unit index.

Processing strategy metadata should become:

```text
vad_alignment_unit_bounded_alignment
```

for the VAD path.

## Display Bounds

Model input bounds and display bounds remain separate:

```text
model input: [input_start, input_end]
display:     [speech_start - display_lead_pad, speech_end + display_tail_pad]
```

Recommended provider display pads remain:

```text
display_lead_pad_sec = 0.20
display_tail_pad_sec = 0.35
```

Display bounds are guardrails, not transcript authority. If VAD under-detects a
real word, token timing repair may still preserve text, but final segment
stabilization should avoid displaying a caption far outside the nearest
alignment unit.

## Token Timing Safety Net

Shorter alignment units reduce the probability of bad aligner output, but they
do not eliminate it. The provider still needs token-level validation before
normal segmentation.

Add or extend internal checks for suspicious token timing:

- English short word duration exceeds a conservative threshold.
- Any token duration is grossly incompatible with text length.
- A token spans a known non-speech gap inside the alignment unit.
- A unit contains multiple suspicious long tokens.
- A segment starts far before the nearest speech bound without a clear aligned
  token anchor.

Suspicious tokens must not pass into `_tokens_to_segments()` as normal token
evidence. Depending on context, they should be:

- repaired from adjacent trustworthy tokens,
- converted to estimated timing with low confidence,
- or excluded from token segmentation and preserved through fallback segment
  timing.

The existing `ProjectedToken` timing provenance design remains compatible with
this safety net.

## Metadata

VAD metadata should report raw spans and alignment units:

```json
{
  "vad": {
    "enabled": true,
    "status": "ok",
    "duration_sec": 1234.56,
    "raw_span_count": 42,
    "alignment_unit_count": 9,
    "config": {
      "threshold": 0.25,
      "min_speech_duration_ms": 80,
      "min_silence_duration_ms": 300,
      "speech_pad_ms": 1200,
      "merge_gap_sec": 3.0,
      "input_padding_sec": 0.8,
      "max_alignment_unit_sec": 180.0
    },
    "alignment_units": [
      {
        "index": 0,
        "speech_start": 10498.6,
        "speech_end": 10500.7,
        "input_start": 10497.8,
        "input_end": 10501.5,
        "source_span_count": 1
      }
    ]
  }
}
```

Public subtitle exports remain unchanged. JSON may include this metadata under
`source_media["vad"]`.

## Error Handling

- If VAD dependencies are missing, keep the existing failed-plan behavior and
  continue without a speech plan.
- If VAD returns no spans, return a successful empty transcription document.
- If unit construction drops every span during sanitation, treat it as no speech.
- If a unit has invalid or non-finite bounds, drop only that unit.
- If every provider window for a unit fails, continue other units and preserve
  failure diagnostics.
- If every unit fails provider execution, keep the existing explicit provider
  error behavior.

## Testing Plan

Add VAD unit construction tests:

- Spans separated by more than `3.0s` become separate alignment units.
- Spans separated by `2.5s` merge into one unit.
- Input padding is applied and clamped to media duration.
- Padded adjacent units remain non-overlapping.
- A candidate unit is split before it exceeds `180.0s` when a raw span boundary
  is available.
- `speech_plan_metadata()` emits `alignment_unit_count` and `alignment_units`.
- `super_chunks` no longer appears in the new VAD metadata.

Add provider tests:

- Qwen provider plans windows from `alignment_units`, not `super_chunks`.
- Provider window diagnostics include `alignment_unit_index`.
- Processing strategy is `vad_alignment_unit_bounded_alignment`.
- A short alignment unit around `You'd better stay in line from now on.` cannot
  produce a token-backed subtitle where `You'd` spans 14 seconds.
- If an alignment unit exceeds the provider budget, existing bounded window
  planning still applies inside the unit.
- VAD disabled or failed still uses the full-duration path.
- VAD `ok` with no units returns an empty document.

Add quality tests:

- Long short-word tokens are marked suspicious.
- A token crossing a non-speech gap is marked suspicious.
- Suspicious tokens do not enter normal token segmentation without repair.

Add exporter tests:

- SRT and VTT rendering remain unchanged for canonical segments.
- JSON preserves VAD alignment-unit metadata.
- Public token payloads do not expose internal timing-provenance fields.

## Rollout Plan

1. Update VAD data structures and config names.
2. Implement alignment-unit construction.
3. Update VAD metadata and observability counts.
4. Update pipeline empty-speech behavior to use `alignment_units`.
5. Update Qwen provider planning to consume alignment units.
6. Remove super-chunk provider fallback and related tests.
7. Add token timing safety-net checks for long or speech-gap-crossing tokens.
8. Update documentation and regression tests.

## Acceptance Criteria

- VAD success plans expose `alignment_units`, not `super_chunks`.
- Default nearby-span merging uses `3.0s`.
- `max_alignment_unit_sec = 180.0` is enforced as a hard ceiling only.
- Provider input clips are planned from alignment units.
- The observed `You'd` failure cannot be emitted as a 14-second token-backed
  subtitle.
- Exporter output contracts stay stable.
