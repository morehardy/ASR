# Caption Timing Repair Design

## Purpose

Fix three timing failures in the windowed Qwen provider without changing the
public exporter contract:

- Captions may appear several seconds early when VAD chunk padding or window
  context is treated as display time.
- Captions may remain visible long after speech ends when fallback segments use
  a whole provider core window.
- Short leading words such as `I` or `we` may disappear when unmatched forced
  alignment tokens fall into left overlap and the window keeps only core tokens.

The fix belongs in provider timing and segmentation, not in SRT/VTT rendering.
Exporters should continue to render canonical `Segment` and `Token` timestamps
without provider-specific correction.

## Current Root Cause

The current pipeline keeps ASR text authoritative and borrows timing from forced
alignment:

```text
ASR text
-> build_transcript_tokens()
-> project_timing_onto_transcript()
-> offset tokens by window.context_start
-> split tokens into left/core/right by token.start_time
-> merge preferred window tokens
-> build subtitle segments
-> render exporters
```

Two internal states are currently indistinguishable:

- A token truly aligned at local time `0.0`.
- A token that failed to match any aligner item and still has the initial
  `0.0 -> 0.0` timestamp from transcript token construction.

After provider offsetting, an unmatched token becomes a real-looking timestamp at
`window.context_start`. In VAD mode, `window.context_start` can be the padded
`chunk_start`, several seconds before `speech_start`.

This produces two bad downstream effects:

- The subtitle can start at context or chunk padding instead of speech.
- A leading unmatched token can be classified as left overlap and dropped when
  the same window has core tokens.

Fallback segments have a separate problem: when token-level segmentation fails,
the provider currently creates one segment covering `window.core_start` through
`window.core_end`. A default core window can be about 150 seconds, so fallback
text can remain on screen for the whole window.

## Design Goals

- Preserve ASR text whenever possible.
- Never treat an unmatched token's initial zero timestamp as trustworthy timing.
- Repair token timings before overlap/core ownership decisions.
- Keep model input ranges and subtitle display ranges separate.
- Use VAD speech bounds as display guardrails, not as transcript authority.
- Keep public `Token`, `Segment`, JSON, SRT, VTT, and TXT output contracts stable.
- Add tests that make the formerly accepted zero-timestamp behavior invalid
  before final subtitle generation.

## Non-Goals

- Do not redesign the provider abstraction.
- Do not expose VAD tuning flags through the CLI.
- Do not add provider-specific fields to public JSON token payloads.
- Do not attempt perfect linguistic duration modeling.
- Do not move timestamp correction into exporters.
- Do not require diarization or translation changes.

## Recommended Architecture

Add an internal timing-status layer inside `asr.providers.authority` and consume
it from `QwenMlxProvider`.

The public `Token` model remains unchanged. Provider internals may use a wrapper
that carries timing provenance:

```python
from dataclasses import dataclass
from typing import Literal

TimingSource = Literal["aligner", "estimated", "unresolved"]

@dataclass(frozen=True, slots=True)
class ProjectedToken:
    token: Token
    timing_source: TimingSource
    aligner_index: int | None = None
```

Add a detailed projection function:

```python
def project_timing_onto_transcript_detailed(
    transcript_tokens: list[Token],
    aligner_tokens: list[Token],
) -> list[ProjectedToken]:
    ...
```

Keep `project_timing_onto_transcript()` as a compatibility wrapper returning
plain tokens:

```python
def project_timing_onto_transcript(
    transcript_tokens: list[Token],
    aligner_tokens: list[Token],
) -> list[Token]:
    return [
        projected.token
        for projected in project_timing_onto_transcript_detailed(
            transcript_tokens,
            aligner_tokens,
        )
    ]
```

The Qwen provider should use the detailed path, repair unmatched timings, then
drop back to plain `Token` objects before public document construction.

## Revised Data Flow

```text
ASR text
-> build_transcript_tokens()
-> project_timing_onto_transcript_detailed()
-> repair_unmatched_timings()
-> offset repaired tokens by window.context_start
-> split repaired tokens into left/core/right
-> prefer core plus short same-utterance edge tokens
-> merge window results
-> build token-based subtitle segments
-> stabilize segments with optional VAD display bounds
-> exporters render canonical data
```

The important ordering rule is:

```text
repair unmatched timings before _split_window_tokens()
```

No code should classify left/core/right ownership from an unrepaired
`0.0 -> 0.0` fallback timestamp.

## Unmatched Token Timing Repair

Add an authority-layer helper:

```python
def repair_unmatched_timings(
    projected_tokens: list[ProjectedToken],
    *,
    clip_duration_sec: float | None = None,
    max_estimated_token_duration_sec: float = 0.32,
) -> list[ProjectedToken]:
    ...
```

The helper works in local clip time, before `window.context_start` is added.

### Leading Unmatched Tokens

When one or more unmatched tokens appear before the first aligner-matched token,
place them immediately before that first matched token.

Example:

```text
ASR:     I have
aligner: have = 5.20 -> 5.50
repair: I    = 5.08 -> 5.18
        have = 5.20 -> 5.50
```

Use a small gap such as `0.02s` before the matched token. Estimate durations
conservatively:

- English short words up to three characters: `0.10s`.
- English longer words: `0.18s`, capped by
  `max_estimated_token_duration_sec`.
- Chinese characters: `0.10s`.

If there is not enough room before the first matched token, clamp to `0.0` while
preserving monotonic order and non-negative duration.

### Middle Unmatched Tokens

When unmatched tokens are between two matched tokens, distribute them between
`previous_matched.end_time` and `next_matched.start_time`.

If the gap is large enough, use estimated token durations with small gaps. If the
gap is too small, divide the available space evenly while keeping token order and
non-negative durations.

### Trailing Unmatched Tokens

When unmatched tokens appear after the final matched token, place them after the
last matched token using estimated durations.

If `clip_duration_sec` is known, cap the final estimated token at that duration.
If the available tail is too small, compress estimated durations rather than
letting tokens exceed the clip.

### Fully Unmatched Transcripts

If no token matched the aligner:

- Do not leave tokens at `0.0 -> 0.0` as if that were trustworthy timing.
- Estimate a short sequence starting at local `0.0`.
- Mark tokens as `unresolved` if there is no timing anchor.
- Let provider fallback and display-bound clamping decide whether to keep,
  shorten, or drop the resulting segment.

This keeps ASR text available while making low-confidence timing explicit inside
the provider.

## VAD Display Bounds

The VAD model intentionally stores two boundary concepts:

- `speech_start` and `speech_end`: speech coverage.
- `chunk_start` and `chunk_end`: padded processing envelope.

The provider should keep using `chunk_start/chunk_end` to extract model input,
but subtitle display should be guarded by speech bounds when available.

Extend provider window planning metadata so each VAD-derived `AlignmentWindow`
can be associated with display bounds:

```python
@dataclass(frozen=True)
class WindowDisplayBounds:
    start_time: float
    end_time: float
    super_chunk_index: int
```

The Qwen provider may store this in an internal dictionary keyed by
`window.index`, avoiding any change to public `AlignmentWindow` if that keeps
the patch smaller.

Recommended defaults:

```text
lead_pad_sec = 0.20
tail_pad_sec = 0.35
```

For a super-chunk:

```text
display_start = max(0.0, speech_start - lead_pad_sec)
display_end = min(total_duration_sec, speech_end + tail_pad_sec)
```

Use display bounds during fallback segment creation and final segment
stabilization. Do not use them to clip model input or erase matched token timing
before segment construction.

## Overlap/Core Ownership

After timing repair and global offsetting, split tokens by overlap with the core
range rather than only by `token.start_time`:

```python
def token_overlaps_core(token: Token, window: AlignmentWindow) -> bool:
    token_end = max(token.end_time, token.start_time)
    return token_end > window.core_start and token.start_time < window.core_end
```

Use start-only classification only for tokens with zero duration after repair.

Update `_preferred_tokens_for_window()` so it can retain very short same-utterance
edge tokens:

```text
preferred = protected_prefix_left_overlap + core_tokens + protected_suffix_right_overlap
```

A protected prefix token is eligible when:

- It is immediately before the first core token in ASR order.
- It is estimated or unresolved, not a confidently aligned long context token.
- Its end time is within about `0.35s` of the first core token start.
- Its text is short enough to be a plausible leading function word or character.

This keeps examples such as `I have` and `we have` intact without reintroducing
large duplicate overlap regions.

## Segment Construction

Keep `_tokens_to_segments()` as the normal path, but make it consume only repaired
tokens. Its existing gap and punctuation splitting remain useful.

Add a maximum readable cue duration, defaulting to `8.0s`. If a segment exceeds
that duration, split it at the best available boundary in this order:

1. Existing token gap of at least `1.0s`.
2. Sentence-ending punctuation.
3. Last token whose end time is at or before `segment.start_time + 8.0s`.

Do not split by character count alone when reliable token timings are available.
If no valid split point exists, leave the segment intact and let stabilization
clamp only against total duration, next segment, and VAD display bounds.

## Fallback Segment Creation

Replace whole-core fallback segments with short estimated segments.

For each successful window with text but no usable tokens:

```text
start = display_start if available else window.core_start
duration = estimate_text_duration(text)
duration = min(duration, max_fallback_duration_sec)
end = start + duration
end = min(end, display_end if available else window.core_end)
end = max(start, end)
```

Recommended default:

```text
max_fallback_duration_sec = 6.0
```

Text duration estimation should be conservative:

- English: approximately `0.22s` per word, minimum `1.0s`.
- Chinese: approximately `0.12s` per character, minimum `1.0s`.
- Cap all fallback estimates at `max_fallback_duration_sec`.

Fallback segments should be marked only internally if needed. Public segment
payloads remain unchanged.

## Segment Stabilization

Extend `_stabilize_segment_boundaries()` with optional display bounds:

```python
def _stabilize_segment_boundaries(
    self,
    segments: list[Segment],
    *,
    total_duration_sec: float,
    display_bounds: Sequence[WindowDisplayBounds] | None = None,
    tail_padding_sec: float = 0.12,
    max_segment_duration_sec: float = 8.0,
) -> list[Segment]:
    ...
```

Stabilization should continue to guarantee:

- `start_time >= 0.0`
- `end_time >= start_time`
- `end_time <= total_duration_sec`
- no overlap with the next segment

When display bounds are available, also clamp each segment to the nearest
overlapping speech display bound:

```text
segment.start_time >= bound.start_time
segment.end_time <= bound.end_time
```

If a segment does not overlap any display bound, use the existing total-duration
and next-segment rules rather than dropping the segment. This avoids losing text
when VAD under-detects speech.

## Quality Metrics

The current quality gate already tracks zero or flat timestamp ratio. Timing
repair should reduce zero-duration tokens before quality evaluation.

Update quality evaluation inputs so they use repaired token timings. Do not count
estimated tokens as aligner-perfect, but do allow them to pass if the overall
window is monotonic, non-flat, and text-consistent.

If detailed projected tokens are still available at diagnostic time, provider
metadata may include aggregate internal counts:

```json
{
  "timing_source_counts": {
    "aligner": 42,
    "estimated": 2,
    "unresolved": 0
  }
}
```

This diagnostic is optional and provider-specific. It must not appear in public
token payloads.

## Error Handling

- Invalid aligner tokens with `end_time < start_time` are treated as unmatched.
- Non-finite token times are treated as unmatched.
- Repair must never raise for ordinary bad aligner output; it should produce
  monotonic best-effort tokens.
- If repair cannot create a positive duration, it may create zero-duration
  `unresolved` tokens internally, but provider fallback and segment construction
  must avoid presenting a long or early subtitle from those tokens.
- VAD display bounds must be clamped to `[0.0, total_duration_sec]`.

## Testing Plan

Add regression tests that describe final subtitle behavior, not just internal
projection behavior.

### Authority Tests

Add tests in `tests/test_authority.py`:

- `test_unmatched_leading_short_word_gets_interpolated`
  - Transcript: `I have`.
  - Aligner only returns `have = 5.20 -> 5.50`.
  - Expected: `I` has a start near `5.0`, not `0.0`.
- `test_unmatched_middle_token_is_interpolated`
  - Transcript: `we really have`.
  - Aligner returns `we` and `have`.
  - Expected: `really` is between them.
- `test_unmatched_trailing_token_is_estimated_after_last_match`
  - Transcript has one trailing unmatched token.
  - Expected: trailing token starts after the final matched token and is capped
    by `clip_duration_sec`.
- `test_existing_projection_wrapper_preserves_public_token_return_type`
  - Existing callers of `project_timing_onto_transcript()` still receive
    `list[Token]`.

Update the old test that allowed unmatched `C++ C#` tokens to remain at zero.
The new expected behavior is that those tokens are internally low-confidence but
not left as final zero-timestamp tokens after repair.

### Provider Tests

Add tests in `tests/test_qwen_provider_windowed.py`:

- `test_vad_chunk_padding_does_not_shift_caption_to_chunk_start`
  - VAD chunk has `speech_start=105.0`, `chunk_start=100.0`.
  - Leading unmatched token must not make the final segment start at `100.0`.
- `test_left_overlap_unmatched_prefix_is_not_dropped`
  - Window has `context_start=90.0`, `core_start=105.0`.
  - ASR text is `we have`; aligner only matches `have`.
  - Expected final segment text keeps `we have`.
- `test_fallback_segment_does_not_last_until_window_core_end`
  - A window with text but no tokens has a long core range.
  - Expected fallback segment duration is at most `6.0s`.
- `test_vad_display_bounds_clamp_segment_tail`
  - A token or fallback segment extends beyond `speech_end`.
  - Expected segment end is no later than `speech_end + tail_pad`.
- `test_matched_token_timings_are_not_shifted_by_repair`
  - Fully matched tokens keep their aligner timings after repair and offset.

### Exporter Tests

Add or update exporter tests to confirm:

- SRT/VTT still render canonical segment timestamps.
- JSON token payloads do not include `timing_source`.
- Token-level output still exposes word-level English and character-level
  Chinese tokens.

## Rollout Plan

Implement in small steps:

1. Add `ProjectedToken` and detailed projection while keeping the old public
   authority function intact.
2. Add timing repair and authority tests.
3. Switch Qwen provider internals to detailed projection plus repair.
4. Adjust split/preferred-token behavior for protected edge tokens.
5. Add VAD display bounds to provider window metadata and stabilization.
6. Replace whole-core fallback segments with estimated short segments.
7. Add max segment duration splitting only after repaired token timing is stable.
8. Verify full existing unit suite and targeted new regression tests.

## Acceptance Criteria

- An unmatched leading `I` or `we` before a matched core word is retained in the
  final subtitle text.
- VAD chunk padding no longer causes captions to start at `chunk_start` when
  speech starts later.
- Fallback subtitles no longer last until a 150-second core window ends.
- Matched aligner timings remain stable.
- Public exporters and JSON schema remain unchanged.
- Existing non-VAD windowing behavior continues to pass tests.

## Spec Self-Review

- Placeholder scan: no placeholder requirements remain.
- Internal consistency: timing repair happens before overlap/core splitting
  throughout the design.
- Scope check: the spec is one focused provider timing repair and does not mix in
  CLI, model selection, translation, or diarization work.
- Ambiguity check: defaults for lead pad, tail pad, max fallback duration, and
  max readable segment duration are explicit.
