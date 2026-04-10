# Provider Windowing Design

Date: `2026-04-10`

## Goal

Define how the current default provider (`Qwen3-ASR + Qwen3-ForcedAligner` via `mlx-audio`) should implement internal windowing while keeping one unified full-media output contract.

## References

- [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- [Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)
- [mlx-audio README](https://github.com/Blaizzy/mlx-audio)

## Core Decision

Provider output is always full-media and backend-neutral.

- Internal processing may use windows.
- External output must represent one complete input media result.
- Chunk/window internals must not leak into CLI contract or export schemas.

## Product Constraint

No runtime model-capability routing is required.

- Strategy is fixed in implementation for this provider.
- The current provider uses windowed alignment.
- We do not branch runtime behavior based on dynamic capability checks.
- We only guarantee stable boundary semantics at provider output.

## Chosen Strategy (Current Provider)

Use bounded alignment windows with silence/pause-anchored boundaries.

### Window Budget Constraints

The provider must explicitly enforce a hard alignment envelope:

- `max_alignment_window_sec = 180` (hard upper bound)
- `target_core_window_sec` default `150` (allowed range `120-150`)
- `context_margin_sec` default `15` (allowed range `15-30`)

Hard invariant:

- `context_end - context_start <= max_alignment_window_sec`

This invariant is mandatory and must be checked in planner output.

### Budget Adjustment Rule

When constructing each window:

1. pick a core target in `[120, 150]` seconds
2. add front/back margins
3. if total exceeds max budget, shrink margins first
4. if still over budget, shrink core duration
5. re-validate hard invariant

The planner must never emit a window that violates the hard budget.

### Boundary Anchoring Rule

Each planned split point follows fallback order:

1. silence or low-energy anchor near target
2. pause-like boundary near target
3. fixed-time fallback

This keeps forward progress while reducing mid-token and mid-sentence cuts.

## Internal Pipeline

1. `WindowPlanner`
   - Build full-media windows with validated budget invariants.

2. `WindowTranscriber`
   - Run ASR on each window context span.
   - Produce transcript candidates and metadata.

3. `WindowAligner`
   - Run forced alignment for per-window transcript text.
   - Produce window-local timestamp candidates.

4. `WindowMerger`
   - Merge window outputs into one global stream.
   - Resolve overlap with sequence alignment (not greedy nearest-neighbor dedupe).
   - Map timestamps to original media timeline.
   - Emit one canonical full-media document.

## Authority Split

Transcript authority and timing authority are explicitly separated.

- Transcript authority: ASR core-owned transcript plus overlap consensus.
- Timing authority: aligner output for timestamp assignment only.

Rules:

- Aligner must not be used to invent or overwrite transcript tokens.
- Final token text comes from transcript authority.
- Aligner only attaches timing to existing transcript tokens.
- If timing quality is invalid, degrade timing for that portion instead of rewriting text from aligner output.

## Overlap Merge (Sequence Alignment)

Overlap merge is treated as a local sequence alignment problem.

For adjacent windows in overlap region:

1. build token sequences in global time coordinates
2. run constrained DP alignment (LCS/edit-distance style) with time-aware scoring
3. accept matches only under time-distance and text-similarity constraints
4. resolve duplicates based on alignment path and core ownership
5. preserve global order and continuity

This avoids failure modes in repeated words, disfluencies, Chinese reduplication, function-word drift, and minor transcript rewrites.

## Quality Gates (Window-Level)

In addition to coverage, provider must compute and gate on:

- `monotonic_timestamp_ratio`
- `zero_or_flat_timestamp_ratio`
- `boundary_disagreement_score`
- `core_context_text_divergence`

Suggested default thresholds:

- `monotonic_timestamp_ratio >= 0.98`
- `zero_or_flat_timestamp_ratio <= 0.05`
- `boundary_disagreement_score <= 0.20`
- `core_context_text_divergence <= 0.15`

When a gate fails:

- degrade only the affected window/segment first
- keep global merge running when possible
- mark degradation in provider metadata

## Canonical Unit Policy

Provider-internal canonical unit must not be hardcoded as language-binary word/char rules.

Use one of:

- aligner-native output unit
- tokenizer-normalized token
- grapheme-cluster-like atomic text unit

Exporter layer is responsible for user-facing reconstruction:

- English word view
- Chinese character/block view
- sentence-level subtitle view

This keeps mixed-language, numeric strings, abbreviations, and URL-like text more stable.

## Failure Handling

Failure policy is window-local first, then global:

- if one window fails alignment, continue other windows
- if window timing gates fail, degrade that window timing only
- if all windows lose alignment quality, keep transcript output with degraded timing metadata

Do not convert local window issues into full-file failure when meaningful full-media output is still possible.

## Provider Metadata (Diagnostic)

Provider metadata may include:

- `processing_strategy`: `windowed_bounded_alignment`
- `max_alignment_window_sec`
- `window_count`
- `degraded_window_count`
- `alignment_coverage_ratio`
- `monotonic_timestamp_ratio`
- `zero_or_flat_timestamp_ratio`
- `boundary_disagreement_score`
- `core_context_text_divergence`
- `merge_conflict_count`

Metadata is diagnostic only and must not change external schema shape.

## External Contract Invariants

These stay stable regardless of internal implementation:

- one unified full-media result per input media
- same canonical document shape at provider boundary
- same export semantics (`srt`, `vtt`, `json`, `txt`)
- exporter-controlled user-facing unit policy

## Future Compatibility

If a future provider can align full media without windows:

- it still returns the same canonical full-media output
- no CLI contract change is required
- no exporter contract change is required

Internal strategy remains replaceable. Boundary contract does not.

## Scope

This spec covers provider internal strategy and output boundary semantics only.

Out of scope:

- CLI redesign
- runtime multi-provider routing
- diarization implementation
- translation output

## Self-Review Checklist

- Placeholder scan: no TODO/TBD placeholders remain.
- Consistency scan: hard window budget invariant is explicit and testable.
- Merge scan: overlap merge is sequence alignment with time constraints, not greedy dedupe.
- Authority scan: transcript and timing authority are explicitly separated.
- Quality scan: boundary quality gates include the required four metrics.
- Unit scan: canonical provider unit is not fixed to language-binary word/char internals.
