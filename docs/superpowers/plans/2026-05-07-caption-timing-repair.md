# Caption Timing Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair provider caption timing so unmatched aligner tokens do not create early captions, dropped leading words, or long fallback subtitles.

**Architecture:** Add internal timing provenance in `authority.py`, repair unmatched timings before provider ownership decisions, and keep fully unresolved windows on fallback segment timing. Carry display bounds and timing source counts through `WindowRun`, use one shared core-overlap helper, and keep exporters rendering the canonical public model unchanged.

**Tech Stack:** Python 3.14, dataclasses, `unittest`, existing `uv run python -m unittest` test workflow.

---

## File Structure

- Modify `src/asr/providers/authority.py`
  - Owns transcript token construction, detailed projection, timing provenance, and local-time repair.
  - Exposes `ProjectedToken`, `TimingSource`, `project_timing_onto_transcript_detailed()`, and `repair_unmatched_timings()`.
  - Keeps `project_timing_onto_transcript()` returning `list[Token]` for compatibility.

- Modify `src/asr/providers/window_merge.py`
  - Owns adjacent-window token merge and core ownership helper.
  - Adds `token_overlaps_core()` and updates `_in_core()` to use it.

- Modify `src/asr/providers/qwen_mlx.py`
  - Owns Qwen provider execution, window provenance, display bounds, merge routing, fallback segments, and segment stabilization.
  - Extends `WindowRun` and adds `WindowDisplayBounds`.
  - Switches provider projection to detailed timing + repair before offset/split.

- Modify `src/asr/providers/quality.py`
  - Keeps existing public quality metrics.
  - Adds timing-source thresholds to `QualityThresholds`, timing-source ratios to `QualityResult`, and timing counts to `evaluate_quality()`.
  - The provider uses this to prevent low-confidence windows from joining passing merge blocks.

- Modify `tests/test_authority.py`
  - Covers detailed projection and timing repair in local clip time.

- Modify `tests/test_qwen_provider_windowed.py`
  - Covers VAD padding, leading prefix preservation, fallback routing, display bounds, core ownership, and provider diagnostics.

- Modify `tests/test_quality.py`
  - Covers unresolved and high-estimated timing source quality behavior.

- Modify `tests/test_exporters.py`
  - Confirms public JSON tokens do not expose internal timing metadata.

Use `uv run python -m unittest ...` for verification. `uv run pytest ...` is not available in this workspace.

---

### Task 1: Add Detailed Projection Types

**Files:**
- Modify: `src/asr/providers/authority.py`
- Test: `tests/test_authority.py`

- [ ] **Step 1: Write failing detailed projection tests**

Add these imports to `tests/test_authority.py`:

```python
from asr.providers.authority import (
    ProjectedToken,
    build_transcript_tokens,
    project_timing_onto_transcript,
    project_timing_onto_transcript_detailed,
)
```

Add this test to `AuthorityTest`:

```python
    def test_detailed_projection_marks_unmatched_tokens_unresolved(self) -> None:
        transcript_tokens = build_transcript_tokens("I have", language="en")
        aligner_tokens = [
            Token("have", 5.20, 5.50, unit="token"),
        ]

        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            aligner_tokens,
        )

        self.assertEqual([item.token.text for item in projected], ["I", "have"])
        self.assertEqual([item.timing_source for item in projected], ["unresolved", "aligner"])
        self.assertIsNone(projected[0].aligner_index)
        self.assertEqual(projected[1].aligner_index, 0)
        self.assertEqual((projected[1].token.start_time, projected[1].token.end_time), (5.20, 5.50))
```

Add this compatibility test:

```python
    def test_existing_projection_wrapper_preserves_public_token_return_type(self) -> None:
        transcript_tokens = build_transcript_tokens("hello", language="en")
        aligner_tokens = [Token("hello", 1.0, 1.2, unit="token")]

        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertIsInstance(projected[0], Token)
        self.assertEqual([(token.text, token.start_time, token.end_time) for token in projected], [("hello", 1.0, 1.2)])
```

- [ ] **Step 2: Run detailed projection tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_authority
```

Expected: fail with `ImportError` for `ProjectedToken` or `project_timing_onto_transcript_detailed`.

- [ ] **Step 3: Implement detailed projection types**

Update the top of `src/asr/providers/authority.py`:

```python
from dataclasses import dataclass
import math
import re
from difflib import SequenceMatcher
from typing import List, Literal, Optional
```

Add these definitions after the `Token` import:

```python
TimingSource = Literal["aligner", "estimated", "unresolved"]


@dataclass(frozen=True, slots=True)
class ProjectedToken:
    token: Token
    timing_source: TimingSource
    aligner_index: int | None = None
```

Replace `project_timing_onto_transcript()` with this wrapper and add the detailed function:

```python
def project_timing_onto_transcript(
    transcript_tokens: List[Token], aligner_tokens: List[Token]
) -> List[Token]:
    return [
        projected.token
        for projected in project_timing_onto_transcript_detailed(
            transcript_tokens,
            aligner_tokens,
        )
    ]


def project_timing_onto_transcript_detailed(
    transcript_tokens: List[Token], aligner_tokens: List[Token]
) -> List[ProjectedToken]:
    projected: List[ProjectedToken] = []
    aligner_index = 0

    for transcript_token in transcript_tokens:
        match_index = _find_forward_match(transcript_token.text, aligner_tokens, aligner_index)
        if match_index is None:
            projected.append(ProjectedToken(_clone_token(transcript_token), "unresolved", None))
            continue

        aligner_token = aligner_tokens[match_index]
        aligner_index = match_index + 1

        if not _valid_token_timing(aligner_token):
            projected.append(ProjectedToken(_clone_token(transcript_token), "unresolved", None))
            continue

        projected.append(
            ProjectedToken(
                Token(
                    text=transcript_token.text,
                    start_time=aligner_token.start_time,
                    end_time=aligner_token.end_time,
                    unit=transcript_token.unit,
                    language=transcript_token.language,
                ),
                "aligner",
                match_index,
            )
        )

    return projected
```

Add this helper near `_clone_token()`:

```python
def _valid_token_timing(token: Token) -> bool:
    return (
        math.isfinite(token.start_time)
        and math.isfinite(token.end_time)
        and token.end_time >= token.start_time
    )
```

- [ ] **Step 4: Run authority tests**

Run:

```bash
uv run python -m unittest tests.test_authority
```

Expected: all authority tests pass.

- [ ] **Step 5: Commit detailed projection**

Run:

```bash
git add src/asr/providers/authority.py tests/test_authority.py
git commit -m "feat: add detailed timing projection"
```

---

### Task 2: Add Unmatched Timing Repair

**Files:**
- Modify: `src/asr/providers/authority.py`
- Test: `tests/test_authority.py`

- [ ] **Step 1: Write failing repair tests**

Add `repair_unmatched_timings` to the `tests/test_authority.py` import list:

```python
from asr.providers.authority import (
    ProjectedToken,
    build_transcript_tokens,
    project_timing_onto_transcript,
    project_timing_onto_transcript_detailed,
    repair_unmatched_timings,
)
```

Add these tests to `AuthorityTest`:

```python
    def test_unmatched_leading_short_word_gets_interpolated(self) -> None:
        transcript_tokens = build_transcript_tokens("I have", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [Token("have", 5.20, 5.50, unit="token")],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=10.0)

        self.assertEqual([item.token.text for item in repaired], ["I", "have"])
        self.assertEqual([item.timing_source for item in repaired], ["estimated", "aligner"])
        self.assertGreaterEqual(repaired[0].token.start_time, 5.0)
        self.assertLess(repaired[0].token.end_time, repaired[1].token.start_time)
        self.assertEqual((repaired[1].token.start_time, repaired[1].token.end_time), (5.20, 5.50))

    def test_unmatched_middle_token_is_interpolated(self) -> None:
        transcript_tokens = build_transcript_tokens("we really have", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("we", 2.00, 2.12, unit="token"),
                Token("have", 2.80, 3.10, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=5.0)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "estimated", "aligner"])
        self.assertGreaterEqual(repaired[1].token.start_time, repaired[0].token.end_time)
        self.assertLessEqual(repaired[1].token.end_time, repaired[2].token.start_time)

    def test_unmatched_trailing_token_is_estimated_after_last_match(self) -> None:
        transcript_tokens = build_transcript_tokens("hello there", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [Token("hello", 1.00, 1.30, unit="token")],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=1.50)

        self.assertEqual([item.timing_source for item in repaired], ["aligner", "estimated"])
        self.assertGreaterEqual(repaired[1].token.start_time, repaired[0].token.end_time)
        self.assertLessEqual(repaired[1].token.end_time, 1.50)

    def test_fully_unmatched_tokens_remain_unresolved_for_provider_fallback(self) -> None:
        transcript_tokens = build_transcript_tokens("C++ C#", language="en")
        projected = project_timing_onto_transcript_detailed(
            transcript_tokens,
            [
                Token("C", 1.00, 1.10, unit="token"),
                Token("C", 1.10, 1.20, unit="token"),
            ],
        )

        repaired = repair_unmatched_timings(projected, clip_duration_sec=2.0)

        self.assertEqual([item.token.text for item in repaired], ["C++", "C#"])
        self.assertEqual([item.timing_source for item in repaired], ["unresolved", "unresolved"])
```

- [ ] **Step 2: Run repair tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_authority
```

Expected: fail with `ImportError` for `repair_unmatched_timings`.

- [ ] **Step 3: Implement timing repair**

Add this public helper to `src/asr/providers/authority.py` after detailed projection:

```python
def repair_unmatched_timings(
    projected_tokens: List[ProjectedToken],
    *,
    clip_duration_sec: float | None = None,
    max_estimated_token_duration_sec: float = 0.32,
) -> List[ProjectedToken]:
    if not projected_tokens:
        return []

    anchor_indexes = [
        index
        for index, projected in enumerate(projected_tokens)
        if projected.timing_source == "aligner" and _valid_token_timing(projected.token)
    ]
    if not anchor_indexes:
        return list(projected_tokens)

    repaired = list(projected_tokens)
    first_anchor = anchor_indexes[0]
    if first_anchor > 0:
        _repair_leading_tokens(
            repaired,
            start_index=0,
            end_index=first_anchor,
            next_anchor=repaired[first_anchor].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )

    for left_anchor, right_anchor in zip(anchor_indexes, anchor_indexes[1:]):
        if right_anchor - left_anchor > 1:
            _repair_middle_tokens(
                repaired,
                start_index=left_anchor + 1,
                end_index=right_anchor,
                previous_anchor=repaired[left_anchor].token,
                next_anchor=repaired[right_anchor].token,
                max_estimated_token_duration_sec=max_estimated_token_duration_sec,
            )

    last_anchor = anchor_indexes[-1]
    if last_anchor < len(repaired) - 1:
        _repair_trailing_tokens(
            repaired,
            start_index=last_anchor + 1,
            previous_anchor=repaired[last_anchor].token,
            clip_duration_sec=clip_duration_sec,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )

    return repaired
```

Add these private helpers below it:

```python
_ESTIMATED_TOKEN_GAP_SEC = 0.02


def _repair_leading_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    end_index: int,
    next_anchor: Token,
    max_estimated_token_duration_sec: float,
) -> None:
    cursor = next_anchor.start_time
    for index in range(end_index - 1, start_index - 1, -1):
        duration = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        end_time = max(0.0, cursor - _ESTIMATED_TOKEN_GAP_SEC)
        start_time = max(0.0, end_time - duration)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = start_time


def _repair_middle_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    end_index: int,
    previous_anchor: Token,
    next_anchor: Token,
    max_estimated_token_duration_sec: float,
) -> None:
    count = end_index - start_index
    available_start = previous_anchor.end_time
    available_end = next_anchor.start_time
    available = max(0.0, available_end - available_start)
    if count <= 0:
        return

    slot = available / count if available > 0.0 else 0.0
    cursor = available_start
    for index in range(start_index, end_index):
        requested = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        duration = min(requested, max(0.0, slot - _ESTIMATED_TOKEN_GAP_SEC))
        start_time = cursor
        end_time = min(available_end, start_time + duration)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = min(available_end, start_time + slot)


def _repair_trailing_tokens(
    repaired: List[ProjectedToken],
    *,
    start_index: int,
    previous_anchor: Token,
    clip_duration_sec: float | None,
    max_estimated_token_duration_sec: float,
) -> None:
    cursor = previous_anchor.end_time
    clip_end = clip_duration_sec if clip_duration_sec is not None and math.isfinite(clip_duration_sec) else None
    for index in range(start_index, len(repaired)):
        duration = _estimated_token_duration(
            repaired[index].token,
            max_estimated_token_duration_sec=max_estimated_token_duration_sec,
        )
        start_time = cursor + _ESTIMATED_TOKEN_GAP_SEC
        end_time = start_time + duration
        if clip_end is not None:
            start_time = min(start_time, clip_end)
            end_time = min(end_time, clip_end)
        repaired[index] = _with_estimated_timing(repaired[index], start_time, end_time)
        cursor = end_time


def _with_estimated_timing(projected: ProjectedToken, start_time: float, end_time: float) -> ProjectedToken:
    token = projected.token
    return ProjectedToken(
        Token(
            text=token.text,
            start_time=max(0.0, start_time),
            end_time=max(max(0.0, start_time), end_time),
            unit=token.unit,
            language=token.language,
        ),
        "estimated",
        projected.aligner_index,
    )


def _estimated_token_duration(
    token: Token,
    *,
    max_estimated_token_duration_sec: float,
) -> float:
    text = token.text.strip()
    if token.unit == "char" or _contains_cjk(text):
        return min(0.10, max_estimated_token_duration_sec)
    if len(text) <= 3:
        return min(0.10, max_estimated_token_duration_sec)
    return min(0.18, max_estimated_token_duration_sec)


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)
```

- [ ] **Step 4: Run authority tests**

Run:

```bash
uv run python -m unittest tests.test_authority
```

Expected: all authority tests pass.

- [ ] **Step 5: Commit timing repair**

Run:

```bash
git add src/asr/providers/authority.py tests/test_authority.py
git commit -m "feat: repair unmatched token timings"
```

---

### Task 3: Add Timing-Source-Aware Quality Gate

**Files:**
- Modify: `src/asr/providers/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write failing quality tests**

Add these tests to `QualityGateTest` in `tests/test_quality.py`:

```python
    def test_unresolved_tokens_cannot_pass_quality(self) -> None:
        tokens = [
            Token(text="hello", start_time=1.0, end_time=1.2, unit="token"),
            Token(text="world", start_time=1.3, end_time=1.5, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="hello world",
            context_text="hello world",
            thresholds=QualityThresholds(),
            timing_source_counts={"aligner": 1, "estimated": 0, "unresolved": 1},
            has_timing_anchor=True,
        )

        self.assertFalse(result.passed)
        self.assertAlmostEqual(result.unresolved_token_ratio, 0.5)

    def test_high_estimated_ratio_downgrades_merge_confidence(self) -> None:
        tokens = [
            Token(text="I", start_time=1.0, end_time=1.1, unit="token"),
            Token(text="have", start_time=1.2, end_time=1.5, unit="token"),
            Token(text="one", start_time=1.6, end_time=1.8, unit="token"),
        ]

        result = evaluate_quality(
            tokens=tokens,
            left_overlap_tokens=tokens,
            right_overlap_tokens=tokens,
            core_text="I have one",
            context_text="I have one",
            thresholds=QualityThresholds(),
            timing_source_counts={"aligner": 2, "estimated": 1, "unresolved": 0},
            has_timing_anchor=True,
        )

        self.assertFalse(result.passed)
        self.assertGreater(result.estimated_token_ratio, 0.30)
```

- [ ] **Step 2: Run quality tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_quality
```

Expected: fail because `evaluate_quality()` does not accept timing-source arguments.

- [ ] **Step 3: Implement timing-source quality fields**

Update `QualityThresholds` and `QualityResult` in `src/asr/providers/quality.py`:

```python
@dataclass(frozen=True, slots=True)
class QualityThresholds:
    monotonic_timestamp_ratio_min: float = 0.98
    zero_or_flat_timestamp_ratio_max: float = 0.05
    boundary_disagreement_score_max: float = 0.20
    core_context_text_divergence_max: float = 0.15
    estimated_token_ratio_max: float = 0.30
    unresolved_token_ratio_max: float = 0.0


@dataclass(frozen=True, slots=True)
class QualityResult:
    passed: bool
    monotonic_timestamp_ratio: float
    zero_or_flat_timestamp_ratio: float
    boundary_disagreement_score: float
    core_context_text_divergence: float
    estimated_token_ratio: float = 0.0
    unresolved_token_ratio: float = 0.0
```

Update `evaluate_quality()` signature and body:

```python
def evaluate_quality(
    tokens: Sequence[Token],
    left_overlap_tokens: Sequence[Token],
    right_overlap_tokens: Sequence[Token],
    core_text: str,
    context_text: str,
    thresholds: QualityThresholds,
    timing_source_counts: dict[str, int] | None = None,
    has_timing_anchor: bool = True,
) -> QualityResult:
    if not tokens:
        return QualityResult(False, 0.0, 1.0, 1.0, 1.0)

    timing_source_counts = timing_source_counts or {}
    timing_total = max(1, sum(timing_source_counts.values()))
    estimated_token_ratio = timing_source_counts.get("estimated", 0) / timing_total
    unresolved_token_ratio = timing_source_counts.get("unresolved", 0) / timing_total

    monotonic_timestamp_ratio = _monotonic_timestamp_ratio(tokens)
    zero_or_flat_timestamp_ratio = _zero_or_flat_timestamp_ratio(tokens)
    boundary_disagreement_score = 1.0 - SequenceMatcher(
        None,
        _joined_token_text(left_overlap_tokens),
        _joined_token_text(right_overlap_tokens),
    ).ratio()
    core_context_text_divergence = 1.0 - SequenceMatcher(
        None,
        core_text,
        context_text,
    ).ratio()

    passed = (
        has_timing_anchor
        and monotonic_timestamp_ratio >= thresholds.monotonic_timestamp_ratio_min
        and zero_or_flat_timestamp_ratio <= thresholds.zero_or_flat_timestamp_ratio_max
        and boundary_disagreement_score <= thresholds.boundary_disagreement_score_max
        and core_context_text_divergence <= thresholds.core_context_text_divergence_max
        and estimated_token_ratio <= thresholds.estimated_token_ratio_max
        and unresolved_token_ratio <= thresholds.unresolved_token_ratio_max
    )

    return QualityResult(
        passed=passed,
        monotonic_timestamp_ratio=monotonic_timestamp_ratio,
        zero_or_flat_timestamp_ratio=zero_or_flat_timestamp_ratio,
        boundary_disagreement_score=boundary_disagreement_score,
        core_context_text_divergence=core_context_text_divergence,
        estimated_token_ratio=estimated_token_ratio,
        unresolved_token_ratio=unresolved_token_ratio,
    )
```

Keep positional compatibility: existing callers still pass the first six arguments.

- [ ] **Step 4: Run quality tests**

Run:

```bash
uv run python -m unittest tests.test_quality
```

Expected: all quality tests pass.

- [ ] **Step 5: Commit quality gate**

Run:

```bash
git add src/asr/providers/quality.py tests/test_quality.py
git commit -m "feat: account for timing sources in quality"
```

---

### Task 4: Add Shared Core Ownership Helper

**Files:**
- Modify: `src/asr/providers/window_merge.py`
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_window_merge.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing window merge helper test**

Add this test to `WindowMergeTest` in `tests/test_window_merge.py`:

```python
    def test_in_core_uses_token_overlap_not_only_start_time(self) -> None:
        left_tokens = [
            Token(text="we", start_time=0.70, end_time=0.82, unit="word"),
        ]
        right_tokens = [
            Token(text="we", start_time=0.70, end_time=0.82, unit="word"),
        ]
        left_span = WindowSpan(
            core_start=0.0,
            core_end=0.8,
            context_start=0.0,
            context_end=1.0,
        )
        right_span = WindowSpan(
            core_start=0.8,
            core_end=1.4,
            context_start=0.6,
            context_end=1.4,
        )

        merged = merge_adjacent_windows(
            left_tokens,
            right_tokens,
            left_span,
            right_span,
            max_time_delta=0.20,
        )

        self.assertEqual([(token.text, token.start_time, token.end_time) for token in merged], [("we", 0.70, 0.82)])
```

Add this provider ownership test to `QwenProviderWindowedTest`:

```python
    def test_owned_tokens_for_block_uses_token_overlap_not_only_start_time(self) -> None:
        provider = QwenMlxProvider()
        window_runs = [
            WindowRun(
                window=AlignmentWindow(0, 105.0, 120.0, 100.0, 125.0),
                text="we have",
            )
        ]
        tokens = [
            Token("we", 104.95, 105.05, unit="word"),
            Token("have", 105.20, 105.50, unit="word"),
        ]

        owned = provider._owned_tokens_for_block(tokens, window_runs)

        self.assertEqual([token.text for token in owned], ["we", "have"])
```

- [ ] **Step 2: Run ownership tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_window_merge tests.test_qwen_provider_windowed
```

Expected: provider ownership test fails because `_owned_tokens_for_block()` filters by `start_time`.

- [ ] **Step 3: Implement shared overlap helper**

Update `src/asr/providers/window_merge.py`:

```python
def token_overlaps_core(
    token: Token,
    *,
    core_start: float,
    core_end: float,
) -> bool:
    token_end = max(token.end_time, token.start_time)
    if token_end == token.start_time:
        return core_start <= token.start_time < core_end
    return token_end > core_start and token.start_time < core_end


def _in_core(token: Token, span: WindowSpan) -> bool:
    return token_overlaps_core(
        token,
        core_start=span.core_start,
        core_end=span.core_end,
    )
```

Update the import in `src/asr/providers/qwen_mlx.py`:

```python
from asr.providers.window_merge import WindowSpan, merge_adjacent_windows, token_overlaps_core
```

Update `_split_window_tokens()`:

```python
        for token in tokens:
            if token_overlaps_core(
                token,
                core_start=window.core_start,
                core_end=window.core_end,
            ):
                core_tokens.append(token)
            elif token.start_time < window.core_start:
                left_overlap.append(token)
            else:
                right_overlap.append(token)
```

Update `_owned_tokens_for_block()`:

```python
        owned_tokens = [
            token
            for token in tokens
            if any(
                token_overlaps_core(
                    token,
                    core_start=window_run.window.core_start,
                    core_end=window_run.window.core_end,
                )
                for window_run in window_runs
            )
        ]
```

- [ ] **Step 4: Run ownership tests**

Run:

```bash
uv run python -m unittest tests.test_window_merge tests.test_qwen_provider_windowed
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit core ownership helper**

Run:

```bash
git add src/asr/providers/window_merge.py src/asr/providers/qwen_mlx.py tests/test_window_merge.py tests/test_qwen_provider_windowed.py
git commit -m "fix: use overlap for core token ownership"
```

---

### Task 5: Carry WindowRun Provenance and Display Bounds

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing `WindowRun` provenance tests**

Update imports in `tests/test_qwen_provider_windowed.py`:

```python
from asr.providers.qwen_mlx import QwenMlxProvider, WindowDisplayBounds, WindowRun
```

Add this test to `QwenProviderWindowedTest`:

```python
    def test_window_run_carries_display_bounds_without_provider_lookup_state(self) -> None:
        bounds = WindowDisplayBounds(start_time=104.8, end_time=120.35, super_chunk_index=0)
        run = WindowRun(
            window=AlignmentWindow(0, 100.0, 130.0, 100.0, 130.0, super_chunk_index=0),
            text="hello",
            display_bounds=bounds,
        )

        self.assertEqual(run.display_bounds, bounds)
        self.assertFalse(hasattr(QwenMlxProvider(), "_display_bounds_by_window_index"))
        self.assertFalse(hasattr(QwenMlxProvider(), "_window_display_bounds"))

    def test_window_run_defaults_to_no_timing_anchor(self) -> None:
        run = WindowRun(window=AlignmentWindow(0, 0.0, 1.0, 0.0, 1.0), text="hello")

        self.assertFalse(run.has_timing_anchor)
        self.assertEqual(run.timing_source_counts, {})
        self.assertEqual(run.projected_tokens, [])
```

- [ ] **Step 2: Run provider tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: fail with `ImportError` for `WindowDisplayBounds` or missing `WindowRun` fields.

- [ ] **Step 3: Extend provider dataclasses**

Update imports in `src/asr/providers/qwen_mlx.py`:

```python
from asr.providers.authority import (
    ProjectedToken,
    build_transcript_tokens,
    project_timing_onto_transcript_detailed,
    repair_unmatched_timings,
)
```

Add `WindowDisplayBounds` before `WindowRun`:

```python
@dataclass(frozen=True, slots=True)
class WindowDisplayBounds:
    start_time: float
    end_time: float
    super_chunk_index: int
```

Extend `WindowRun`:

```python
@dataclass(slots=True)
class WindowRun:
    window: AlignmentWindow
    text: str = ""
    language: Optional[str] = None
    tokens: List[Token] = field(default_factory=list)
    core_tokens: List[Token] = field(default_factory=list)
    left_overlap_tokens: List[Token] = field(default_factory=list)
    right_overlap_tokens: List[Token] = field(default_factory=list)
    core_text: str = ""
    quality: Optional[QualityResult] = None
    error: Optional[str] = None
    projected_tokens: List[ProjectedToken] = field(default_factory=list)
    timing_source_counts: dict[str, int] = field(default_factory=dict)
    has_timing_anchor: bool = False
    display_bounds: WindowDisplayBounds | None = None
```

- [ ] **Step 4: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: all provider tests pass after import updates.

- [ ] **Step 5: Commit `WindowRun` provenance fields**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "feat: carry window timing provenance"
```

---

### Task 6: Wire Detailed Projection Into Provider

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing leading-prefix provider test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_left_overlap_unmatched_prefix_is_not_dropped(self) -> None:
        provider, asr_model, align_model = self._build_provider_with_models(
            asr_responses=[FakeChunk("we have", language="en")],
            align_responses=[
                [FakeChunk("have", start_time=15.20, end_time=15.50)],
            ],
            quality_thresholds=QualityThresholds(estimated_token_ratio_max=1.0),
        )
        provider._probe_duration_sec = lambda _: 130.0
        provider.window_config = provider.window_config.__class__(
            max_alignment_window_sec=30.0,
            target_core_window_sec=25.0,
            min_core_window_sec=10.0,
            context_margin_sec=15.0,
            max_context_margin_sec=15.0,
            anchor_search_radius_sec=2.0,
        )

        doc = provider.transcribe(Path("demo.wav"))

        self.assertEqual([segment.text for segment in doc.segments], ["we have"])
        token_text = [token.text for segment in doc.segments for token in segment.tokens]
        self.assertEqual(token_text, ["we", "have"])
```

- [ ] **Step 2: Run provider test to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_left_overlap_unmatched_prefix_is_not_dropped
```

Expected: fail because current provider drops `we` when it falls outside core and core tokens exist.

- [ ] **Step 3: Use detailed projection and repair in `_transcribe_window()`**

Replace the projection block in `src/asr/providers/qwen_mlx.py`:

```python
        transcript_tokens = self._build_authoritative_tokens(text, language)
        projected_tokens = project_timing_onto_transcript(
            transcript_tokens,
            aligner_tokens,
        )
        global_tokens = self._offset_tokens(projected_tokens, window.context_start)
```

with:

```python
        transcript_tokens = self._build_authoritative_tokens(text, language)
        projected_tokens = repair_unmatched_timings(
            project_timing_onto_transcript_detailed(
                transcript_tokens,
                aligner_tokens,
            ),
            clip_duration_sec=max(0.0, window.context_end - window.context_start),
        )
        timing_source_counts = self._timing_source_counts(projected_tokens)
        has_timing_anchor = timing_source_counts.get("aligner", 0) > 0

        usable_projected_tokens = (
            self._usable_projected_tokens(projected_tokens)
            if has_timing_anchor
            else []
        )
        global_tokens = self._offset_tokens(
            [projected.token for projected in usable_projected_tokens],
            window.context_start,
        )
```

Return the new fields:

```python
            projected_tokens=projected_tokens,
            timing_source_counts=timing_source_counts,
            has_timing_anchor=has_timing_anchor,
```

Add these helpers near `_offset_tokens()`:

```python
    def _timing_source_counts(self, projected_tokens: Iterable[ProjectedToken]) -> dict[str, int]:
        counts = {"aligner": 0, "estimated": 0, "unresolved": 0}
        for projected in projected_tokens:
            counts[projected.timing_source] = counts.get(projected.timing_source, 0) + 1
        return counts

    def _usable_projected_tokens(self, projected_tokens: Iterable[ProjectedToken]) -> List[ProjectedToken]:
        return [
            projected
            for projected in projected_tokens
            if projected.timing_source in {"aligner", "estimated"}
        ]
```

- [ ] **Step 4: Pass timing-source counts into quality**

Update `_evaluate_window_qualities()` call:

```python
            window_run.quality = evaluate_quality(
                tokens=window_run.tokens,
                left_overlap_tokens=left_overlap_tokens,
                right_overlap_tokens=right_overlap_tokens,
                core_text=window_run.core_text or window_run.text,
                context_text=window_run.text,
                thresholds=self.quality_thresholds,
                timing_source_counts=window_run.timing_source_counts,
                has_timing_anchor=window_run.has_timing_anchor,
            )
```

- [ ] **Step 5: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed tests.test_quality tests.test_authority
```

Expected: selected tests pass.

- [ ] **Step 6: Commit provider repair wiring**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "fix: repair provider token timings before split"
```

---

### Task 7: Protect Same-Utterance Edge Tokens

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing protected prefix unit test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_preferred_tokens_include_short_estimated_prefix_before_core(self) -> None:
        provider = QwenMlxProvider()
        window = AlignmentWindow(0, 105.0, 120.0, 100.0, 125.0)
        prefix = Token("I", 104.98, 105.08, unit="word")
        core = Token("have", 105.20, 105.50, unit="word")
        run = WindowRun(
            window=window,
            text="I have",
            tokens=[prefix, core],
            left_overlap_tokens=[prefix],
            core_tokens=[core],
            timing_source_counts={"aligner": 1, "estimated": 1, "unresolved": 0},
            has_timing_anchor=True,
        )

        preferred = provider._preferred_tokens_for_window(run)

        self.assertEqual([token.text for token in preferred], ["I", "have"])
```

- [ ] **Step 2: Run protected prefix test to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_preferred_tokens_include_short_estimated_prefix_before_core
```

Expected: fail because `_preferred_tokens_for_window()` returns only `core_tokens`.

- [ ] **Step 3: Implement protected edge token selection**

Replace `_preferred_tokens_for_window()` in `src/asr/providers/qwen_mlx.py`:

```python
    def _preferred_tokens_for_window(self, window_run: WindowRun) -> List[Token]:
        if not window_run.core_tokens:
            return list(window_run.tokens)

        protected_prefix = self._protected_prefix_tokens(window_run)
        protected_suffix = self._protected_suffix_tokens(window_run)
        return protected_prefix + list(window_run.core_tokens) + protected_suffix
```

Add these helpers:

```python
    def _protected_prefix_tokens(self, window_run: WindowRun) -> List[Token]:
        if not window_run.left_overlap_tokens or not window_run.core_tokens:
            return []
        first_core = window_run.core_tokens[0]
        prefix: List[Token] = []
        for token in reversed(window_run.left_overlap_tokens):
            if first_core.start_time - token.end_time > 0.35:
                break
            if not self._short_edge_token(token):
                break
            prefix.append(token)
        prefix.reverse()
        return prefix

    def _protected_suffix_tokens(self, window_run: WindowRun) -> List[Token]:
        if not window_run.right_overlap_tokens or not window_run.core_tokens:
            return []
        last_core = window_run.core_tokens[-1]
        suffix: List[Token] = []
        for token in window_run.right_overlap_tokens:
            if token.start_time - last_core.end_time > 0.35:
                break
            if not self._short_edge_token(token):
                break
            suffix.append(token)
        return suffix

    def _short_edge_token(self, token: Token) -> bool:
        text = token.text.strip()
        if token.unit == "char":
            return len(text) == 1
        return 0 < len(text) <= 3
```

- [ ] **Step 4: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: provider tests pass.

- [ ] **Step 5: Commit protected edge tokens**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "fix: keep short repaired edge tokens"
```

---

### Task 8: Attach VAD Display Bounds to WindowRun

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing VAD display-bound test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_vad_window_run_carries_speech_display_bounds(self) -> None:
        provider, _, _ = self._build_provider_with_models(
            asr_responses=[FakeChunk("hello", language="en")],
            align_responses=[[FakeChunk("hello", start_time=5.0, end_time=5.3)]],
        )
        plan = self._speech_plan(
            [SuperChunk(0, 105.0, 120.0, 100.0, 130.0, 1)],
            duration_sec=200.0,
        )

        doc = provider.transcribe(Path("demo.wav"), speech_plan=plan)
        diagnostic = doc.source_media["provider_metadata"]["window_diagnostics"][0]

        self.assertEqual(diagnostic["display_start"], 104.8)
        self.assertEqual(diagnostic["display_end"], 120.35)
```

- [ ] **Step 2: Run display-bound test to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_vad_window_run_carries_speech_display_bounds
```

Expected: fail because diagnostics do not include display bounds.

- [ ] **Step 3: Create display bounds explicitly while executing windows**

Add constants to `QwenMlxProvider`:

```python
    vad_display_lead_pad_sec: float = 0.20
    vad_display_tail_pad_sec: float = 0.35
```

In the `transcribe()` window loop, compute bounds from the current window and
speech plan and pass them into `_execute_window()`:

```python
            for index, window in enumerate(windows, start=1):
                window_runs.append(
                    self._execute_window(
                        audio_path,
                        window,
                        window_index=index,
                        window_count=len(windows),
                        display_bounds=self._display_bounds_for_window(
                            window,
                            speech_plan=speech_plan,
                            total_duration_sec=total_duration_sec,
                        ),
                    )
                )
```

Update `_execute_window()` signature:

```python
    def _execute_window(
        self,
        audio_path: Path,
        window: AlignmentWindow,
        *,
        window_index: int,
        window_count: int,
        display_bounds: WindowDisplayBounds | None = None,
    ) -> WindowRun:
```

Pass bounds into `_transcribe_window()`:

```python
                return self._transcribe_window(
                    audio_path,
                    window,
                    display_bounds=display_bounds,
                )
```

Update `_transcribe_window()` signature:

```python
    def _transcribe_window(
        self,
        audio_path: Path,
        window: AlignmentWindow,
        *,
        display_bounds: WindowDisplayBounds | None = None,
    ) -> WindowRun:
```

Set the field in both success and failure returns:

```python
            display_bounds=display_bounds,
```

Add the display-bound lookup helper near `_super_chunk_count()`:

```python
    def _display_bounds_for_window(
        self,
        window: AlignmentWindow,
        *,
        speech_plan: SpeechPlan | None,
        total_duration_sec: float,
    ) -> WindowDisplayBounds | None:
        if not self._uses_speech_plan(speech_plan):
            return None
        if window.super_chunk_index is None:
            return None
        assert speech_plan is not None
        chunk = next(
            (
                candidate
                for candidate in speech_plan.super_chunks
                if candidate.index == window.super_chunk_index
            ),
            None,
        )
        if chunk is None:
            return None
        return WindowDisplayBounds(
            start_time=max(0.0, chunk.speech_start - self.vad_display_lead_pad_sec),
            end_time=min(total_duration_sec, chunk.speech_end + self.vad_display_tail_pad_sec),
            super_chunk_index=chunk.index,
        )
```

Do not add provider-level display-bound state. The bounds are computed from the
current `speech_plan` and then travel on `WindowRun`.

- [ ] **Step 4: Add diagnostic fields**

Update `_build_window_diagnostic()`:

```python
        if window_run.display_bounds is not None:
            diagnostic["display_start"] = window_run.display_bounds.start_time
            diagnostic["display_end"] = window_run.display_bounds.end_time
```

- [ ] **Step 5: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: provider tests pass.

- [ ] **Step 6: Commit VAD display bounds**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "feat: carry vad display bounds on windows"
```

---

### Task 9: Route Fully Unresolved Windows to Fallback Segments

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing unresolved fallback test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_fully_unresolved_window_uses_fallback_not_token_segmentation(self) -> None:
        provider, _, _ = self._build_provider_with_models(
            asr_responses=[FakeChunk("C++ C#", language="en")],
            align_responses=[
                [
                    FakeChunk("C", start_time=1.00, end_time=1.10),
                    FakeChunk("C", start_time=1.10, end_time=1.20),
                ]
            ],
        )
        provider._probe_duration_sec = lambda _: 40.0

        doc = provider.transcribe(Path("demo.wav"))

        self.assertEqual([segment.text for segment in doc.segments], ["C++ C#"])
        self.assertEqual(doc.segments[0].tokens, [])
        self.assertLessEqual(doc.segments[0].end_time - doc.segments[0].start_time, 6.0)
```

Add this fallback duration test:

```python
    def test_fallback_segment_does_not_last_until_window_core_end(self) -> None:
        provider = QwenMlxProvider()
        run = WindowRun(
            window=AlignmentWindow(0, 100.0, 250.0, 95.0, 255.0),
            text="bad fallback text",
            language="en",
            tokens=[],
            has_timing_anchor=False,
        )

        segments = provider._fallback_segments_from_windows([run])

        self.assertEqual(len(segments), 1)
        self.assertLessEqual(segments[0].end_time - segments[0].start_time, 6.0)
        self.assertLess(segments[0].end_time, 250.0)
```

- [ ] **Step 2: Run fallback tests to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_fully_unresolved_window_uses_fallback_not_token_segmentation tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_fallback_segment_does_not_last_until_window_core_end
```

Expected: fail because fallback uses full core window or unresolved tokens enter normal segmentation.

- [ ] **Step 3: Ensure unresolved windows produce no usable tokens**

In `_transcribe_window()`, this was started in Task 6:

```python
        usable_projected_tokens = (
            self._usable_projected_tokens(projected_tokens)
            if has_timing_anchor
            else []
        )
```

Confirm `WindowRun.tokens`, `core_tokens`, `left_overlap_tokens`, and `right_overlap_tokens` are empty when `has_timing_anchor` is false.

- [ ] **Step 4: Add unresolved fallback append after token segmentation**

In `transcribe()`, after token segmentation, append fallback segments for successful unresolved windows:

```python
                merged_tokens = self._merge_window_runs(window_runs)
                segments = self._tokens_to_segments(merged_tokens)
                fallback_segments = self._fallback_segments_from_windows(
                    [
                        run
                        for run in window_runs
                        if run.error is None and run.text and not run.has_timing_anchor
                    ]
                )
                if fallback_segments:
                    segments = self._append_segments(segments, fallback_segments)
                if not segments:
                    segments = self._fallback_segments_from_windows(window_runs)
```

Add `_append_segments()` near `_append_tokens()`:

```python
    def _append_segments(self, existing: List[Segment], new_segments: List[Segment]) -> List[Segment]:
        if not new_segments:
            return existing
        merged = list(existing)
        for segment in new_segments:
            merged.append(
                Segment(
                    id=f"seg-{len(merged) + 1}",
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    language=segment.language,
                    tokens=list(segment.tokens),
                    speaker=segment.speaker,
                )
            )
        merged.sort(key=lambda segment: (segment.start_time, segment.end_time))
        for index, segment in enumerate(merged, start=1):
            segment.id = f"seg-{index}"
        return merged
```

- [ ] **Step 5: Replace whole-core fallback duration**

Replace `_fallback_segments_from_windows()`:

```python
    def _fallback_segments_from_windows(self, window_runs: List[WindowRun]) -> List[Segment]:
        segments: List[Segment] = []
        for window_run in window_runs:
            if window_run.error is not None or not window_run.text:
                continue
            start_time = self._fallback_start_time(window_run)
            end_time = self._fallback_end_time(window_run, start_time)
            segments.append(
                Segment(
                    id=f"seg-{len(segments) + 1}",
                    text=window_run.text,
                    start_time=start_time,
                    end_time=end_time,
                    language=window_run.language,
                    tokens=[],
                )
            )
        return segments
```

Add helpers:

```python
    def _fallback_start_time(self, window_run: WindowRun) -> float:
        if window_run.display_bounds is not None:
            return window_run.display_bounds.start_time
        return window_run.window.core_start

    def _fallback_end_time(self, window_run: WindowRun, start_time: float) -> float:
        max_duration = 6.0
        estimated_duration = min(
            max_duration,
            self._estimate_fallback_text_duration(window_run.text, window_run.language),
        )
        end_time = start_time + estimated_duration
        if window_run.display_bounds is not None:
            end_time = min(end_time, window_run.display_bounds.end_time)
        else:
            end_time = min(end_time, window_run.window.core_end)
        return max(start_time, end_time)

    def _estimate_fallback_text_duration(self, text: str, language: Optional[str]) -> float:
        normalized = (language or "").lower()
        if normalized.startswith("zh") or "chinese" in normalized or self._contains_cjk(text):
            char_count = sum(1 for char in text if not char.isspace())
            return max(1.0, char_count * 0.12)
        word_count = len([piece for piece in text.split() if piece])
        return max(1.2, word_count * 0.35)
```

- [ ] **Step 6: Run fallback tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: provider tests pass.

- [ ] **Step 7: Commit fallback routing**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "fix: route unresolved windows to short fallback"
```

---

### Task 10: Clamp Segments to Display Bounds

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing display-bound stabilization test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_vad_display_bounds_clamp_segment_tail(self) -> None:
        provider = QwenMlxProvider()
        segments = [
            Segment(
                id="seg-1",
                text="hello",
                start_time=104.0,
                end_time=123.0,
                language="en",
                tokens=[],
            )
        ]
        bounds = [WindowDisplayBounds(start_time=104.8, end_time=120.35, super_chunk_index=0)]

        stabilized = provider._stabilize_segment_boundaries(
            segments,
            total_duration_sec=200.0,
            display_bounds=bounds,
        )

        self.assertEqual(stabilized[0].start_time, 104.8)
        self.assertEqual(stabilized[0].end_time, 120.35)
```

- [ ] **Step 2: Run display-bound test to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_vad_display_bounds_clamp_segment_tail
```

Expected: fail because `_stabilize_segment_boundaries()` has no `display_bounds` parameter.

- [ ] **Step 3: Pass bounds into stabilization**

Update the stabilization call in `transcribe()`:

```python
                segments = self._stabilize_segment_boundaries(
                    segments,
                    total_duration_sec=total_duration_sec,
                    display_bounds=[
                        run.display_bounds
                        for run in window_runs
                        if run.display_bounds is not None
                    ],
                )
```

Update the signature:

```python
    def _stabilize_segment_boundaries(
        self,
        segments: List[Segment],
        *,
        total_duration_sec: float,
        display_bounds: Iterable[WindowDisplayBounds] | None = None,
        tail_padding_sec: float = 0.12,
        target_max_segment_duration_sec: float = 8.0,
    ) -> List[Segment]:
```

Import `Iterable` is already available in `qwen_mlx.py`; keep using it.

- [ ] **Step 4: Apply display-bound clamp inside stabilization**

After the first normalization loop in `_stabilize_segment_boundaries()`, add:

```python
        bounds = list(display_bounds or [])
        for segment in stabilized:
            bound = self._nearest_display_bound(segment, bounds)
            if bound is None:
                continue
            segment.start_time = max(segment.start_time, bound.start_time)
            segment.end_time = min(segment.end_time, bound.end_time)
            segment.end_time = max(segment.start_time, segment.end_time)
```

Add helper:

```python
    def _nearest_display_bound(
        self,
        segment: Segment,
        bounds: List[WindowDisplayBounds],
    ) -> WindowDisplayBounds | None:
        overlapping = [
            bound
            for bound in bounds
            if segment.end_time >= bound.start_time and segment.start_time <= bound.end_time
        ]
        if not overlapping:
            return None
        return min(
            overlapping,
            key=lambda bound: abs(segment.start_time - bound.start_time),
        )
```

- [ ] **Step 5: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: provider tests pass.

- [ ] **Step 6: Commit display-bound stabilization**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "fix: clamp subtitles to vad display bounds"
```

---

### Task 11: Add Target Segment Duration Splitting

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Test: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing target-duration split test**

Add this test to `QwenProviderWindowedTest`:

```python
    def test_long_token_segment_splits_at_target_duration(self) -> None:
        provider = QwenMlxProvider()
        tokens = [
            Token("one", 0.0, 1.0, unit="word"),
            Token("two", 1.1, 2.0, unit="word"),
            Token("three", 2.1, 3.0, unit="word"),
            Token("four", 3.1, 4.0, unit="word"),
        ]

        segments = provider._tokens_to_segments(tokens, target_max_segment_duration_sec=2.5)

        self.assertEqual([segment.text for segment in segments], ["one two", "three four"])
        self.assertEqual([(segment.start_time, segment.end_time) for segment in segments], [(0.0, 2.0), (2.1, 4.0)])
```

- [ ] **Step 2: Run duration split test to verify failure**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_long_token_segment_splits_at_target_duration
```

Expected: fail because `_tokens_to_segments()` does not accept `target_max_segment_duration_sec`.

- [ ] **Step 3: Add target split parameter**

Update `_tokens_to_segments()` signature:

```python
    def _tokens_to_segments(
        self,
        tokens: Iterable[Token],
        *,
        target_max_segment_duration_sec: float = 8.0,
    ) -> List[Segment]:
```

Update the break logic:

```python
            if current_tokens:
                if previous_end is not None and token.start_time - previous_end >= 1.0:
                    should_break = True
                if self._ends_segment(current_tokens[-1].text):
                    should_break = True
                if (
                    current_tokens
                    and token.end_time - current_tokens[0].start_time > target_max_segment_duration_sec
                ):
                    should_break = True
```

Leave the call in `transcribe()` as:

```python
                segments = self._tokens_to_segments(merged_tokens)
```

- [ ] **Step 4: Run provider tests**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed
```

Expected: provider tests pass.

- [ ] **Step 5: Commit duration splitting**

Run:

```bash
git add src/asr/providers/qwen_mlx.py tests/test_qwen_provider_windowed.py
git commit -m "feat: split long token-backed segments"
```

---

### Task 12: Preserve Exporter Contract

**Files:**
- Modify: `tests/test_exporters.py`
- Test: `tests/test_exporters.py`

- [ ] **Step 1: Write exporter contract test**

Add this test to `ExporterTest` in `tests/test_exporters.py`:

```python
    def test_json_tokens_do_not_expose_provider_timing_source(self) -> None:
        document = TranscriptionDocument(
            source_path="demo.wav",
            provider_name="fake",
            source_media={
                "provider_metadata": {
                    "window_diagnostics": [
                        {
                            "timing_source_counts": {
                                "aligner": 1,
                                "estimated": 1,
                                "unresolved": 0,
                            }
                        }
                    ]
                }
            },
            segments=[
                Segment(
                    id="seg-1",
                    text="I have",
                    start_time=5.08,
                    end_time=5.50,
                    language="en",
                    tokens=[
                        Token(text="I", start_time=5.08, end_time=5.18, unit="word", language="en"),
                        Token(text="have", start_time=5.20, end_time=5.50, unit="word", language="en"),
                    ],
                )
            ],
        )

        payload = json.loads(render_json(document, granularity="token"))

        self.assertNotIn("timing_source", payload["segments"][0]["tokens"][0])
        self.assertEqual(
            payload["source_media"]["provider_metadata"]["window_diagnostics"][0]["timing_source_counts"]["estimated"],
            1,
        )
```

- [ ] **Step 2: Run exporter test**

Run:

```bash
uv run python -m unittest tests.test_exporters
```

Expected: pass because exporters serialize only public `Token` fields.

- [ ] **Step 3: Commit exporter contract test**

Run:

```bash
git add tests/test_exporters.py
git commit -m "test: preserve exporter timing contract"
```

---

### Task 13: Full Regression Verification

**Files:**
- No source edits unless a regression appears.

- [ ] **Step 1: Run full unit suite**

Run:

```bash
uv run python -m unittest
```

Expected: all tests pass.

- [ ] **Step 2: Inspect provider diagnostics shape**

Run:

```bash
uv run python -m unittest tests.test_qwen_provider_windowed.QwenProviderWindowedTest.test_vad_window_run_carries_speech_display_bounds
```

Expected: pass and diagnostics include `display_start` / `display_end`.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff --stat
git diff --check
```

Expected: no whitespace errors; only planned files changed since the last commit.

- [ ] **Step 4: Commit any regression-only fixes**

If Step 1 exposed a regression and the fix touches already planned files, commit with:

```bash
git add src/asr/providers/authority.py src/asr/providers/qwen_mlx.py src/asr/providers/quality.py src/asr/providers/window_merge.py tests/test_authority.py tests/test_qwen_provider_windowed.py tests/test_quality.py tests/test_exporters.py tests/test_window_merge.py
git commit -m "fix: stabilize caption timing regressions"
```

Expected: either no commit is needed, or the commit contains only regression fixes for this plan.

---

## Self-Review

- Spec coverage: Tasks 1-2 cover `ProjectedToken`, detailed projection, repair, and fully unmatched provenance. Tasks 4, 6, and 7 cover repair-before-split, protected edge tokens, and shared core ownership. Tasks 5, 8, and 10 cover `WindowRun.display_bounds` and VAD display clamping. Task 9 covers short fallback segments and fully unresolved fallback routing. Task 3 covers timing-source quality gating. Task 11 covers target segment duration splitting. Task 12 covers unchanged exporter payloads.
- Placeholder scan: this plan contains concrete file paths, tests, commands, expected outcomes, and implementation snippets for every task.
- Type consistency: `ProjectedToken`, `TimingSource`, `WindowDisplayBounds`, `WindowRun.projected_tokens`, `timing_source_counts`, `has_timing_anchor`, and `display_bounds` names are consistent across tasks.
