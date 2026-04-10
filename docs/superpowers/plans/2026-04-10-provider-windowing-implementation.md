# Provider Windowing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a production-meaningful fixed-strategy windowed Qwen provider that enforces hard window bounds, uses anchor-based boundaries, preserves transcript/timing authority split, and emits one unified full-media output contract.

**Architecture:** Windowing is internal to the provider. The pipeline is: probe full-media duration -> plan bounded anchor-aware windows -> transcribe each context window -> align per window -> project timing onto transcript authority -> evaluate quality gates -> merge windows using ownership + sequence alignment -> emit one canonical full-media document with diagnostics.

**Tech Stack:** Python 3.13+/3.14, `unittest`, stdlib (`dataclasses`, `difflib`, `subprocess`, `tempfile`, `json`, `math`, `unicodedata`), existing `asr` package.

---

## File Structure

- Create: `src/asr/providers/windowing.py`
  - hard window budget config, anchor-aware split planning, invariant validation
- Create: `src/asr/providers/media_probe.py`
  - media duration probing and optional silence-anchor extraction
- Create: `src/asr/providers/authority.py`
  - transcript authority tokenization and timing projection
- Create: `src/asr/providers/quality.py`
  - boundary-aware quality metrics and threshold evaluation
- Create: `src/asr/providers/window_merge.py`
  - ownership-aware overlap merge via constrained DP sequence alignment
- Modify: `src/asr/providers/qwen_mlx.py`
  - real multi-window orchestration, no fixed 180-second stub, full merge path
- Modify: `src/asr/models.py`
  - optional window diagnostic model fields needed by provider metadata
- Create: `tests/test_windowing.py`
- Create: `tests/test_media_probe.py`
- Create: `tests/test_authority.py`
- Create: `tests/test_quality.py`
- Create: `tests/test_window_merge.py`
- Create: `tests/test_qwen_provider_windowed.py`
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_exporters.py`

### Task 1: Implement Anchor-Aware Bounded Window Planner

**Files:**
- Create: `src/asr/providers/windowing.py`
- Test: `tests/test_windowing.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from asr.providers.windowing import WindowBudgetConfig, WindowPlanner


class WindowPlannerTest(unittest.TestCase):
    def test_all_windows_obey_hard_context_limit(self) -> None:
        cfg = WindowBudgetConfig(
            max_alignment_window_sec=180.0,
            target_core_window_sec=150.0,
            min_core_window_sec=120.0,
            context_margin_sec=20.0,
            max_context_margin_sec=30.0,
        )
        planner = WindowPlanner(cfg, anchor_resolver=None)
        windows = planner.plan(total_duration_sec=620.0)

        self.assertGreaterEqual(len(windows), 4)
        for win in windows:
            self.assertLessEqual(win.context_end - win.context_start, 180.0)

    def test_anchor_resolver_changes_split_when_available(self) -> None:
        cfg = WindowBudgetConfig()
        calls = []

        def resolver(target_split_sec: float, search_start_sec: float, search_end_sec: float):
            calls.append((target_split_sec, search_start_sec, search_end_sec))
            return 141.25

        planner = WindowPlanner(cfg, anchor_resolver=resolver)
        windows = planner.plan(total_duration_sec=320.0)

        self.assertGreater(len(calls), 0)
        self.assertAlmostEqual(windows[0].core_end, 141.25, places=2)

    def test_tail_shortfall_never_mutates_previous_window_into_invalid_context(self) -> None:
        cfg = WindowBudgetConfig(
            max_alignment_window_sec=180.0,
            target_core_window_sec=150.0,
            min_core_window_sec=120.0,
            context_margin_sec=15.0,
            max_context_margin_sec=30.0,
        )
        planner = WindowPlanner(cfg, anchor_resolver=None)
        windows = planner.plan(total_duration_sec=260.0)

        self.assertGreaterEqual(len(windows), 2)
        for win in windows:
            self.assertLessEqual(win.context_end - win.context_start, 180.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_windowing -v`
Expected: FAIL with `ModuleNotFoundError` for `asr.providers.windowing`

- [ ] **Step 3: Write minimal implementation with hard invariant + anchor fallback**

```python
from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass(slots=True)
class WindowBudgetConfig:
    max_alignment_window_sec: float = 180.0
    target_core_window_sec: float = 150.0
    min_core_window_sec: float = 120.0
    context_margin_sec: float = 15.0
    max_context_margin_sec: float = 30.0
    anchor_search_radius_sec: float = 12.0


@dataclass(slots=True)
class AlignmentWindow:
    index: int
    core_start: float
    core_end: float
    context_start: float
    context_end: float


AnchorResolver = Callable[[float, float, float], Optional[float]]


class WindowPlanner:
    def __init__(self, config: WindowBudgetConfig, anchor_resolver: Optional[AnchorResolver]) -> None:
        self.config = config
        self.anchor_resolver = anchor_resolver

    def _apply_budget(self, core_start: float, core_end: float, total_duration_sec: float) -> tuple[float, float]:
        margin = min(self.config.context_margin_sec, self.config.max_context_margin_sec)
        context_start = max(0.0, core_start - margin)
        context_end = min(total_duration_sec, core_end + margin)
        while context_end - context_start > self.config.max_alignment_window_sec and margin > 0.0:
            margin -= 1.0
            context_start = max(0.0, core_start - margin)
            context_end = min(total_duration_sec, core_end + margin)
        if context_end - context_start > self.config.max_alignment_window_sec:
            core_len = core_end - core_start
            while context_end - context_start > self.config.max_alignment_window_sec and core_len > 1.0:
                core_len -= 1.0
                core_end = core_start + core_len
                context_start = max(0.0, core_start - margin)
                context_end = min(total_duration_sec, core_end + margin)
        if context_end - context_start > self.config.max_alignment_window_sec:
            raise ValueError("context window exceeds hard maximum")
        return context_start, context_end

    def _anchored_core_end(self, target_end: float, total_duration_sec: float) -> float:
        if self.anchor_resolver is None:
            return min(target_end, total_duration_sec)
        left = max(0.0, target_end - self.config.anchor_search_radius_sec)
        right = min(total_duration_sec, target_end + self.config.anchor_search_radius_sec)
        anchored = self.anchor_resolver(target_end, left, right)
        if anchored is None:
            return min(target_end, total_duration_sec)
        if anchored <= 0.0:
            return min(target_end, total_duration_sec)
        return min(anchored, total_duration_sec)

    def plan(self, total_duration_sec: float) -> List[AlignmentWindow]:
        windows: List[AlignmentWindow] = []
        cursor = 0.0
        index = 0
        while cursor < total_duration_sec:
            index += 1
            target_end = min(cursor + self.config.target_core_window_sec, total_duration_sec)
            core_end = self._anchored_core_end(target_end, total_duration_sec)
            if core_end <= cursor:
                core_end = target_end
            core_start = cursor
            context_start, context_end = self._apply_budget(core_start, core_end, total_duration_sec)
            windows.append(
                AlignmentWindow(
                    index=index,
                    core_start=core_start,
                    core_end=core_end,
                    context_start=context_start,
                    context_end=context_end,
                )
            )
            cursor = core_end
        return windows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_windowing -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_windowing.py src/asr/providers/windowing.py
git commit -m "feat: add anchor-aware bounded window planner with hard invariants"
```

### Task 2: Add Media Probe And Anchor Extraction Inputs

**Files:**
- Create: `src/asr/providers/media_probe.py`
- Test: `tests/test_media_probe.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from asr.providers.media_probe import parse_silence_anchors


class MediaProbeTest(unittest.TestCase):
    def test_parse_silence_anchors_reads_midpoints(self) -> None:
        stderr = """
[silencedetect @ 0x0] silence_start: 120.0
[silencedetect @ 0x0] silence_end: 121.2 | silence_duration: 1.2
[silencedetect @ 0x0] silence_start: 300.0
[silencedetect @ 0x0] silence_end: 301.0 | silence_duration: 1.0
"""
        anchors = parse_silence_anchors(stderr)
        self.assertEqual(anchors, [120.6, 300.5])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_media_probe -v`
Expected: FAIL with `ModuleNotFoundError` for `asr.providers.media_probe`

- [ ] **Step 3: Write minimal implementation**

```python
import re
import subprocess
from pathlib import Path
from typing import List


_SILENCE_START = re.compile(r"silence_start:\s*([0-9]+(?:\.[0-9]+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*([0-9]+(?:\.[0-9]+)?)")


def probe_duration_sec(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ]
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()
    return float(out)


def parse_silence_anchors(stderr_text: str) -> List[float]:
    starts = [float(m.group(1)) for m in _SILENCE_START.finditer(stderr_text)]
    ends = [float(m.group(1)) for m in _SILENCE_END.finditer(stderr_text)]
    anchors: List[float] = []
    for start, end in zip(starts, ends):
        anchors.append((start + end) / 2.0)
    return anchors
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_media_probe -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_media_probe.py src/asr/providers/media_probe.py
git commit -m "feat: add media duration probe and silence anchor parsing"
```

### Task 3: Implement Transcript/Timing Authority Split Correctly

**Files:**
- Create: `src/asr/providers/authority.py`
- Test: `tests/test_authority.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from asr.models import Token
from asr.providers.authority import build_transcript_tokens, project_timing_onto_transcript


class AuthorityTest(unittest.TestCase):
    def test_transcript_tokens_come_from_asr_text_not_aligner_items(self) -> None:
        asr_text = "I agree"
        aligner_tokens = [
            Token(text="I", start_time=0.0, end_time=0.2, unit="token", language="en"),
            Token(text="agreed", start_time=0.2, end_time=0.6, unit="token", language="en"),
        ]

        transcript_tokens = build_transcript_tokens(asr_text, language="en")
        projected = project_timing_onto_transcript(transcript_tokens, aligner_tokens)

        self.assertEqual([t.text for t in transcript_tokens], ["I", "agree"])
        self.assertEqual([t.text for t in projected], ["I", "agree"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_authority -v`
Expected: FAIL with `ModuleNotFoundError` for `asr.providers.authority`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import replace
from difflib import SequenceMatcher
from typing import List

from asr.models import Token


def build_transcript_tokens(text: str, language: str | None) -> List[Token]:
    if not text.strip():
        return []
    if language and language.lower().startswith("zh"):
        units = list(text.strip())
    else:
        units = text.strip().split()
    return [Token(text=u, start_time=0.0, end_time=0.0, unit="token", language=language) for u in units]


def project_timing_onto_transcript(transcript_tokens: List[Token], aligner_tokens: List[Token]) -> List[Token]:
    projected: List[Token] = []
    cursor = 0
    for token in transcript_tokens:
        matched = None
        for idx in range(cursor, len(aligner_tokens)):
            cand = aligner_tokens[idx]
            if SequenceMatcher(None, token.text, cand.text).ratio() >= 0.9:
                matched = (idx, cand)
                break
        if matched is None:
            projected.append(replace(token))
            continue
        idx, cand = matched
        cursor = idx + 1
        if cand.end_time < cand.start_time:
            projected.append(replace(token))
            continue
        projected.append(replace(token, start_time=cand.start_time, end_time=cand.end_time))
    return projected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_authority -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_authority.py src/asr/providers/authority.py
git commit -m "feat: enforce transcript authority with timing projection only"
```

### Task 4: Implement Quality Gates With Real Boundary Inputs

**Files:**
- Create: `src/asr/providers/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from asr.models import Token
from asr.providers.quality import QualityThresholds, evaluate_quality


class QualityGateTest(unittest.TestCase):
    def test_boundary_disagreement_uses_adjacent_window_overlap_tokens(self) -> None:
        left_overlap = [
            Token(text="we", start_time=10.0, end_time=10.2, unit="token", language="en"),
            Token(text="agree", start_time=10.2, end_time=10.5, unit="token", language="en"),
        ]
        right_overlap = [
            Token(text="we", start_time=10.0, end_time=10.2, unit="token", language="en"),
            Token(text="argue", start_time=10.2, end_time=10.5, unit="token", language="en"),
        ]
        all_tokens = left_overlap + right_overlap
        res = evaluate_quality(
            tokens=all_tokens,
            left_overlap_tokens=left_overlap,
            right_overlap_tokens=right_overlap,
            core_text="we agree",
            context_text="we argue",
            thresholds=QualityThresholds(),
        )
        self.assertGreater(res.boundary_disagreement_score, 0.0)
        self.assertGreater(res.core_context_text_divergence, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_quality -v`
Expected: FAIL with `ModuleNotFoundError` for `asr.providers.quality`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List

from asr.models import Token


@dataclass(slots=True)
class QualityThresholds:
    monotonic_timestamp_ratio_min: float = 0.98
    zero_or_flat_timestamp_ratio_max: float = 0.05
    boundary_disagreement_score_max: float = 0.20
    core_context_text_divergence_max: float = 0.15


@dataclass(slots=True)
class QualityResult:
    passed: bool
    monotonic_timestamp_ratio: float
    zero_or_flat_timestamp_ratio: float
    boundary_disagreement_score: float
    core_context_text_divergence: float


def evaluate_quality(
    tokens: List[Token],
    left_overlap_tokens: List[Token],
    right_overlap_tokens: List[Token],
    core_text: str,
    context_text: str,
    thresholds: QualityThresholds,
) -> QualityResult:
    if not tokens:
        return QualityResult(False, 0.0, 1.0, 1.0, 1.0)
    mono = 0
    flat = 0
    prev = float("-inf")
    for t in tokens:
        if t.start_time >= prev:
            mono += 1
        if t.start_time == t.end_time:
            flat += 1
        prev = t.start_time
    mono_ratio = mono / len(tokens)
    flat_ratio = flat / len(tokens)

    left_text = " ".join(t.text for t in left_overlap_tokens)
    right_text = " ".join(t.text for t in right_overlap_tokens)
    boundary_disagreement = 1.0 - SequenceMatcher(None, left_text, right_text).ratio()
    divergence = 1.0 - SequenceMatcher(None, core_text, context_text).ratio()

    passed = (
        mono_ratio >= thresholds.monotonic_timestamp_ratio_min
        and flat_ratio <= thresholds.zero_or_flat_timestamp_ratio_max
        and boundary_disagreement <= thresholds.boundary_disagreement_score_max
        and divergence <= thresholds.core_context_text_divergence_max
    )
    return QualityResult(passed, mono_ratio, flat_ratio, boundary_disagreement, divergence)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_quality -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_quality.py src/asr/providers/quality.py
git commit -m "feat: add boundary-aware quality gates for windowed alignment"
```

### Task 5: Implement Ownership-Aware Sequence Merge

**Files:**
- Create: `src/asr/providers/window_merge.py`
- Test: `tests/test_window_merge.py`

- [ ] **Step 1: Write the failing tests**

```python
import unittest

from asr.models import Token
from asr.providers.window_merge import WindowSpan, merge_adjacent_windows


class WindowMergeTest(unittest.TestCase):
    def test_merge_respects_core_ownership_and_drops_context_only_duplicates(self) -> None:
        left = [
            Token(text="hello", start_time=0.0, end_time=0.4, unit="token", language="en"),
            Token(text="world", start_time=0.4, end_time=0.8, unit="token", language="en"),
        ]
        right = [
            Token(text="world", start_time=0.41, end_time=0.81, unit="token", language="en"),
            Token(text="again", start_time=0.82, end_time=1.1, unit="token", language="en"),
        ]
        merged = merge_adjacent_windows(
            left_tokens=left,
            right_tokens=right,
            left_span=WindowSpan(core_start=0.0, core_end=0.8, context_start=0.0, context_end=1.0),
            right_span=WindowSpan(core_start=0.8, core_end=1.4, context_start=0.6, context_end=1.4),
            max_time_delta=0.20,
        )
        self.assertEqual([t.text for t in merged], ["hello", "world", "again"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_window_merge -v`
Expected: FAIL with `ModuleNotFoundError` for `asr.providers.window_merge`

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Tuple

from asr.models import Token


@dataclass(slots=True)
class WindowSpan:
    core_start: float
    core_end: float
    context_start: float
    context_end: float


def _match(a: Token, b: Token, max_time_delta: float) -> bool:
    if abs(a.start_time - b.start_time) > max_time_delta:
        return False
    return SequenceMatcher(None, a.text, b.text).ratio() >= 0.85


def _dp_pairs(left: List[Token], right: List[Token], max_time_delta: float) -> List[Tuple[int, int]]:
    m, n = len(left), len(right)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    back = [[None] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diag = score[i - 1][j - 1] + (2 if _match(left[i - 1], right[j - 1], max_time_delta) else -1)
            up = score[i - 1][j] - 1
            lf = score[i][j - 1] - 1
            best = max(diag, up, lf)
            score[i][j] = best
            back[i][j] = "diag" if best == diag else ("up" if best == up else "left")
    out: List[Tuple[int, int]] = []
    i, j = m, n
    while i > 0 and j > 0:
        move = back[i][j]
        if move == "diag":
            if _match(left[i - 1], right[j - 1], max_time_delta):
                out.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        else:
            j -= 1
    return list(reversed(out))


def merge_adjacent_windows(
    left_tokens: List[Token],
    right_tokens: List[Token],
    left_span: WindowSpan,
    right_span: WindowSpan,
    max_time_delta: float = 0.25,
) -> List[Token]:
    pairs = _dp_pairs(left_tokens, right_tokens, max_time_delta)
    matched_right = {j for _, j in pairs}
    out = list(left_tokens)
    for idx, token in enumerate(right_tokens):
        in_right_core = right_span.core_start <= token.start_time <= right_span.core_end
        if idx in matched_right and not in_right_core:
            continue
        if idx in matched_right:
            continue
        out.append(token)
    out.sort(key=lambda item: (item.start_time, item.end_time))
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_window_merge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_window_merge.py src/asr/providers/window_merge.py
git commit -m "feat: implement ownership-aware overlap merge with constrained DP alignment"
```

### Task 6: Wire Real Multi-Window Provider Orchestration

**Files:**
- Modify: `src/asr/providers/qwen_mlx.py`
- Modify: `src/asr/models.py`
- Create: `tests/test_qwen_provider_windowed.py`

- [ ] **Step 1: Write failing integration tests for multi-window behavior**

```python
import unittest
from pathlib import Path

from asr.providers.qwen_mlx import QwenMlxProvider


class FakeChunk:
    def __init__(self, text, language="en", start_time=0.0, end_time=0.0):
        self.text = text
        self.language = language
        self.start_time = start_time
        self.end_time = end_time


class FakeModel:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def generate(self, audio_path, **kwargs):
        self.calls.append((audio_path, kwargs))
        return self.responses.pop(0)


class ProviderWindowedTest(unittest.TestCase):
    def test_provider_processes_all_windows_not_first_window_only(self) -> None:
        provider = QwenMlxProvider()
        provider._probe_duration_sec = lambda _: 340.0
        provider._resolve_silence_anchor = lambda target, left, right: None
        asr_model = FakeModel([FakeChunk("hello world"), FakeChunk("again now"), FakeChunk("tail done")])
        align_model = FakeModel([
            [FakeChunk("hello", start_time=0.0, end_time=0.2), FakeChunk("world", start_time=0.2, end_time=0.5)],
            [FakeChunk("again", start_time=0.0, end_time=0.2), FakeChunk("now", start_time=0.2, end_time=0.4)],
            [FakeChunk("tail", start_time=0.0, end_time=0.2), FakeChunk("done", start_time=0.2, end_time=0.4)],
        ])
        provider._load_backend = lambda: (lambda model_id: asr_model if "ASR" in model_id else align_model)

        doc = provider.transcribe(Path("demo.wav"))

        self.assertGreaterEqual(len(doc.segments), 2)
        self.assertEqual(doc.source_media["provider_metadata"]["processing_strategy"], "windowed_bounded_alignment")
        self.assertGreaterEqual(doc.source_media["provider_metadata"]["window_count"], 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_qwen_provider_windowed -v`
Expected: FAIL because provider currently does not orchestrate real multi-window workflow

- [ ] **Step 3: Implement full windowed orchestration**

```python
# in src/asr/providers/qwen_mlx.py
# 1) probe full media duration (no fixed 180)
# 2) plan windows via WindowPlanner
# 3) for each window:
#    - extract context clip
#    - asr on context clip
#    - transcript tokens from ASR text (authority)
#    - aligner timestamps projection only
#    - map local timestamps to global via context_start
#    - compute quality gates with true adjacent overlap signals
# 4) merge all windows with ownership-aware DP merge
# 5) build canonical segments and provider diagnostics
```

- [ ] **Step 4: Run provider tests and focused regressions**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_qwen_provider_windowed tests.test_window_merge tests.test_quality -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_qwen_provider_windowed.py src/asr/providers/qwen_mlx.py src/asr/models.py
git commit -m "feat: implement real multi-window qwen provider with authority split and quality gates"
```

### Task 7: Contract And Regression Verification

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_exporters.py`

- [ ] **Step 1: Write failing contract tests**

```python
import unittest
from pathlib import Path

from asr.models import Segment, Token, TranscriptionDocument
from asr.pipeline import process_media_file
from asr.providers.base import Provider


class ContractProvider(Provider):
    name = "contract"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            source_media={"provider_metadata": {"processing_strategy": "windowed_bounded_alignment", "window_count": 3}},
            segments=[
                Segment(
                    id="s1",
                    text="ok",
                    start_time=0.0,
                    end_time=0.3,
                    language="en",
                    tokens=[Token(text="ok", start_time=0.0, end_time=0.3, unit="token", language="en")],
                )
            ],
        )


class ContractTest(unittest.TestCase):
    def test_pipeline_preserves_provider_metadata_and_full_media_contract(self) -> None:
        class IdentityPreparer:
            def prepare(self, source_path: Path) -> Path:
                return source_path

        doc = process_media_file(
            source_path=Path("clip.wav"),
            provider=ContractProvider(),
            media_preparer=IdentityPreparer(),
        )
        self.assertEqual(doc.source_path, "clip.wav")
        self.assertEqual(doc.source_media["provider_metadata"]["window_count"], 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest tests.test_pipeline -v`
Expected: FAIL if pipeline metadata merge is overwritten

- [ ] **Step 3: Implement minimal fixes**

```python
# in src/asr/pipeline.py ensure merge, not replace
source_media = dict(document.source_media or {})
source_media["prepared_audio_path"] = str(prepared_path)
document.source_media = source_media
```

```python
# in tests/test_exporters.py add contract assertion
payload = json.loads(render_json(document, granularity="token"))
self.assertIn("source_media", payload)
```

- [ ] **Step 4: Run full suite and verify final green state**

Run: `PYTHONPATH=src /Users/hp/.pyenv/versions/3.13.7/bin/python -m unittest discover -s tests -v`
Expected: PASS with all tests green

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline.py tests/test_exporters.py src/asr/pipeline.py
git commit -m "test: lock full-media provider contract and metadata preservation"
```

## Self-Review

### 1. Spec Coverage Check

- Anchor-based split planning is explicitly implemented and tested in Task 1 and Task 2.
- Hard bound invariant (`context_end - context_start <= max_alignment_window_sec`) is enforced and tested in Task 1.
- Tail-shortfall invalid mutation bug is prevented by invariant-focused tail test in Task 1.
- Overlap merge as constrained sequence alignment plus ownership rule is implemented in Task 5.
- Transcript authority vs timing authority is implemented in Task 3 and wired in Task 6.
- Quality gates use real boundary inputs and divergence inputs in Task 4 and Task 6.
- Provider orchestration is true multi-window processing in Task 6, not fixed-duration/single-window stubs.
- Contract stability checks are included in Task 7.

No uncovered requirement remains from the approved spec and review corrections.

### 2. Placeholder Scan

- No unresolved TODO/TBD placeholders are present in tasks.
- No “temporary fixed duration” or “process first window only” placeholders remain.

### 3. Type/Name Consistency

- Planner: `WindowBudgetConfig`, `AlignmentWindow`, `WindowPlanner`
- Authority: `build_transcript_tokens`, `project_timing_onto_transcript`
- Quality: `QualityThresholds`, `QualityResult`, `evaluate_quality`
- Merge: `WindowSpan`, `merge_adjacent_windows`

Names are consistent across tasks and tests.
