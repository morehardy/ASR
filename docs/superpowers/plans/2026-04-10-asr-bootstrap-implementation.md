# ASR Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first runnable `asr` project skeleton with a stable CLI contract, canonical transcript model, exporter pipeline, media normalization, and a provider abstraction that can host the planned MLX Qwen backend.

**Architecture:** Keep the public CLI narrow and backend-neutral. Normalize media up front, convert provider output into a canonical transcript document, then render that document into `srt`, `vtt`, and `json`. The planned MLX provider lives behind an internal provider interface so later backends can reuse the same pipeline and exporters.

**Tech Stack:** Python 3.14 target, `uv`, standard-library-first implementation, `unittest`, `ffmpeg` / `ffprobe`, optional `mlx-audio`

---

### Task 1: Project Skeleton And Tooling

**Files:**
- Create: `.python-version`
- Create: `pyproject.toml`
- Create: `src/asr/__init__.py`
- Create: `src/asr/__main__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_package.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from asr import __version__


class PackageMetadataTest(unittest.TestCase):
    def test_version_is_defined(self) -> None:
        self.assertTrue(__version__)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_package -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'asr'`

- [ ] **Step 3: Write minimal implementation**

```python
__all__ = ["__version__"]

__version__ = "0.1.0"
```

```python
from asr.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_package -v`
Expected: PASS

### Task 2: Discovery And Output Planning

**Files:**
- Create: `src/asr/discovery.py`
- Create: `src/asr/output.py`
- Test: `tests/test_discovery.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr.discovery import discover_media_files
from asr.output import build_output_path


class DiscoveryTest(unittest.TestCase):
    def test_directory_scan_is_non_recursive_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp4").write_text("x", encoding="utf-8")
            (root / "nested").mkdir()
            (root / "nested" / "b.wav").write_text("x", encoding="utf-8")

            files = discover_media_files(root, recursive=False)

            self.assertEqual(files, [root / "a.mp4"])

    def test_output_path_keeps_relative_structure(self) -> None:
        source = Path("/project/media/nested/demo.wav")
        root = Path("/project/media")
        target = build_output_path(source=source, input_root=root, output_root=root / "outputs", suffix=".srt")
        self.assertEqual(target, Path("/project/media/outputs/nested/demo.srt"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_discovery -v`
Expected: FAIL with import errors for `asr.discovery` or `asr.output`

- [ ] **Step 3: Write minimal implementation**

```python
SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".aac",
    ".mp4", ".mov", ".m4v", ".mkv", ".webm",
}
```

```python
def discover_media_files(path: Path, recursive: bool = False) -> list[Path]:
    ...
```

```python
def build_output_path(source: Path, input_root: Path, output_root: Path, suffix: str) -> Path:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_discovery -v`
Expected: PASS

### Task 3: Canonical Transcript Model And Exporters

**Files:**
- Create: `src/asr/models.py`
- Create: `src/asr/exporters.py`
- Test: `tests/test_exporters.py`

- [ ] **Step 1: Write the failing test**

```python
import json
import unittest

from asr.exporters import render_json, render_srt, render_vtt
from asr.models import Segment, Token, TranscriptionDocument


class ExporterTest(unittest.TestCase):
    def test_renderers_emit_expected_formats(self) -> None:
        document = TranscriptionDocument(
            source_path="demo.wav",
            provider_name="fake",
            segments=[
                Segment(
                    id="seg-1",
                    text="Hello world",
                    start_time=0.0,
                    end_time=1.5,
                    language="en",
                    tokens=[
                        Token(text="Hello", start_time=0.0, end_time=0.7, unit="word", language="en"),
                        Token(text="world", start_time=0.8, end_time=1.5, unit="word", language="en"),
                    ],
                )
            ],
        )

        srt_text = render_srt(document)
        vtt_text = render_vtt(document)
        payload = json.loads(render_json(document))

        self.assertIn("1\n00:00:00,000 --> 00:00:01,500\nHello world", srt_text)
        self.assertTrue(vtt_text.startswith("WEBVTT"))
        self.assertEqual(payload["segments"][0]["tokens"][0]["text"], "Hello")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_exporters -v`
Expected: FAIL with import errors for model and exporter modules

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass(slots=True)
class Token:
    text: str
    start_time: float
    end_time: float
    unit: str
    language: str | None = None
```

```python
def render_srt(document: TranscriptionDocument) -> str:
    ...
```

```python
def render_vtt(document: TranscriptionDocument) -> str:
    ...
```

```python
def render_json(document: TranscriptionDocument) -> str:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_exporters -v`
Expected: PASS

### Task 4: Media Normalization, Provider Boundary, And Pipeline

**Files:**
- Create: `src/asr/media.py`
- Create: `src/asr/providers/__init__.py`
- Create: `src/asr/providers/base.py`
- Create: `src/asr/providers/qwen_mlx.py`
- Create: `src/asr/pipeline.py`
- Test: `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path

from asr.models import Segment, Token, TranscriptionDocument
from asr.pipeline import process_media_file
from asr.providers.base import Provider


class FakeProvider(Provider):
    name = "fake"

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        return TranscriptionDocument(
            source_path=str(audio_path),
            provider_name=self.name,
            segments=[
                Segment(
                    id="seg-1",
                    text="hello",
                    start_time=0.0,
                    end_time=0.4,
                    language="en",
                    tokens=[Token(text="hello", start_time=0.0, end_time=0.4, unit="word", language="en")],
                )
            ],
        )


class FakeMediaPreparer:
    def prepare(self, source_path: Path) -> Path:
        return source_path.with_suffix(".wav")


class PipelineTest(unittest.TestCase):
    def test_pipeline_uses_media_preparer_before_provider(self) -> None:
        document = process_media_file(
            source_path=Path("demo.mp4"),
            provider=FakeProvider(),
            media_preparer=FakeMediaPreparer(),
        )

        self.assertEqual(document.source_path, "demo.wav")
        self.assertEqual(document.provider_name, "fake")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_pipeline -v`
Expected: FAIL because the provider base and pipeline do not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
class Provider(Protocol):
    name: str

    def transcribe(self, audio_path: Path) -> TranscriptionDocument:
        ...
```

```python
def process_media_file(source_path: Path, provider: Provider, media_preparer: MediaPreparer) -> TranscriptionDocument:
    prepared_path = media_preparer.prepare(source_path)
    return provider.transcribe(prepared_path)
```

```python
class FfmpegMediaPreparer:
    def prepare(self, source_path: Path) -> Path:
        ...
```

```python
class QwenMlxProvider:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_pipeline -v`
Expected: PASS

### Task 5: CLI Integration

**Files:**
- Create: `src/asr/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from asr.cli import build_parser


class CliParserTest(unittest.TestCase):
    def test_defaults_to_current_directory_when_input_missing(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.input, Path.cwd())

    def test_recursive_is_opt_in(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["./media", "--recursive"])
        self.assertTrue(args.recursive)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src python3 -m unittest tests.test_cli -v`
Expected: FAIL with import errors for `asr.cli`

- [ ] **Step 3: Write minimal implementation**

```python
def build_parser() -> argparse.ArgumentParser:
    ...
```

```python
def main(argv: Sequence[str] | None = None) -> int:
    ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src python3 -m unittest tests.test_cli -v`
Expected: PASS

## Self-Review

- Spec coverage: the bootstrap plan covers project tooling, public CLI defaults, supported input discovery, canonical transcript data, output exporters, media normalization, and the provider abstraction. It intentionally does not promise a fully production-ready MLX integration in the first pass.
- Placeholder scan: no `TODO` or unresolved placeholders remain in the execution steps. Ellipses appear only inside sketch snippets and are resolved during implementation.
- Type consistency: `TranscriptionDocument`, `Segment`, `Token`, `Provider`, and `process_media_file()` signatures are consistent across tasks.
