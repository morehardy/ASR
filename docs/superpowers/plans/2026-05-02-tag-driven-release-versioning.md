# Tag-Driven Release Versioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Git release tags the single source of truth for Python package versions.

**Architecture:** Keep the existing Hatchling build backend and add `hatch-vcs` as the dynamic version source. Runtime version display reads installed package metadata, while the publish workflow fetches full Git tag history and validates release tag shape.

**Tech Stack:** Python 3.14, uv, Hatchling, hatch-vcs, GitHub Actions, PyPI Trusted Publishing.

---

### Task 1: Runtime Version Metadata

**Files:**
- Modify: `src/asr/__init__.py`
- Modify: `tests/test_package.py`

- [x] **Step 1: Replace the package test with metadata behavior tests**

```python
import unittest
from unittest.mock import patch

import asr


class PackageMetadataTest(unittest.TestCase):
    def test_version_comes_from_installed_distribution_metadata(self) -> None:
        with patch("asr.metadata.version", return_value="1.2.3") as version:
            self.assertEqual(asr._read_version(), "1.2.3")
            version.assert_called_once_with("echoalign-asr-mlx")

    def test_version_has_source_checkout_fallback(self) -> None:
        with patch("asr.metadata.version", side_effect=asr.metadata.PackageNotFoundError):
            self.assertEqual(asr._read_version(), "0+unknown")

    def test_version_is_defined(self) -> None:
        self.assertTrue(asr.__version__)
```

- [x] **Step 2: Run test to verify current hard-coded version fails the new contract**

Run: `PYTHONPATH=src uv run --frozen --python 3.14 python -m unittest tests.test_package -v`

Expected: FAIL because `asr._read_version` does not exist.

- [x] **Step 3: Implement runtime metadata lookup**

```python
"""Top-level package for the asr CLI."""

from importlib import metadata

__all__ = ["__version__"]

_DISTRIBUTION_NAME = "echoalign-asr-mlx"
_UNKNOWN_VERSION = "0+unknown"


def _read_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return _UNKNOWN_VERSION


__version__ = _read_version()
```

- [x] **Step 4: Run package metadata tests**

Run: `PYTHONPATH=src uv run --frozen --python 3.14 python -m unittest tests.test_package -v`

Expected: PASS.

### Task 2: Dynamic Build Version

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [x] **Step 1: Configure Hatchling to use hatch-vcs**

```toml
[build-system]
requires = ["hatchling>=1.27.0", "hatch-vcs>=0.5.0"]
build-backend = "hatchling.build"

[project]
name = "echoalign-asr-mlx"
dynamic = ["version"]
description = "Local CLI for extracting subtitles and aligned timestamps from audio and video."
```

```toml
[tool.hatch.version]
source = "vcs"
tag-pattern = "^v(?P<version>.+)$"
```

- [x] **Step 2: Update the lock file**

Run: `uv lock`

Expected: `uv.lock` records the project as dynamically versioned and includes build requirements needed by `hatch-vcs`.

- [x] **Step 3: Build distributions from the current tag history**

Run: `uv build`

Expected: `dist/echoalign_asr_mlx-<derived-version>.tar.gz` and `dist/echoalign_asr_mlx-<derived-version>-py3-none-any.whl`.

### Task 3: Publish Workflow

**Files:**
- Modify: `.github/workflows/publish-pypi.yml`

- [x] **Step 1: Fetch full Git history and tags**

```yaml
      - name: Checkout
        uses: actions/checkout@v6
        with:
          persist-credentials: false
          fetch-depth: 0
          fetch-tags: true
```

- [x] **Step 2: Replace static version comparison with release tag validation**

```yaml
      - name: Check release tag format
        if: github.event_name == 'release'
        env:
          RELEASE_TAG: ${{ github.event.release.tag_name }}
        run: |
          python - <<'PY'
          import os
          import re

          release_tag = os.environ["RELEASE_TAG"]
          if not re.fullmatch(r"v\d+\.\d+\.\d+(?:[a-zA-Z0-9.-]+)?", release_tag):
              raise SystemExit(
                  f"Release tag {release_tag!r} must look like 'vX.Y.Z', for example 'v0.2.1'."
              )
          print(f"Release tag {release_tag} is valid.")
          PY
```

- [x] **Step 3: Build once through the workflow-equivalent local commands**

Run:

```bash
PYTHONPATH=src uv run --frozen --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
uv build
uvx twine check dist/*
```

Expected: all commands pass.

### Task 4: Release Documentation

**Files:**
- Modify: `README.md`

- [x] **Step 1: Update the Python Package Distribution and GitHub Release sections**

Replace the existing distribution example that names a static wheel with:

````markdown
Build source + wheel artifacts:

```bash
uv build
```

The package version is derived from Git tags. A clean release build from tag
`v0.2.1` produces `0.2.1` distributions.
````

Replace the release flow with:

```markdown
Release flow:

1. Configure a Trusted Publisher in PyPI for this project.
2. Merge release-ready code to `main`.
3. Create and publish a GitHub Release tagged `vX.Y.Z`, for example `v0.2.1`.
4. GitHub Actions runs tests, builds distributions with the tag-derived version,
   checks them with Twine, and publishes to PyPI.
```

- [x] **Step 2: Re-run the full verification set**

Run:

```bash
PYTHONPATH=src uv run --frozen --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
uv build
uvx twine check dist/*
```

Expected: all commands pass.
