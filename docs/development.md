# Development Guide

This document collects contributor and maintainer workflows for
`echoalign-asr-mlx`. The user-facing README stays focused on installing and
running `easr`.

## Project Shape

- package name: `echoalign-asr-mlx`
- public CLI command: `easr`
- source package: `src/asr`
- test suite: `tests`
- Python target: `>=3.14,<3.15`
- environment manager: `uv`
- CI workflow: `.github/workflows/ci.yml`
- PyPI publish workflow: `.github/workflows/publish-pypi.yml`

Use `easr` in user-facing documentation. Some older planning notes may mention
`asr`, but the current public entry point is `easr`.

## Local Setup

Install system media dependencies:

```bash
brew install ffmpeg
```

Create/update the project environment with the MLX runtime extra:

```bash
uv sync --extra mlx
```

Verify the CLI:

```bash
uv run --python 3.14 --extra mlx easr --help
uv run --python 3.14 --extra mlx easr --version
```

## Test Commands

Run the full unit test suite:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
```

Run a focused test module:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_authority
```

Dry-check CLI parsing/help without the MLX extra:

```bash
uv run --python 3.14 easr --help
```

Run a local transcription smoke test against your own media:

```bash
uv run --python 3.14 --extra mlx easr /path/to/demo.mp4 --verbose --output-dir tmp/easr-smoke
```

## Build Distributions

Build source and wheel artifacts:

```bash
uv build
```

The package version is derived from Git tags through `hatch-vcs`. A clean
release build from tag `v0.2.1` produces `0.2.1` distributions.

Install a local wheel in a target environment:

```bash
python3.14 -m pip install dist/echoalign_asr_mlx-<version>-py3-none-any.whl
```

For full transcription runtime from a source checkout, install with the MLX
extra:

```bash
python3.14 -m pip install ".[mlx]"
```

After publishing to an index such as PyPI, end users can install with:

```bash
python3.14 -m pip install "echoalign-asr-mlx[mlx]"
```

## Release to PyPI

This repository includes a publish workflow at:

```text
.github/workflows/publish-pypi.yml
```

Release flow:

1. Configure a Trusted Publisher in PyPI for this project:
   - project: `echoalign-asr-mlx`
   - owner/repo: the GitHub repository
   - workflow: `publish-pypi.yml`
   - environment: `pypi`
2. Merge release-ready code to `main`.
3. Create and publish a GitHub Release tagged `vX.Y.Z`, for example `v0.2.1`.
4. GitHub Actions runs tests, builds distributions with the tag-derived version,
   checks them with Twine, and publishes to PyPI.

The workflow also supports manual `workflow_dispatch`, but manual publishing
must run from a release tag such as `v0.2.1`.

## Developer Notes

- The provider boundary is intentionally hidden from the public CLI.
- Keep output layout stable across provider changes.
- Keep JSON exports rich enough to preserve fine-grained token alignment.
- Do not expose ASR and aligner model selection as separate public options
  unless the CLI contract is intentionally redesigned.
- `--verbose` writes metrics JSON and is the primary local optimization aid.
