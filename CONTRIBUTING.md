# Contributing to echoalign-asr-mlx

Thanks for taking the time to improve `echoalign-asr-mlx`.

This project is a local Apple Silicon CLI for extracting subtitles and timestamp-aligned transcription data from audio and video files. Contributions are most useful when they preserve the stable CLI contract, the output layout, and the backend-neutral transcription model.

## Before You Start

- Open an issue first for larger behavior changes, new providers, output format changes, or anything that changes the JSON contract.
- Small documentation fixes, typo fixes, and focused test improvements can go straight to a pull request.
- Do not commit large media files, model files, generated output directories, private audio/video, or local environment folders.
- Contributions are accepted under the MIT License.

## Development Setup

Install the project dependencies:

```bash
uv sync
```

For the full local transcription runtime, install the MLX extra:

```bash
uv sync --extra mlx
```

Run the unit test suite:

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
```

Dry-check CLI parsing:

```bash
uv run --python 3.14 easr --help
```

## Pull Request Checklist

- Keep changes focused on one behavior or documentation improvement.
- Add or update tests for code changes.
- Update `README.md`, `docs/development.md`, or `CHANGELOG.md` when user-facing behavior changes.
- Keep default output files stable: `.srt`, `.vtt`, `.json`, and optional `.metrics.json` with `--verbose`.
- Preserve the provider boundary; provider-specific details should not leak into the public CLI contract.
- Confirm generated files and local outputs are not included in the diff.

## Useful Areas for Contributions

- Documentation examples and troubleshooting notes.
- Small fixtures or synthetic media-free tests.
- Subtitle segmentation and timing quality checks.
- Better diagnostics for installation and runtime failures.
- Provider-boundary improvements that keep the CLI stable.

## Reporting Problems

Use the bug report template and include:

- macOS version and Apple Silicon chip.
- Python version.
- install method (`pip`, `uv`, source checkout).
- exact `easr` command.
- whether `ffmpeg`, `ffprobe`, and MLX preflight pass.
- a short, non-private sample or reproducible fixture when possible.
