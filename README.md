# asr

`asr` is a local CLI for extracting subtitles and aligned timestamps from audio/video files.

It currently targets `macOS + Apple Silicon` and uses an MLX-based provider (`Qwen3-ASR + Qwen3-ForcedAligner`) behind a stable CLI interface.

## Current Status

The project is implemented and runnable.

Current capabilities:

- local file and directory processing
- audio/video normalization to mono 16 kHz WAV
- sentence-level and token-level export views
- subtitle export to `srt` and `vtt`
- rich JSON export (segments, tokens, provider metadata)
- windowed transcription/alignment pipeline with diagnostics

## Environment Requirements

- OS: `macOS` on `Apple Silicon`
- Python: `>=3.14,<3.15`
- environment/tooling: `uv`
- system dependencies on `PATH`:
  - `ffmpeg`
  - `ffprobe`
- network access on first run (model download from Hugging Face)

Default provider models:

- [`mlx-community/Qwen3-ASR-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [`mlx-community/Qwen3-ForcedAligner-0.6B-bf16`](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)

## Installation

1. Install `uv` (see [uv docs](https://docs.astral.sh/uv/)).
2. Install `ffmpeg` and `ffprobe` with your package manager.
3. Sync dependencies in project root.

```bash
uv sync --extra mlx
```

4. Verify CLI is available.

```bash
uv run --python 3.14 asr --help
```

## Quick Start

### Single file

```bash
uv run --python 3.14 --extra mlx asr ./demo.mp4 --verbose
```

### No input path (defaults to current directory)

```bash
uv run --python 3.14 --extra mlx asr
```

### Directory (non-recursive by default)

```bash
uv run --python 3.14 --extra mlx asr ./media
```

### Directory (recursive)

```bash
uv run --python 3.14 --extra mlx asr ./media --recursive --verbose
```

### Glob pattern input

```bash
uv run --python 3.14 --extra mlx asr "./media/**/*.mp4" --recursive --verbose
```

## CLI Reference

Help output:

```text
usage: asr [-h] [--recursive] [--output-dir OUTPUT_DIR]
           [--granularity {sentence,token}] [--verbose]
           [inputs ...]
```

Arguments and flags:

- `inputs`: file, directory, or glob pattern; defaults to current directory if omitted
- `--recursive`: recurse when scanning directory inputs
- `--output-dir`: override default output root
- `--granularity {sentence,token}`:
  - `sentence`: subtitle entries come from segment boundaries
  - `token`: subtitle/JSON `items` are generated from tokens
- `--verbose`: print detailed per-step timing and export `<name>.metrics.json`

## Shell Completion (fish)

Generate fish completion script:

```bash
asr completion fish
```

Install fish completion script:

```bash
asr completion install fish
```

Install target path:

- `~/.config/fish/completions/asr.fish`
- existing file is overwritten on install

## Supported Input Formats

Audio:

- `wav`
- `mp3`
- `m4a`
- `flac`
- `aac`

Video:

- `mp4`
- `mov`
- `m4v`
- `mkv`
- `webm`

## Output Files and Layout

For each input media file, the CLI writes:

- `<name>.srt`
- `<name>.vtt`
- `<name>.json`
- `<name>.metrics.json` (only when `--verbose` is enabled)

Default output directory name: `outputs`

Layout rules:

- input is a directory/current directory:
  - output root defaults to `<input_root>/outputs`
- input is a single file:
  - output root defaults to `<file_parent>/outputs`
- recursive batch keeps relative directory structure
- `--output-dir` overrides output root

Example:

```text
/project/media/
  a.mp4
  nested/b.wav

/project/media/outputs/
  a.srt
  a.vtt
  a.json
  nested/b.srt
  nested/b.vtt
  nested/b.json
```

## JSON Contract (Practical)

`json` output includes:

- top-level transcription document:
  - `source_path`
  - `provider_name`
  - `detected_language`
  - `segments`
  - `source_media`
- export view:
  - `granularity`
  - `items`

`source_media` currently includes:

- `prepared_audio_path`
- `provider_metadata`:
  - `processing_strategy`
  - `window_count`
  - `duration_sec`
  - `quality_pass_count`
  - `failed_window_count`
  - `window_diagnostics`

## Real E2E Example (Current Repo)

Command used in this repository:

```bash
uv run --python 3.14 --extra mlx asr tests/e2e/test1.mov --verbose
```

Observed output files:

- `tests/e2e/outputs/test1.srt`
- `tests/e2e/outputs/test1.vtt`
- `tests/e2e/outputs/test1.json`

Observed sample stats from `test1.json`:

- `window_count`: `3`
- `failed_window_count`: `0`
- `segments`: ~`49` (depends on current alignment behavior/model output)

Sample subtitle excerpt:

```text
00:00:09,600 --> 00:00:17,640
But despite all the buzz and hype, one of the things that's still underestimated by many people is their power as a developer too.
```

## Common Commands

### Run unit tests

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest discover -s tests -p 'test_*.py'
```

### Run a single focused test

```bash
PYTHONPATH=src uv run --python 3.14 python -m unittest tests.test_authority
```

### Dry-check CLI parsing/help

```bash
uv run --python 3.14 asr --help
```

### Token-level subtitle export

```bash
uv run --python 3.14 --extra mlx asr ./demo.mp4 --granularity token --verbose
```

### Export verbose metrics JSON for optimization

```bash
uv run --python 3.14 --extra mlx asr ./demo.mp4 --verbose
```

## Runtime Flow (What Happens Internally)

For each media file:

1. CLI discovers supported inputs (file/dir/glob).
2. Environment preflight validates:
   - `ffmpeg` and `ffprobe` availability
   - MLX/Metal basic runtime check
   - progress starts rendering in terminal (single-line by default)
3. Media is normalized to mono 16 kHz WAV.
4. Provider runs windowed ASR + alignment.
5. Results are exported to `srt`, `vtt`, and `json`.
6. When `--verbose` is enabled, `<name>.metrics.json` is also exported.

## Exit Codes and Runtime Behavior

- `0`: all discovered files processed successfully
- `1`: no supported input found, preflight failed, or at least one file failed in batch

Batch behavior:

- files are processed one-by-one
- failures are reported per file to stderr
- other files continue processing

## Troubleshooting

### `[asr] environment check failed: Missing required media dependency...`

Cause: `ffmpeg` and/or `ffprobe` not found on `PATH`.

Fix:

- install both binaries
- ensure they are visible in the shell used to run `asr`

### `[asr] environment check failed: MLX/Metal preflight failed...`

Cause: MLX runtime could not initialize Metal backend (or crashed during preflight).

Fix:

- verify Apple Silicon + supported macOS runtime
- re-check Python/venv and `mlx` dependency installation
- retry from a clean shell/session

### First run is slow

Cause: model download and cache warm-up.

Fix:

- expected on first run
- later runs should be faster after cache is populated

## Notes and Scope

- translation is out of scope in current phase
- speaker diarization is not implemented
- subtitle segmentation quality depends on model + alignment behavior
- provider abstraction is in place for future backend extension
