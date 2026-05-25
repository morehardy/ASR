<p align="center">
  <img src="docs/assets/asr-logo.png" alt="echoalign-asr-mlx logo" width="720">
</p>

# echoalign-asr-mlx

`easr` is a local Apple Silicon CLI that turns audio and video files into
subtitle files (`.srt`, `.vtt`) and timestamp-aligned JSON.

Use it when you want local transcription, readable subtitles, and
machine-friendly alignment data without running a server.

Current scope:

- runtime target: macOS on Apple Silicon
- backend: MLX with Qwen3 ASR and Qwen3 ForcedAligner
- output: SRT, WebVTT, and JSON
- not included yet: translation, speaker diarization, Linux/Windows support

## What You Get

For each supported media file, `easr` writes:

- `<name>.srt` for subtitle players and editors
- `<name>.vtt` for web video workflows
- `<name>.json` for downstream tools that need segments, tokens, timestamps,
  language metadata, and provider metadata
- `<name>.metrics.json` when `--verbose` is enabled

`easr` accepts files, directories, and glob patterns. Directory scans are
non-recursive by default, and recursive processing is opt-in with `--recursive`.

## Requirements

- macOS on Apple Silicon
- Python `>=3.14,<3.15`
- `ffmpeg` and `ffprobe` available on `PATH`
- `uv` if you run from a source checkout
- network access on first run so the models can be downloaded from Hugging Face

Install the media tools with Homebrew:

```bash
brew install ffmpeg
```

Default provider models:

- [`mlx-community/Qwen3-ASR-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [`mlx-community/Qwen3-ForcedAligner-0.6B-bf16`](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)

## Installation

Install from a published Python package:

```bash
python3.14 -m pip install "echoalign-asr-mlx[mlx]"
easr --help
```

Run from a source checkout:

```bash
uv sync --extra mlx
uv run --python 3.14 --extra mlx easr --help
```

If you use the source checkout flow, prefix examples in this README with:

```bash
uv run --python 3.14 --extra mlx easr ...
```

## Quick Start

Transcribe one file:

```bash
easr ./demo.mp4
```

Write outputs to a custom directory:

```bash
easr ./demo.mp4 --output-dir ./subtitles
```

Process a directory:

```bash
easr ./media
```

Process a directory recursively:

```bash
easr ./media --recursive
```

Process a glob pattern:

```bash
easr "./media/**/*.mp4" --recursive
```

Export token-level subtitle and JSON views:

```bash
easr ./demo.mp4 --granularity token
```

Show detailed progress and write metrics:

```bash
easr ./demo.mp4 --verbose
```

## Supported Formats

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

## Output Layout

Default output directory name: `outputs`.

When the input is a single file, outputs are written next to that file:

```text
/project/media/demo.mp4
/project/media/outputs/demo.srt
/project/media/outputs/demo.vtt
/project/media/outputs/demo.json
```

When the input is a directory or the current directory, outputs are written
under that input root:

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

Use `--output-dir` to choose another output root.

## JSON Output

The JSON export keeps the readable transcript and the alignment data used to
create subtitle views.

Common top-level fields:

- `source_path`
- `provider_name`
- `detected_language`
- `segments`
- `source_media`
- `granularity`
- `items`

Each segment includes text, start/end timestamps, language metadata, optional
speaker metadata, and token timing when available. `source_media` includes the
prepared audio path, VAD metadata, and provider diagnostics such as processing
strategy, duration, window counts, quality pass counts, and window diagnostics.

## CLI Options

| Option | Meaning |
| --- | --- |
| `inputs` | File, directory, or glob pattern. Defaults to the current directory. |
| `--recursive` | Recursively scan directory inputs. |
| `--output-dir PATH` | Override the default output directory root. |
| `--granularity sentence` | Use segment boundaries for subtitle entries and JSON `items`. This is the default. |
| `--granularity token` | Use token timing for subtitle entries and JSON `items`. |
| `--no-vad` | Disable voice activity detection preprocessing. |
| `--verbose` | Print detailed progress and write `<name>.metrics.json`. |
| `--version` | Show the installed package version. |
| `--help` | Show CLI help. |

## VAD Preprocessing

Voice activity detection is enabled by default. `easr` scans the prepared audio,
finds likely speech ranges, groups them into padded chunks, and asks the
provider to process only those ranges. Final subtitle timestamps remain on the
original media timeline.

Disable VAD when you want full-duration provider processing:

```bash
easr ./demo.mp4 --no-vad
```

If VAD fails, `easr` falls back to full-duration processing. If VAD succeeds and
finds no speech, `easr` writes successful empty subtitle outputs.

## Shell Completion

Fish shell users can generate or install completions:

```bash
easr completion fish
easr completion install fish
```

The install command writes:

```text
~/.config/fish/completions/easr.fish
```

Existing completion files at that path are overwritten.

## Runtime Behavior

- exit code `0`: all discovered files processed successfully
- exit code `1`: no supported input was found, environment preflight failed, or
  at least one file failed in a batch
- batch files are processed one by one
- failures are reported per file to stderr
- other files continue processing after a per-file failure
- the first run may be slower because model files are downloaded and cached

## Troubleshooting

### Missing `ffmpeg` or `ffprobe`

Install the media tools and make sure they are visible from the shell running
`easr`:

```bash
brew install ffmpeg
which ffmpeg
which ffprobe
```

### MLX or Metal preflight failed

Check that you are on Apple Silicon, using the expected Python environment, and
installed the MLX runtime extra:

```bash
python3.14 -m pip install "echoalign-asr-mlx[mlx]"
```

For a source checkout:

```bash
uv sync --extra mlx
uv run --python 3.14 --extra mlx easr --help
```

### First run is slow

This is expected when the Qwen3 model files are downloaded and the local cache
is warmed. Later runs should be faster.

## Current Limitations

- Translation is not implemented.
- Speaker diarization is not implemented.
- Subtitle segmentation quality depends on model and alignment behavior.
- The public CLI does not expose provider selection.

## Developer Documentation

Development setup, test commands, build instructions, and release notes live in
[docs/development.md](docs/development.md).
