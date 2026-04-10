# asr

`asr` is a planned local CLI tool for extracting transcripts, subtitles, and aligned timestamps from audio and video.

The current design targets a provider-based architecture so the backend can be replaced later without changing the overall user experience or exported semantics.

## Current Status

This repository is currently in the planning stage. The content below records the intended installation flow, command naming, input/output behavior, and example usage.

## Planned Features

- local CLI workflow
- audio and video input support
- automatic language detection
- multilingual processing with especially strong Chinese and English expectations
- subtitle export
- aligned timestamps
- provider abstraction for future backend replacement or expansion

## Planned Environment

- Platform target: `macOS + Apple Silicon`
- Python target: latest stable Python version available at implementation time
- Environment management: `uv`
- System dependencies: `ffmpeg`, `ffprobe`

## Planned Provider

The initial planned provider uses:

- [`mlx-community/Qwen3-ASR-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
- [`mlx-community/Qwen3-ForcedAligner-0.6B-bf16`](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)

Planned model behavior:

- models are downloaded and cached automatically on first run
- the aligner is treated as part of the provider internals
- users do not separately choose ASR and aligner models from the CLI

## Planned Installation

The exact commands may change after the repository is scaffolded, but the intended setup is:

1. Install the latest stable Python version supported by `uv`.
2. Install `uv`.
3. Install `ffmpeg` and `ffprobe`.
4. Create or sync the project environment with `uv`.
5. Run the `asr` CLI against local media files.

Representative shape only:

```bash
uv sync
asr ./demo.mp4
```

## Planned CLI Usage

Public command name:

```bash
asr
```

Planned usage examples:

```bash
# No input path: use the current working directory
asr

# Single file
asr ./demo.mp4

# Directory input, non-recursive by default
asr ./media

# Recursive directory scan
asr ./media --recursive

# Custom output directory
asr ./media --output-dir /tmp/asr-outputs

# Prefer sentence-like output
asr ./media --granularity sentence

# Request token-level output and verbose logs
asr ./media --granularity token --verbose
```

Planned CLI behavior:

- if no path is provided, use the current working directory
- directory scanning is non-recursive by default
- recursion is opt-in
- outputs are overwritten by default
- provider choice is not exposed in phase 1

## Planned Supported Input Formats

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

## Planned Output Formats

The default planned outputs are:

- `srt`
- `vtt`
- `json`

Planned semantics:

- `srt` and `vtt` are for subtitle consumption
- `json` keeps the richer alignment data for downstream processing

Internal timing target:

- English word-level timing
- Chinese character-level timing

Planned public granularity options:

- `sentence`
- `token`

`token` means:

- English: word
- Chinese: character

## Planned Output Layout

Default output directory name:

```text
outputs
```

Single-file example:

```text
/project/demo.mp4
/project/outputs/demo.srt
/project/outputs/demo.vtt
/project/outputs/demo.json
```

Directory example with recursive processing:

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

Rules:

- for directory or current-directory input, the default output root is `outputs/` under the input root
- for single-file input, the default output root is `outputs/` under the file's parent directory
- batch output preserves relative subdirectory structure
- `--output-dir` overrides the default location

## Notes

- Phase 1 is transcription and alignment only.
- Translation is out of scope for now.
- Speaker diarization is only a reserved future extension and depends on provider capability.
- Subtitle segmentation quality depends partly on model capability.
