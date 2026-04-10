# ASR CLI Design

Date: `2026-04-10`

## Goal

Define the initial product and implementation constraints for a local ASR CLI project that:

- accepts audio and video files
- runs locally on `macOS + Apple Silicon`
- exports subtitles and aligned timestamps
- keeps the provider boundary stable for future backend replacement

## Confirmed Decisions

- Public product shape: local CLI
- Public command name: `asr`
- Runtime target: `macOS + Apple Silicon`
- Environment manager: `uv`
- Python target: latest stable version available at implementation time
- System dependency: `ffmpeg` / `ffprobe`
- Default phase 1 provider:
  - `mlx-community/Qwen3-ASR-1.7B-bf16`
  - `mlx-community/Qwen3-ForcedAligner-0.6B-bf16`
- Model acquisition: automatic download and cache on first run
- No public `--provider` flag in phase 1
- The aligner is a provider-internal concern and is not selected independently

## CLI Contract

- If no input path is given, process the current working directory.
- Accept:
  - single files
  - directories
  - wildcard / pattern-based batch input
- Directory scanning is non-recursive by default.
- Recursive scanning is opt-in.
- Logging is concise by default, with a verbose mode.
- Default behavior overwrites existing outputs.

## Input / Output Contract

Supported formats:

- Audio: `wav`, `mp3`, `m4a`, `flac`, `aac`
- Video: `mp4`, `mov`, `m4v`, `mkv`, `webm`

Default outputs:

- `srt`
- `vtt`
- `json`

Default output layout:

- use `outputs/` as the directory name
- single file: sibling `outputs/` under the file parent
- directory input: `<input_root>/outputs/`
- preserve relative subdirectory structure for batch outputs
- allow override with `--output-dir`

## Timestamp Model

Internal canonical timing must preserve:

- English word-level timing
- Chinese character-level timing

Public outputs should support:

- sentence-level rendering
- token-level rendering

Where token-level means:

- English token = word
- Chinese token = character

Default readable subtitle output should prefer sentence-like segments, but actual quality depends on the provider and model capability.

## Provider Boundary

Provider is the main backend abstraction.

Rules:

- transcription and alignment belong to one provider implementation
- provider internals must not leak into the public CLI contract
- future providers must emit the same canonical intermediate representation
- exporter behavior should not depend on which provider produced the data

## Deferred Scope

Not required in phase 1:

- translation
- mandatory speaker diarization
- Linux / Windows runtime
- HTTP API surface
- distributed or service orchestration

## Architectural Guidance

The implementation should separate:

- CLI
- media discovery and extraction
- provider execution
- canonical transcript model
- exporters

The system should start with synchronous local execution, while preserving room for future batch concurrency and additional providers.

## Review Notes

Self-check completed:

- no unresolved placeholders
- no internal contradictions found
- scope is appropriate for an initial implementation plan
- provider boundary and output semantics are explicit
