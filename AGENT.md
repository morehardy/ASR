# AGENT.md

## Project Intent

This project is a local CLI tool, managed with `uv`, for analyzing audio and video with an ASR provider and exporting subtitles plus aligned timestamps.

The current first-phase goal is:

- Accept local audio/video inputs.
- Transcribe source speech in the original language.
- Export subtitle files and aligned timing data.
- Keep the provider boundary stable so the backend can be replaced or extended later.

## Current Scope

- Phase 1 runtime target: `macOS + Apple Silicon`
- Python target: the latest stable Python version available at implementation time
- Environment management: `uv`
- System dependency: `ffmpeg` / `ffprobe`
- Phase 1 default provider is planned around:
  - [`mlx-community/Qwen3-ASR-1.7B-bf16`](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16)
  - [`mlx-community/Qwen3-ForcedAligner-0.6B-bf16`](https://huggingface.co/mlx-community/Qwen3-ForcedAligner-0.6B-bf16)
- Model download behavior: automatically download and cache on first run

## Non-Goals For Phase 1

- No translation output
- No required speaker diarization output
- No Linux or Windows runtime target yet
- No HTTP API or service deployment target yet
- No public provider selection in the CLI yet

## CLI Contract

- Public command name: `asr`
- Phase 1 user experience should feel like a single-command CLI.
- If no input path is provided, the CLI must default to the current working directory.
- The CLI must support:
  - single file input
  - directory input
  - wildcard / pattern-based batch input
- Directory scanning must be non-recursive by default.
- Recursion must be opt-in through a CLI flag such as `--recursive`.
- Logging must be concise by default and support a verbose mode such as `--verbose`.
- The phase 1 CLI must not expose a `--provider` flag.
- The aligner model must not be configured separately from the ASR model at the CLI layer.

## Supported Media Inputs

Phase 1 should explicitly support these formats:

- Audio: `wav`, `mp3`, `m4a`, `flac`, `aac`
- Video: `mp4`, `mov`, `m4v`, `mkv`, `webm`

## Output Rules

- Default output formats: `srt`, `vtt`, `json`, `txt`(txt do not have time stamp )
- Default overwrite behavior: overwrite existing outputs
- Default output directory name: `outputs`
- The CLI must support overriding the output directory through a flag such as `--output-dir`

Default output layout:

- Single file input:
  - write outputs to a sibling `outputs/` directory under the file's parent directory
- Directory input or current-directory input:
  - write outputs under `<input_root>/outputs/`
  - preserve the original relative subdirectory structure for batch output files

Examples:

- `/path/media/demo.mp4` -> `/path/media/outputs/demo.srt`
- `/path/media/demo.mp4` -> `/path/media/outputs/demo.vtt`
- `/path/media/demo.mp4` -> `/path/media/outputs/demo.txt`
- `/path/media/demo.mp4` -> `/path/media/outputs/demo.json`
- `/path/media/sub/a.wav` with input root `/path/media` -> `/path/media/outputs/sub/a.srt`

## Timestamp Semantics

The internal canonical representation must preserve fine-grained timing:

- English: word-level timing
- Chinese: character-level timing

Public outputs must be able to expose coarser or alternate granularities without losing the internal source data:

- sentence-level
- token-level

Token-level semantics must be consistent:

- English token = word
- Chinese token = character

Readable subtitle exports should default to sentence-like segments, but this behavior is constrained by actual provider/model capability. Do not over-promise segmentation quality beyond what the provider can reliably produce.

## Language Behavior

- Language handling must default to automatic detection.
- The system should be designed for multilingual input, with especially strong support expectations for Chinese and English.
- The canonical model should allow language metadata at least at the segment level, and ideally at the token level when available.

## Provider Architecture

Provider is the primary backend abstraction boundary in this project.

Rules:

- A provider owns transcription and alignment as one cohesive implementation.
- The forced aligner is part of the provider internals, not a separately selected public dependency.
- The provider must emit the same canonical intermediate representation regardless of backend implementation.
- Provider-specific quirks must not leak into the public CLI contract or exported file layout.
- Adding a new provider later must not require redesigning subtitle exporters or the canonical data model.

Recommended provider responsibilities:

- media preparation required by the backend
- language detection when supported
- transcription
- forced alignment
- capability reporting

Recommended provider capability surface:

- supports_alignment
- supports_language_detection
- supports_diarization

These capabilities are internal planning aids. They must not turn the public interface into a provider-specific UX.

## Canonical Data Model

Implementation should converge on a stable backend-neutral representation. The exact class layout can evolve, but the semantics should stay fixed.

Recommended layers:

- `TranscriptionDocument`
  - source path
  - source media metadata
  - provider metadata
  - detected language metadata
  - segments
- `Segment`
  - id
  - text
  - start time
  - end time
  - language
  - tokens
  - optional speaker field
- `Token`
  - text
  - start time
  - end time
  - unit type: `word` or `char`
  - language

Exporters should transform this canonical representation into:

- readable subtitle files
- machine-friendly JSON
- alternate public granularities

The JSON output should retain the fine-grained alignment data and must not be reduced to subtitle-level timing only.

## Speaker Diarization Position

- The data model should reserve room for speaker metadata.
- Phase 1 does not require speaker diarization to be implemented.
- Any future diarization support must be optional and capability-driven.
- Availability depends on provider/model capability.

## Execution Model

- Phase 1 should prioritize synchronous local execution.
- Internal boundaries should still allow future expansion into batch orchestration or concurrent processing.
- Future concurrency work must not force a redesign of the canonical model or provider interface.

## Engineering Guidance

Prefer a clean separation of concerns:

- CLI parsing
- media discovery and normalization
- media decoding / extraction
- provider execution
- canonical data model
- exporters

When in doubt:

- keep backend logic behind the provider boundary
- keep public behavior backend-neutral
- keep exported semantics stable

## Change Safety Rules

Future changes must preserve these invariants:

- Do not expose provider-specific model pairings as separate public configuration unless the entire public contract is intentionally redesigned.
- Do not let exported JSON lose fine-grained token alignment.
- Do not let a new provider silently change default output layout.
- Do not let a new provider silently change token semantics.
- Do not let future feature work break the default `asr` CLI flow.

If a future provider cannot support the same alignment depth or language behavior, degrade explicitly and document the limitation rather than changing the canonical contract implicitly.
