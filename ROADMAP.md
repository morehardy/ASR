# Roadmap

This roadmap is directional, not a release commitment. The project should keep its default local CLI simple while improving transcription quality, contribution readiness, and backend flexibility.

## Current Focus

- Make the project easier to discover on GitHub and PyPI.
- Improve contributor entry points and issue reporting.
- Keep the Apple Silicon + MLX runtime stable.
- Preserve the existing output contract: `.srt`, `.vtt`, `.json`, and optional `.metrics.json`.

## Near-Term

- Add small, safe sample fixtures that do not require large media files.
- Expand troubleshooting docs for MLX, Metal, model downloads, and `ffmpeg` setup.
- Document the JSON output contract with field-level examples.
- Add release-note discipline through `CHANGELOG.md`.

## Product Direction

- Improve subtitle segmentation and timing repair.
- Add richer quality diagnostics for failed or low-confidence windows.
- Add plain text export if it can be done without weakening the timestamped JSON contract.
- Expand provider support behind the existing provider boundary.
- Explore speaker metadata and diarization when a provider can support it reliably.

## Out of Scope for Now

- Hosted API or service deployment.
- Linux and Windows runtime support.
- Public provider selection flags before the provider contract is ready for that UX.
- Translation output as part of the default transcription flow.
