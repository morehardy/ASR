# ASR Command Completion Design

Date: `2026-04-11`

## Goal

Improve command completion support for the `asr` CLI with first-class `fish` support while preserving existing transcription behavior.

## Confirmed Decisions

- Shell support required in this scope: `fish`
- Provide both:
  - `asr completion fish` (print script to stdout)
  - `asr completion install fish` (install script automatically)
- Install target path: `~/.config/fish/completions/asr.fish`
- Install behavior when target exists: overwrite by default
- `install` must **not** modify `~/.config/fish/config.fish`
- Migration approach selected by user: migrate CLI from `argparse` to `Typer`

## Context

Current CLI is implemented with `argparse` in `src/asr/cli.py` and has no command completion feature.
The existing public command contract (`asr <inputs> [flags]`) is stable and should remain compatible.

## Approaches Considered

### 1. Keep `argparse` + build completion generator in-house

Pros:
- smallest code delta
- low migration risk

Cons:
- custom completion generation maintenance burden
- less ergonomic CLI extension over time

### 2. Keep `argparse` + static fish completion template

Pros:
- fastest initial implementation

Cons:
- template can drift from real CLI options
- ongoing manual sync cost

### 3. Migrate to `Typer` and use framework completion support (**Selected**)

Pros:
- built-in completion workflows
- better long-term maintainability for subcommands and help UX

Cons:
- moderate migration risk due to parser/runtime entry changes

Rationale:
User explicitly chose option 3 to prioritize long-term CLI evolution over minimal short-term edits.

## CLI Contract (Post-Change)

### Existing root command behavior (must remain equivalent)

- `asr [inputs ...] [--recursive] [--output-dir PATH] [--granularity sentence|token] [--verbose]`
- If no input is provided, defaults to current directory
- Discovery, preflight, provider execution, and export behavior remain unchanged
- Exit code semantics remain unchanged:
  - `0`: all discovered files processed successfully
  - `1`: no files found, preflight failure, or per-file processing failure

### New completion commands

- `asr completion fish`
  - prints fish completion script to stdout
  - no preflight/provider/model activity
  - exits `0` on success

- `asr completion install fish`
  - writes completion script to `~/.config/fish/completions/asr.fish`
  - creates parent directory when missing
  - overwrites existing file
  - prints installed path on success
  - exits `0` on success
  - exits `1` with readable error on IO/permission failure

## Architecture

## Command Layer

- Replace `argparse` parser with `Typer` app in `src/asr/cli.py`
- Keep root command for transcription workflow
- Add `completion` command group with `fish` and `install fish` commands

## Workflow Isolation

- Transcription workflow and completion workflow are isolated entry paths
- Completion commands must not trigger:
  - media discovery for transcription
  - environment preflight
  - provider construction/model loading

## Installation Responsibility

- Installer only manages fish completion file placement
- No shell startup file mutation
- No cross-shell installation in this scope

## Error Handling

- Argument and shell validation errors use Typer/Click default CLI error behavior (`2`)
- File write errors in `install` return `1` and emit actionable stderr message
- Root transcription errors preserve current behavior and messaging style

## Testing Strategy

## Migration Regression Tests

- Preserve root command default behavior (no args -> current directory)
- Preserve flags and parsing semantics (`--recursive`, `--output-dir`, `--granularity`, `--verbose`)
- Preserve preflight failure surface and exit code behavior

## Completion Tests

- `asr completion fish` emits non-empty script containing `asr` command references and key options
- `asr completion install fish` writes expected file under temporary `HOME`
- verify overwrite behavior when completion file already exists
- verify install error path on unwritable destination

## Documentation Updates

- Update `README.md` with:
  - `asr completion fish` usage
  - `asr completion install fish` usage
  - expected install location for fish

## Risks and Mitigations

- Risk: subtle parsing differences after `argparse` -> `Typer` migration
  - Mitigation: keep explicit tests for current CLI contract and exit codes

- Risk: completion script shape changes across Typer/Click versions
  - Mitigation: assert semantic invariants in tests (contains key command/option markers), avoid overfitting exact script bytes

- Risk: user environment permissions prevent install
  - Mitigation: clear stderr messaging with target path and failure reason

## Acceptance Criteria

- `asr --help` remains usable and coherent
- Existing root command usage works as before
- `asr completion fish` prints valid fish completion content
- `asr completion install fish` installs to `~/.config/fish/completions/asr.fish`
- Existing file is overwritten on install
- Project tests pass with updated/added CLI tests

## Out of Scope

- `bash`/`zsh` completion support
- auto-editing shell startup files
- changing ASR pipeline/provider/export behavior
- refactoring unrelated modules

## Review Notes

Spec self-check completed:

- no unresolved placeholders remain
- no internal contradictions found between command contract and testing
- scope is constrained to completion + CLI migration
- overwrite and install-path semantics are explicit
