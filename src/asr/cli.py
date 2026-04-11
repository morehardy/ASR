"""Command-line interface for the ASR tool."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import click
import typer

from asr.discovery import discover_media_files
from asr.exporters import render_json, render_srt, render_vtt
from asr.media import FfmpegMediaPreparer
from asr.output import build_output_path, default_output_root
from asr.pipeline import process_media_file
from asr.providers import create_default_provider

_MLX_PREFLIGHT_CODE = (
    "import mlx.core as mx\n"
    "_ = mx.array([0], dtype=mx.int32)\n"
)

app = typer.Typer(
    name="asr",
    help="Extract subtitles and aligned timestamps from local audio and video.",
    add_completion=False,
)


def build_parser() -> argparse.ArgumentParser:
    """Backward-compatible parser builder retained for tests and integrations."""

    parser = argparse.ArgumentParser(
        prog="asr",
        description="Extract subtitles and aligned timestamps from local audio and video.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="File, directory, or glob pattern. Defaults to the current directory.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan directory inputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the default output directory root.",
    )
    parser.add_argument(
        "--granularity",
        choices=("sentence", "token"),
        default="sentence",
        help="Subtitle and JSON view granularity.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress information.",
    )
    return parser


def resolve_cli_inputs(inputs: Sequence[str]) -> List[Path]:
    """Resolve CLI inputs into concrete filesystem paths."""

    if not inputs:
        return [Path.cwd()]

    resolved: List[Path] = []
    for value in inputs:
        if any(char in value for char in "*?[]"):
            matches = sorted(glob.glob(value, recursive=True))
            resolved.extend(Path(match) for match in matches)
            continue
        resolved.append(Path(value))
    return resolved


def discover_cli_sources(inputs: Sequence[str], recursive: bool) -> List[Tuple[Path, Path]]:
    """Discover concrete media files and their output roots."""

    discovered: List[Tuple[Path, Path]] = []
    for raw_input in resolve_cli_inputs(inputs):
        path = raw_input.expanduser()
        if path.is_file():
            files = discover_media_files(path, recursive=False)
            discovered.extend((source, path.parent) for source in files)
            continue
        files = discover_media_files(path, recursive=recursive)
        discovered.extend((source, path) for source in files)
    return discovered


def _first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def run_environment_preflight() -> Tuple[bool, str]:
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    missing: List[str] = []
    if ffmpeg_path is None:
        missing.append("ffmpeg")
    if ffprobe_path is None:
        missing.append("ffprobe")
    if missing:
        return (
            False,
            f"Missing required media dependency: {', '.join(missing)} not found on PATH.",
        )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _MLX_PREFLIGHT_CODE],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return False, f"Unable to run MLX/Metal preflight: {exc}"

    if proc.returncode == 0:
        return True, ""

    detail = _first_non_empty_line(proc.stderr) or _first_non_empty_line(proc.stdout)
    if proc.returncode < 0:
        reason = f"process terminated by signal {-proc.returncode}"
    else:
        reason = f"exit code {proc.returncode}"
    if detail:
        return False, f"MLX/Metal preflight failed ({reason}): {detail}"
    return False, f"MLX/Metal preflight failed ({reason})."


def _validate_granularity(value: str) -> str:
    if value not in ("sentence", "token"):
        raise typer.BadParameter("Granularity must be one of: sentence, token.")
    return value


def build_fish_completion_script() -> str:
    env = dict(os.environ)
    env["_ASR_COMPLETE"] = "fish_source"
    proc = subprocess.run(
        [sys.executable, "-m", "asr"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        detail = _first_non_empty_line(proc.stderr) or _first_non_empty_line(proc.stdout)
        raise RuntimeError(detail or f"completion process failed with exit code {proc.returncode}")
    if not proc.stdout.strip():
        raise RuntimeError("completion script output was empty")
    return proc.stdout


def run_completion_fish() -> int:
    try:
        script = build_fish_completion_script()
    except RuntimeError as exc:
        print(f"[asr] completion generation failed: {exc}", file=sys.stderr)
        return 1
    print(script, end="")
    return 0


def fish_completion_target(home: Path | None = None) -> Path:
    base = home if home is not None else Path.home()
    return base / ".config" / "fish" / "completions" / "asr.fish"


def install_fish_completion(script: str, home: Path | None = None) -> Path:
    target = fish_completion_target(home=home)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(script, encoding="utf-8")
    return target


def run_completion_install_fish() -> int:
    try:
        script = build_fish_completion_script()
        target = install_fish_completion(script)
    except RuntimeError as exc:
        print(f"[asr] completion generation failed: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"[asr] completion install failed: {exc}", file=sys.stderr)
        return 1
    print(f"[asr] fish completion installed at {target}")
    return 0


def _run_transcription(
    inputs: Sequence[str],
    recursive: bool,
    output_dir: Path | None,
    granularity: str,
    verbose: bool,
) -> int:
    discovered_sources = discover_cli_sources(inputs, recursive=recursive)
    if not discovered_sources:
        print("No supported media files found.", file=sys.stderr)
        return 1

    ok, message = run_environment_preflight()
    if not ok:
        print(f"[asr] environment check failed: {message}", file=sys.stderr)
        return 1

    provider = create_default_provider()
    media_preparer = FfmpegMediaPreparer()
    had_error = False

    for source_path, input_root in discovered_sources:
        output_root = default_output_root(input_root, explicit_output_dir=output_dir)
        try:
            document = process_media_file(
                source_path=source_path,
                provider=provider,
                media_preparer=media_preparer,
            )
            rendered_outputs = {
                ".srt": render_srt(document, granularity=granularity),
                ".vtt": render_vtt(document, granularity=granularity),
                ".json": render_json(document, granularity=granularity),
            }
            for suffix, content in rendered_outputs.items():
                target = build_output_path(
                    source=source_path,
                    input_root=input_root,
                    output_root=output_root,
                    suffix=suffix,
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            if verbose:
                print(f"[asr] processed {source_path} -> {output_root}")
            else:
                print(source_path)
        except Exception as exc:  # pragma: no cover - surfaced to CLI output
            had_error = True
            print(f"[asr] failed for {source_path}: {exc}", file=sys.stderr)

    return 1 if had_error else 0


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    inputs: List[str] = typer.Argument(
        None,
        help="File, directory, or glob pattern. Defaults to the current directory.",
    ),
    recursive: bool = typer.Option(False, "--recursive", help="Recursively scan directory inputs."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Override the default output directory root."),
    granularity: str = typer.Option(
        "sentence",
        "--granularity",
        callback=_validate_granularity,
        help="Subtitle and JSON view granularity.",
    ),
    verbose: bool = typer.Option(False, "--verbose", help="Print detailed progress information."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return

    code = _run_transcription(
        inputs=inputs or [],
        recursive=recursive,
        output_dir=output_dir,
        granularity=granularity,
        verbose=verbose,
    )
    raise typer.Exit(code=code)


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else sys.argv[1:]
    if args and args[0] == "completion":
        if args == ["completion", "fish"]:
            return run_completion_fish()
        if args == ["completion", "install", "fish"]:
            return run_completion_install_fish()
        print("Usage: asr completion fish | asr completion install fish", file=sys.stderr)
        return 2
    try:
        result = app(args=args, prog_name="asr", standalone_mode=False)
    except typer.Exit as exc:
        return int(exc.exit_code)
    except click.ClickException as exc:
        exc.show(file=sys.stderr)
        return int(exc.exit_code)
    if isinstance(result, int):
        return result
    return 0
