"""Command-line interface for the ASR tool."""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

import click
import typer

from asr.discovery import discover_media_files
from asr.exporters import render_json, render_srt, render_vtt
from asr.media import FfmpegMediaPreparer
from asr.observability.console import ConsoleProgressObserver
from asr.observability.events import ObservabilityEvent
from asr.observability.metrics import MetricsCollectorObserver
from asr.observability.observer import ObserverMux
from asr.observability.timing import observe_step
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


def build_fish_completion_script() -> str:
    env = dict(os.environ)
    env["_ASR_COMPLETE"] = "source_fish"
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
    run_id = f"run-{uuid.uuid4().hex[:8]}"
    collector = MetricsCollectorObserver() if verbose else None
    observers = [ConsoleProgressObserver()]
    if collector is not None:
        observers.append(collector)
    observer = ObserverMux(
        observers=observers,
        warning_sink=lambda message: print(message, file=sys.stderr),
    )
    observer.on_event(ObservabilityEvent(event_type="run_start", run_id=run_id))

    try:
        discovered_sources = discover_cli_sources(inputs, recursive=recursive)
        if not discovered_sources:
            print("No supported media files found.", file=sys.stderr)
            return 1

        with observe_step(
            observer,
            run_id=run_id,
            file_id=None,
            source_path=None,
            step="preflight",
        ):
            ok, message = run_environment_preflight()
        if not ok:
            print(f"[asr] environment check failed: {message}", file=sys.stderr)
            return 1

        provider = create_default_provider()
        media_preparer = FfmpegMediaPreparer()
        had_error = False

        for index, (source_path, input_root) in enumerate(discovered_sources, start=1):
            file_id = str(index)
            observer.on_event(
                ObservabilityEvent(
                    event_type="file_start",
                    run_id=run_id,
                    file_id=file_id,
                    source_path=str(source_path),
                    meta={"index": index, "total": len(discovered_sources)},
                )
            )
            output_root = default_output_root(input_root, explicit_output_dir=output_dir)
            try:
                if hasattr(provider, "bind_observer"):
                    provider.bind_observer(
                        observer=observer,
                        run_id=run_id,
                        file_id=file_id,
                        source_path=str(source_path),
                    )
                try:
                    document = process_media_file(
                        source_path=source_path,
                        provider=provider,
                        media_preparer=media_preparer,
                        observer=observer,
                        run_id=run_id,
                        file_id=file_id,
                    )
                finally:
                    if hasattr(provider, "clear_observer"):
                        provider.clear_observer()
                with observe_step(
                    observer,
                    run_id=run_id,
                    file_id=file_id,
                    source_path=str(source_path),
                    step="render_srt",
                ):
                    srt_content = render_srt(document, granularity=granularity)
                with observe_step(
                    observer,
                    run_id=run_id,
                    file_id=file_id,
                    source_path=str(source_path),
                    step="render_vtt",
                ):
                    vtt_content = render_vtt(document, granularity=granularity)
                with observe_step(
                    observer,
                    run_id=run_id,
                    file_id=file_id,
                    source_path=str(source_path),
                    step="render_json",
                ):
                    json_content = render_json(document, granularity=granularity)

                with observe_step(
                    observer,
                    run_id=run_id,
                    file_id=file_id,
                    source_path=str(source_path),
                    step="write_outputs",
                ):
                    rendered_outputs = {
                        ".srt": srt_content,
                        ".vtt": vtt_content,
                        ".json": json_content,
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

                observer.on_event(
                    ObservabilityEvent(
                        event_type="file_end",
                        run_id=run_id,
                        file_id=file_id,
                        source_path=str(source_path),
                        meta={"status": "ok"},
                    )
                )
                if collector is not None:
                    _write_metrics_json(
                        collector=collector,
                        file_id=file_id,
                        source_path=source_path,
                        input_root=input_root,
                        output_root=output_root,
                    )
                if verbose:
                    print(f"[asr] processed {source_path} -> {output_root}")
                else:
                    print(source_path)
            except Exception as exc:  # pragma: no cover - surfaced to CLI output
                had_error = True
                observer.on_event(
                    ObservabilityEvent(
                        event_type="file_end",
                        run_id=run_id,
                        file_id=file_id,
                        source_path=str(source_path),
                        meta={"status": "failed", "error": str(exc)},
                    )
                )
                if collector is not None:
                    _write_metrics_json(
                        collector=collector,
                        file_id=file_id,
                        source_path=source_path,
                        input_root=input_root,
                        output_root=output_root,
                    )
                print(f"[asr] failed for {source_path}: {exc}", file=sys.stderr)

        return 1 if had_error else 0
    finally:
        observer.on_event(ObservabilityEvent(event_type="run_end", run_id=run_id))
        observer.close()


def _write_metrics_json(
    *,
    collector: MetricsCollectorObserver,
    file_id: str,
    source_path: Path,
    input_root: Path,
    output_root: Path,
) -> None:
    try:
        target = build_output_path(
            source=source_path,
            input_root=input_root,
            output_root=output_root,
            suffix=".metrics.json",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        collector.write_file_metrics(file_id=file_id, target_path=target)
    except Exception as exc:  # pragma: no cover - observability should not break main flow
        print(
            f"[asr] warning: failed to write metrics for {source_path}: {exc}",
            file=sys.stderr,
        )


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    inputs: List[str] = typer.Argument(
        None,
        help="File, directory, or glob pattern. Defaults to the current directory.",
    ),
    recursive: bool = typer.Option(False, "--recursive", help="Recursively scan directory inputs."),
    output_dir: Path | None = typer.Option(None, "--output-dir", help="Override the default output directory root."),
    granularity: Literal["sentence", "token"] = typer.Option(
        "sentence",
        "--granularity",
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
    completion_exit_code = _dispatch_completion(args)
    if completion_exit_code is not None:
        return completion_exit_code
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


def _dispatch_completion(args: Sequence[str]) -> int | None:
    positional_index = _first_positional_index(args)
    if positional_index is None:
        return None

    tail = list(args[positional_index:])
    if not tail or tail[0] != "completion":
        return None
    if tail == ["completion", "fish"]:
        return run_completion_fish()
    if tail == ["completion", "install", "fish"]:
        return run_completion_install_fish()
    print("Usage: asr completion fish | asr completion install fish", file=sys.stderr)
    return 2


def _first_positional_index(args: Sequence[str]) -> int | None:
    options_with_values = {"--output-dir", "--granularity"}

    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            next_index = index + 1
            return next_index if next_index < len(args) else None
        if token.startswith("--"):
            if any(token.startswith(f"{option}=") for option in options_with_values):
                index += 1
                continue
            if token in options_with_values:
                index += 2
                continue
            index += 1
            continue
        return index
    return None
