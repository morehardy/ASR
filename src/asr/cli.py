"""Command-line interface for the ASR tool."""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from asr.discovery import discover_media_files
from asr.exporters import render_json, render_srt, render_vtt
from asr.media import FfmpegMediaPreparer
from asr.output import build_output_path, default_output_root
from asr.pipeline import process_media_file
from asr.providers import create_default_provider


def build_parser() -> argparse.ArgumentParser:
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    discovered_sources = discover_cli_sources(args.inputs, recursive=args.recursive)
    if not discovered_sources:
        print("No supported media files found.", file=sys.stderr)
        return 1

    provider = create_default_provider()
    media_preparer = FfmpegMediaPreparer()
    had_error = False

    for source_path, input_root in discovered_sources:
        output_root = default_output_root(input_root, explicit_output_dir=args.output_dir)
        try:
            document = process_media_file(
                source_path=source_path,
                provider=provider,
                media_preparer=media_preparer,
            )
            rendered_outputs = {
                ".srt": render_srt(document, granularity=args.granularity),
                ".vtt": render_vtt(document, granularity=args.granularity),
                ".json": render_json(document, granularity=args.granularity),
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

            if args.verbose:
                print(f"[asr] processed {source_path} -> {output_root}")
            else:
                print(source_path)
        except Exception as exc:  # pragma: no cover - surfaced to CLI output
            had_error = True
            print(f"[asr] failed for {source_path}: {exc}", file=sys.stderr)

    return 1 if had_error else 0
