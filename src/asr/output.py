"""Output path planning."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


DEFAULT_OUTPUT_DIRNAME = "outputs"


def build_output_path(
    *,
    source: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
) -> Path:
    """Build an output path under the output root while preserving layout."""

    relative_source = source.relative_to(input_root)
    return (output_root / relative_source).with_suffix(suffix)


def default_output_root(input_path: Path, explicit_output_dir: Optional[Path] = None) -> Path:
    """Return the default output root for a CLI input target."""

    if explicit_output_dir is not None:
        return explicit_output_dir
    if input_path.is_file():
        return input_path.parent / DEFAULT_OUTPUT_DIRNAME
    return input_path / DEFAULT_OUTPUT_DIRNAME
