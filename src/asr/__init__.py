"""Top-level package for the asr CLI."""

from importlib import metadata

__all__ = ["__version__"]

_DISTRIBUTION_NAME = "echoalign-asr-mlx"
_UNKNOWN_VERSION = "0+unknown"


def _read_version() -> str:
    try:
        return metadata.version(_DISTRIBUTION_NAME)
    except metadata.PackageNotFoundError:
        return _UNKNOWN_VERSION


__version__ = _read_version()
