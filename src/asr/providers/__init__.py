"""Provider implementations and factory helpers."""

from __future__ import annotations

from typing import Any

__all__ = ["QwenMlxProvider", "create_default_provider"]


def __getattr__(name: str) -> Any:
    if name == "QwenMlxProvider":
        from asr.providers.qwen_mlx import QwenMlxProvider

        return QwenMlxProvider
    raise AttributeError(name)


def create_default_provider():
    """Return the planned default MLX provider."""

    from asr.providers.qwen_mlx import QwenMlxProvider

    return QwenMlxProvider()
