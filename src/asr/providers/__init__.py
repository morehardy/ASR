"""Provider implementations and factory helpers."""

from asr.providers.qwen_mlx import QwenMlxProvider

__all__ = ["QwenMlxProvider", "create_default_provider"]


def create_default_provider() -> QwenMlxProvider:
    """Return the planned default MLX provider."""

    return QwenMlxProvider()
