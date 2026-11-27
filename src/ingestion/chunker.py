"""Document chunking driven by config.yaml strategies.

Previously this module hardcoded chunk_size/overlap, so the strategies
declared in config/config.yaml never reached the chunker and switching a
strategy changed nothing at runtime. It now resolves its defaults from the
named strategy.
"""
from __future__ import annotations

from typing import List, Optional

from src.config import get_strategy


def chunk_text(
    text: str,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    strategy: Optional[str] = None,
) -> List[str]:
    """Chunk ``text`` into overlapping segments.

    Precedence: explicit args > named strategy > "baseline" strategy from
    config.yaml. Passing a strategy name and an explicit chunk_size keeps
    the explicit value, so callers can still override per-call.
    """
    if chunk_size is None or overlap is None:
        resolved = get_strategy(strategy or "baseline")
        if chunk_size is None:
            chunk_size = resolved.chunk_size
        if overlap is None:
            overlap = resolved.chunk_overlap

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size}) "
            "or the window never advances"
        )

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


def chunk_text_for_strategy(text: str, strategy: str) -> List[str]:
    """Chunk ``text`` using the named strategy from config.yaml.

    This is the path the API and the indexing script should use: it makes
    the config value the only thing that decides chunk geometry.
    """
    resolved = get_strategy(strategy)
    return chunk_text(text, chunk_size=resolved.chunk_size, overlap=resolved.chunk_overlap)