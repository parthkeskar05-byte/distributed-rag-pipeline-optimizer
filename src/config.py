"""Configuration loading for the RAG pipeline.

Single source of truth: config/config.yaml. Every component that needs a
chunking or retrieval parameter reads it from here instead of hardcoding,
so a strategy change in the YAML actually changes runtime behaviour.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List

import yaml

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "config.yaml",
)


@dataclass(frozen=True)
class Strategy:
    """One retrieval strategy as declared in config.yaml."""

    name: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    top_k: int
    rerank: bool = False


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> Dict[str, Any]:
    """Load and cache config.yaml. Raises on a missing or malformed file."""
    resolved = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(resolved):
        raise FileNotFoundError(f"config not found: {resolved}")
    with open(resolved, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return data


def get_strategy(name: str, path: str | None = None) -> Strategy:
    """Return a validated Strategy by name. Raises KeyError on unknown names."""
    cfg = load_config(path)
    strategies = cfg.get("strategies")
    if not isinstance(strategies, dict) or not strategies:
        raise ValueError("config.yaml declares no strategies")

    if name not in strategies:
        available = ", ".join(sorted(strategies))
        raise KeyError(f"unknown strategy {name!r}; available: {available}")

    raw = strategies[name] or {}

    chunk_size = int(raw.get("chunk_size", 500))
    overlap = int(raw.get("chunk_overlap", 50))

    if chunk_size <= 0:
        raise ValueError(f"strategy {name!r}: chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"strategy {name!r}: chunk_overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"strategy {name!r}: chunk_overlap ({overlap}) must be < "
            f"chunk_size ({chunk_size}) or chunking never advances"
        )

    top_k = int(raw.get("top_k", 3))
    if top_k <= 0:
        raise ValueError(f"strategy {name!r}: top_k must be > 0, got {top_k}")

    return Strategy(
        name=name,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        embedding_model=str(raw.get("embedding_model", "all-MiniLM-L6-v2")),
        top_k=top_k,
        rerank=bool(raw.get("rerank", False)),
    )


def list_strategies(path: str | None = None) -> List[str]:
    """Strategy names declared in config, sorted for stable output."""
    cfg = load_config(path)
    return sorted((cfg.get("strategies") or {}).keys())


def clear_cache() -> None:
    """Drop cached config. Needed by tests that write a temp config."""
    load_config.cache_clear()