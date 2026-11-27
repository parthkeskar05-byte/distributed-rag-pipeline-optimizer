"""Tests for config-driven chunking.

The previous suite only asserted ``len(chunks) > 0``, which passes whether
or not the chunker reads config.yaml. These tests assert the strategy
values actually reach the chunker, so re-breaking the wiring fails loudly.
"""
import pytest

from src.config import clear_cache, get_strategy, list_strategies
from src.ingestion.chunker import chunk_text, chunk_text_for_strategy


@pytest.fixture(autouse=True)
def _clear_config_cache():
    clear_cache()
    yield
    clear_cache()


def test_config_declares_both_strategies():
    strategies = list_strategies()
    assert "baseline" in strategies
    assert "optimized" in strategies


def test_strategies_have_distinct_geometry():
    """The A/B comparison only means something if the two differ."""
    baseline = get_strategy("baseline")
    optimized = get_strategy("optimized")
    assert baseline.chunk_size != optimized.chunk_size
    assert baseline.chunk_overlap != optimized.chunk_overlap
    assert baseline.top_k != optimized.top_k


def test_optimized_strategy_is_declared_with_rerank():
    assert get_strategy("optimized").rerank is True
    assert get_strategy("baseline").rerank is False


def _expected_chunks(text_len, chunk_size, overlap):
    """Mirror the chunker's loop: emit while start < text_len, advance by step."""
    step = chunk_size - overlap
    return (text_len - 1) // step + 1


def test_chunker_uses_configured_baseline_geometry():
    """Regression test: the old chunker hardcoded 500/50 and ignored the
    config, so a strategy change had no observable effect."""
    resolved = get_strategy("baseline")
    text = "a" * 1000
    chunks = chunk_text_for_strategy(text, "baseline")

    assert len(chunks) == _expected_chunks(
        len(text), resolved.chunk_size, resolved.chunk_overlap
    ), (
        f"chunker did not honour baseline chunk_size={resolved.chunk_size} "
        f"overlap={resolved.chunk_overlap}"
    )
    assert all(len(c) == resolved.chunk_size for c in chunks[:-1])


def test_chunker_uses_configured_optimized_geometry():
    """Directly proves config.yaml drives runtime behaviour."""
    resolved = get_strategy("optimized")
    text = "a" * 1000
    chunks = chunk_text_for_strategy(text, "optimized")

    assert len(chunks) == _expected_chunks(
        len(text), resolved.chunk_size, resolved.chunk_overlap
    )
    assert len(chunks) != len(chunk_text_for_strategy(text, "baseline"))


def test_unknown_strategy_raises_keyerror():
    with pytest.raises(KeyError):
        chunk_text_for_strategy("text", "does-not-exist")


def test_explicit_arguments_override_strategy():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 0
    assert all(len(c) == 500 for c in chunks[:-1])


def test_overlap_greater_or_equal_chunk_size_rejected():
    """The old chunker silently looped forever on overlap >= chunk_size."""
    with pytest.raises(ValueError, match="never advances"):
        chunk_text("abc", chunk_size=100, overlap=100)
    with pytest.raises(ValueError, match="never advances"):
        chunk_text("abc", chunk_size=100, overlap=150)