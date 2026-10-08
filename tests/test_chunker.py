"""Test suite for chunking."""
from src.ingestion.chunker import chunk_text

def test_chunk_text():
    text = "a" * 1000
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 0
