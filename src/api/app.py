"""FastAPI application for RAG pipeline.

The /query endpoint previously returned a hardcoded "Sample RAG response"
and only echoed request.strategy, so the strategies declared in
config/config.yaml never affected behaviour and the A/B comparison the
README advertises was unreachable. It now resolves the strategy, reports
the resolved chunking/retrieval parameters, and surfaces a real error when
the requested combination cannot be satisfied.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import get_strategy, list_strategies
from src.ingestion.chunker import chunk_text_for_strategy

app = FastAPI(title="Distributed RAG Pipeline Optimizer")


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    strategy: str = "baseline"
    top_k: int | None = Field(default=None, ge=1)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/strategies")
async def strategies():
    """Return the strategies actually declared in config/config.yaml."""
    return {"strategies": list_strategies()}


@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        resolved = get_strategy(request.strategy)
    except KeyError as exc:
        available = list_strategies()
        raise HTTPException(
            status_code=400,
            detail=f"unknown strategy {request.strategy!r}; available: {available}",
        ) from exc

    # The whole point of the exercise: chunk the query-sized sample using the
    # strategy's own geometry so a config change is observable here.
    try:
        chunks = chunk_text_for_strategy(request.query, request.strategy)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    effective_top_k = request.top_k if request.top_k is not None else resolved.top_k

    return {
        "answer": None,
        "sources": [],
        "latency_ms": None,
        "strategy": request.strategy,
        "resolved_strategy": {
            "chunk_size": resolved.chunk_size,
            "chunk_overlap": resolved.chunk_overlap,
            "embedding_model": resolved.embedding_model,
            "top_k": effective_top_k,
            "rerank": resolved.rerank,
        },
        "chunk_count_for_query": len(chunks),
    }