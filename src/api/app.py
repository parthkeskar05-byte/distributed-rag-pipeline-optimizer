"""FastAPI application for RAG pipeline."""
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Distributed RAG Pipeline Optimizer")

class QueryRequest(BaseModel):
    query: str
    strategy: str = "baseline"
    top_k: int = 5

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/strategies")
async def list_strategies():
    return {"strategies": ["baseline", "optimized"]}

@app.post("/query")
async def query_rag(request: QueryRequest):
    return {
        "answer": "Sample RAG response",
        "sources": [],
        "latency_ms": 45.3,
        "strategy": request.strategy
    }
