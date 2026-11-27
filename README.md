# Distributed RAG Pipeline Optimizer

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A RAG pipeline whose retrieval strategies are **declared in config and actually
reach the chunker**, so strategies can be compared honestly.

## Status

Chunking and strategy configuration are wired end-to-end and tested. The
retrieval, generation, reranking, and A/B-benchmarking layers are **scaffolding
only** — declared in the architecture diagram below but not implemented. See
[What is and isn't implemented](#what-is-and-isnt-implemented).

## Quick start

Index a document with a strategy declared in `config/config.yaml`:

```bash
python scripts/index_documents.py --input data/docs/ --strategy baseline
python scripts/index_documents.py --input data/docs/ --strategy optimized
```

Serve the API:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Query it:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?", "strategy": "optimized", "top_k": 5}'
```

## Configuration

`config/config.yaml` is the single source of truth for retrieval strategies:

```yaml
strategies:
  baseline:
    chunk_size: 500
    chunk_overlap: 50
    embedding_model: "all-MiniLM-L6-v2"
    top_k: 3
  optimized:
    chunk_size: 300
    chunk_overlap: 100
    embedding_model: "all-MiniLM-L6-v2"
    top_k: 5
    rerank: true

vector_store:
  type: "faiss"
  index_type: "Flat"

metrics:
  - precision@k
  - recall@k
  - mrr
```

Every parameter here changes runtime behaviour. `src/config.py` loads and
validates it (rejecting `chunk_overlap >= chunk_size`, which would otherwise
make the chunker loop forever), and the chunker and API read from it rather
than hardcoding values.

## API

### `GET /strategies`

Strategies actually declared in `config/config.yaml`.

### `POST /query`

```json
{"query": "What is machine learning?", "strategy": "optimized", "top_k": 5}
```

```json
{
  "resolved_strategy": {
    "chunk_size": 300,
    "chunk_overlap": 100,
    "embedding_model": "all-MiniLM-L6-v2",
    "top_k": 5,
    "rerank": true
  },
  "chunk_count_for_query": 5,
  "strategy": "optimized"
}
```

`resolved_strategy` echoes the values the request actually resolved to, so a
config change is observable through the API. `answer`, `sources`, and
`latency_ms` are `null` until the retrieval layer is implemented.

An unknown strategy returns `400` with the list of available names, rather
than silently falling back.

## What is and isn't implemented

| Layer | Status |
|---|---|
| `src/config.py` — load + validate strategies | **Implemented, tested** |
| `src/ingestion/chunker.py` — config-driven chunking | **Implemented, tested** |
| `src/api/app.py` — strategy resolution + endpoints | **Implemented, tested** |
| `scripts/index_documents.py` — `--input`/`--strategy` CLI | **Implemented, tested** |
| `src/retrieval/` — FAISS index, embeddings | Scaffolding (module exists, unwired) |
| `src/generation/` — LLM answer generation | Scaffolding (empty) |
| Reranking (`rerank: true` is parsed, not applied) | Parsed only |
| A/B benchmark harness, metrics | Not implemented |
| `scripts/run_ab_test.py`, `scripts/benchmark.py` | Do not exist |

`src/retrieval/embeddings.py` contains an `EmbeddingEngine`, but nothing
imports it. Retrieval quality and latency numbers for the two strategies have
**not** been measured, so none are reported here.

## Tests

```bash
pytest
```

The suite asserts that strategy values from `config.yaml` reach the chunker —
that strategies produce *different* chunk counts, and that `overlap >= chunk_size`
is rejected. This is what the original single assertion (`len(chunks) > 0`)
failed to catch: it passed whether or not the config was wired up at all.

## Project structure

```
distributed-rag-pipeline-optimizer/
├── README.md
├── pyproject.toml          # pytest config (pythonpath)
├── requirements.txt
├── config/
│   └── config.yaml
├── src/
│   ├── config.py           # strategy loading + validation
│   ├── ingestion/
│   │   └── chunker.py      # config-driven chunking
│   ├── api/
│   │   └── app.py
│   ├── retrieval/          # scaffolding
│   ├── generation/         # scaffolding
│   ├── evaluation/         # scaffolding
│   └── utils/              # scaffolding
├── scripts/
│   └── index_documents.py
└── tests/
    └── test_chunker.py
```

## License

MIT License.

## Author

**Parth Keskar** - AI/ML Engineer  
[GitHub](https://github.com/parthkeskar05-byte) | [Portfolio](https://parth-keskar.vercel.app)