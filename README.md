# code-agent

A terminal coding agent for Python repositories, built around a local hybrid code index.

> **Status: work in progress (milestone 1 of 5).** Only indexing and search exist so far.
> See [PLAN.md](PLAN.md) for the full scope. This README only describes what is implemented
> and tested.

## What works today

- `agent index` builds a local index of the current git repository:
  - AST-aware chunks for Python (functions, classes with method signatures, methods, contiguous
    module-level code), line windows for other text files
  - BM25 via SQLite FTS5, vectors via sqlite-vec, both in one `.agent/index.db` file
  - embeddings computed locally (fastembed / ONNX, CPU); code is not sent anywhere to be indexed
  - incremental updates: unchanged files are skipped, and unchanged chunks inside a changed file
    keep their vectors
  - `--watch` re-indexes files as they change
  - `.gitignore`, `.agentignore`, and a built-in list of sensitive paths (`.env*`, keys,
    credential files, ...) are respected; sensitive files are never indexed
- `agent search "query"` runs exact-symbol, BM25 and vector search in parallel and merges them
  with reciprocal rank fusion. If the embedding model is unavailable it falls back to
  symbol + BM25 and says so.

## Measured so far

Indexing [pytest 8.3.4](https://github.com/pytest-dev/pytest) (563 files, 6,950 chunks) on a
laptop CPU (i5-13420H, no GPU), from `benchmarks/bench_index.py`:

| | |
|---|---|
| Keyword + symbol index ready | 2.1 s |
| All embeddings computed (bge-small, local) | ~10 min, one-time |
| Re-sync with nothing changed | 0.42 s |
| One edited file re-indexed (what `--watch` does on save) | p50 263 ms, p95 383 ms |

Retrieval quality has not been measured yet; that is milestone 4 (recall@k on a SWE-bench Lite
subset). Design notes: [ADR 0002](docs/adr/0002-index-storage-and-embeddings.md) (storage and
embeddings), [ADR 0003](docs/adr/0003-ast-chunking.md) (chunking).

## Install (development)

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
agent index            # first run downloads the embedding model once (~70 MB)
agent search "where are expired tokens rejected"
```

## Tests

```bash
pytest
```
