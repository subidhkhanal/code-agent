# ADR 0002: Index storage (SQLite FTS5 + sqlite-vec) and local embeddings

- Status: accepted
- Date: 2026-09-28
- Milestone: M1

## Context

The index has to support three lookups: exact symbol, keyword (BM25), and semantic (vector). It
must update incrementally as files change, survive crashes, and run fully locally. Indexing must
not send code to any server.

## Decision 1: one SQLite file for everything

Chunk rows, the FTS5 BM25 table, the vector table, and the agent's task/approval/audit/undo tables
all live in `<repo>/.agent/index.db`.

**Vector store: sqlite-vec over LanceDB.**

| | sqlite-vec | LanceDB |
|---|---|---|
| Same transaction as chunk + FTS rows | **yes** | no (separate store) |
| Extra files / services | none (a loadable extension) | a directory of Lance files |
| ANN index | brute force (exact) | IVF-PQ etc. |
| Scale where it stays fast | ~100k vectors (a large monorepo) | millions |

Why sqlite-vec won:
- **Consistency is the hard problem here, not scale.** Every per-file update deletes and
  re-inserts chunk rows, BM25 rows, and vectors. With one SQLite transaction they cannot
  disagree. With two stores, a crash between the two writes leaves the retrievers disagreeing
  about which chunks exist, and that needs reconciliation code.
- **Scale fits.** A single repo has thousands to tens of thousands of chunks. Exact KNN over
  10k × 768 floats takes milliseconds, and "exact" also removes ANN recall as a variable in the
  retrieval evals.
- One dependency, and one file to delete when the index is corrupt.

When LanceDB (or a server) would win: a cross-repo or org-wide index with millions of chunks, see
"Scaling to production" in the README.

**BM25: FTS5** with the `porter unicode61` tokenizer. unicode61 already splits `snake_case` and
dotted names. We add the camelCase pieces of identifiers to the indexed text so
`calculateTotal` matches "calculate total". Column weights are symbol 4 > path 2 > body 1.
Queries are reduced to quoted terms joined with OR, so FTS syntax in a query can never be
injected (tested with hostile inputs).

**Trust boundary for the DB location:** the directory writes a `.gitignore` containing `*` into
itself, so it never shows up in `git status`. `.agent/` is also on the hard-coded sensitive-path
list, so the agent can never read or edit its own audit log or undo records through its tools.

## Decision 2: incremental updates and integrity

- Change detection per file: (size, mtime) fast path, then a content hash. There is a
  *racy-mtime guard* borrowed from git: if a file's mtime is within 2 s of when we hashed it, the
  fast path is not trusted, because a same-size edit in the same timestamp tick would otherwise
  go unnoticed. This is tested.
- **Per-chunk vector reuse**: each chunk stores `embed_hash = sha256(model_id, embedded text)`.
  When a file changes, vectors of chunks whose text is unchanged are copied over. Editing one
  method in a 40-method file re-embeds one chunk (tested).
- Chunks and BM25 rows are written first. Embeddings follow in committed batches. An
  interrupted or failed embedding run leaves a fully working keyword index, and the next run
  embeds only what is missing (tested with an embedder that fails mid-run).
- On open: `PRAGMA quick_check`, schema-version check, and drift checks between chunk rows,
  BM25 rows, and vectors. Drift is repaired. An unreadable DB is moved aside
  (`index.db.bad-<ts>`) and rebuilt, because the workspace is always the source of truth.
- Changing the embedding model drops all vectors and re-embeds. Vectors from two models are
  never mixed, and search refuses (with a note) to query vectors from a model other than its own.

## Decision 3: local embeddings via fastembed (ONNX, CPU)

fastembed over sentence-transformers: it runs ONNX models without PyTorch (roughly a 200 MB
install instead of about 2 GB), which matters for `pipx install` and for the Docker image in M4.

Candidates available in fastembed:

| Model | Dim | Size | Trained on code? |
|---|---|---|---|
| `jinaai/jina-embeddings-v2-base-code` | 768 | 0.64 GB | **yes** (code + NL pairs, 30 languages) |
| `BAAI/bge-small-en-v1.5` | 384 | 0.07 GB | no (general English) |
| `nomic-ai/nomic-embed-text-v1.5-Q` | 768 | 0.13 GB | no |

**Default model: `BAAI/bge-small-en-v1.5`**, chosen on measured CPU cost (see Benchmark). jina-code
is the only code-trained option, but on this machine it embeds 3–7× slower (2.0–4.5 chunks/s vs
14–16 chunks/s on the same pytest chunks). That turns a ~10-minute first index of a mid-size repo
into 30–60 minutes. bge-small is not code-trained. It is paired with BM25 and exact-symbol search,
which cover identifiers precisely, so the vector retriever mainly has to handle natural-language
paraphrase, and a general English model is reasonable for that. The choice is **provisional**:
the M4 retrieval ablation measures recall@k for both models on the SWE-bench subset, and the
default follows the data. Switching is one config line (`index.embedding_model`), and the index
re-embeds automatically.

Degraded mode: if the model cannot be loaded (offline before the first download, corrupt cache),
indexing still builds the keyword index, and `agent search` runs symbol + BM25 and prints why
vector search is unavailable.

## Benchmark

Machine: Intel i5-13420H (8 cores / 12 threads), 16 GB RAM, Windows 11, on AC power, CPU only.
Repo: **pytest-dev/pytest @ 8.3.4** (a SWE-bench Lite repo): 563 indexed files (264 Python),
6,950 chunks. Script: `benchmarks/bench_index.py`, raw output in `benchmarks/results/`.

| Metric | Result |
|---|---|
| Discover + hash + chunk + FTS5 write (index usable for BM25/symbol search) | **2.1 s** |
| Embed all chunks, bge-small, length-sorted batches | **599 s** (11.6 chunks/s) |
| Same, before length-sorting (baseline, 4,000-char cap) | 1,130 s (6.2 chunks/s) |
| No-op re-sync (nothing changed) | **0.42 s** |
| Incremental update, one function edited in one file (n=20) | **p50 263 ms, p95 383 ms** |
| Chunks re-embedded per incremental update (median) | 1 |

Throughput findings that drove the design:
- Embedding cost scales with tokens, not chunk count: 8-token inputs ran at 393/s, 152-token
  inputs at 23/s, 512-token inputs at 5/s. Padding is dynamic (to the longest input in each
  batch), so **sorting chunks by length before batching** roughly doubled throughput.
- The ONNX thread count was already fine (default ≈ 4–8 threads; 1 thread = 2.9/s).
- A first measurement of this benchmark reported 43,000 s because the laptop suspended
  mid-run (wall clock includes sleep). That run was discarded, and the benchmark now prints
  progress so a stall is visible.

## Known issue (to be settled with the M4 retrieval eval)

FTS5's `bm25()` normalizes by the length of the *whole row* (symbol + path + body), so very short
chunks (a one-line module constant, a section-divider comment) get a large length bonus. If such
a chunk's only match is its file path, it can outrank the chunks that actually discuss the query.
Lowering the path weight from 2.0 to 0.5 did not fix the observed case; removing the path column
did, but that loses a real signal. Candidate fixes: merge tiny module runs into an adjacent chunk,
move path matching into its own retriever, or retune weights. None is applied yet, because tuning
on a handful of hand-picked queries would overfit. The recall@k ablation in M4 decides.
