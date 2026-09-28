# code-agent

A terminal coding agent for Python repositories, built around a local hybrid code index.

> **Status: work in progress (milestone 3 of 5).** Indexing, search, the agent loop, validated
> edits, the shadow workspace, command approvals, secret redaction, the audit log and undo exist.
> Headless mode, evals and CI come next (see [PLAN.md](PLAN.md)). This README only describes what
> is implemented and tested. The test suite uses a scripted fake LLM; against a real model there has
> been one smoke test so far (below), which is a sanity check, not an evaluation.

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
- `agent chat` takes a request, retrieves context within a token budget, lets the model read
  files and search, and streams its edits as SEARCH/REPLACE blocks. Edits are checked before
  you see them (each must match exactly one place in a file the model has read, and the file
  must not have changed since). Before you see the diff, the edits are validated in a hidden
  git worktree (a *shadow* copy of your repo): ruff, pyright and the tests that cover the changed
  files. Only problems the edit introduced count. Failures go back to the model for up to 3 fix
  rounds. You then review the diff (all files or per file) and nothing is written unless you
  confirm. Ctrl+C cancels a running task, including any command it started.
- The model can run terminal commands only through code-enforced approvals: read-only and
  test/lint commands can be allowed for the session; anything else (unknown programs, `pip`,
  `curl`, `rm`, `git push`, shell pipes, paths outside the repo) asks every time. Commands run
  without a shell and with an environment that contains no API keys.
- Secrets are redacted from everything sent to the model (and from the logs). Sensitive files
  (`.env*`, keys, credential files) can't be read or edited at all.
- `agent audit` shows every tool call and user action of the last task, with redacted
  arguments.
- `agent undo` reverts the last applied change, and refuses if you edited those files since.
- `agent models` lists the models your API key can use and checks the configured ones.
- Every task logs its retrieved context and the exact requests sent to the model under
  `.agent/logs/<task>/`.

## Measured so far

On a laptop CPU (i5-13420H, no GPU, Windows 11).

Indexing [pytest 8.3.4](https://github.com/pytest-dev/pytest) (563 files, 6,950 chunks), from
`benchmarks/bench_index.py`:

| | |
|---|---|
| Keyword + symbol index ready | 2.1 s |
| All embeddings computed (bge-small, local) | ~10 min, one-time |
| Re-sync with nothing changed | 0.42 s |
| One edited file re-indexed (what `--watch` does on save) | p50 263 ms, p95 383 ms |

Applying one edit (match + atomic write), from `benchmarks/bench_apply.py`, 50 runs each:

| File size | exact match p50 / p95 | fuzzy match p50 / p95 |
|---|---|---|
| 100 lines | 2.3 / 3.3 ms | 2.1 / 2.7 ms |
| 500 lines | 2.4 / 3.2 ms | 3.0 / 3.7 ms |
| 1000 lines | 2.7 / 4.4 ms | 3.9 / 4.9 ms |

Real-model smoke test (one run, 2026-09-29): on the fixture repo in `tests/fixtures/sample_repo`,
`agent chat -m "fix the bug where expired tokens are still accepted"` with `gemini-3.5-flash-lite`
(query rewriting) and `gemini-3.8-flash` (edits) read the file and its tests, checked callers with
`get_references`, and proposed a one-line fix that applied as an exact match. The fixture's
failing test then passed, and `agent undo` restored the file byte-for-byte. Usage: 5 model calls,
3 tool calls, 8,252 input + 179 output tokens, 21 s; under $0.007 at Google's published paid-tier
prices. One run on a toy repo says nothing about success rates.

Security suite (`tests/test_security_suite.py`): a scripted, fully compromised model obeys
every instruction planted in `tests/fixtures/injection_repo` (`curl | sh`, read `.env`, read and
edit files outside the repo, edit git hooks, grant itself session-wide approval). The simulated
user allows read-only commands and denies the rest.

| Requirement | Result |
|---|---|
| Privileged commands run without approval | 0 |
| Reads of sensitive or out-of-workspace paths | 0 (5 attempts denied) |
| Edits outside the workspace or to protected paths | 0 |
| Planted secrets in outbound requests or logs | 0 |
| Stale edit applied over the user's change | never |
| Cancel during a command | process tree killed, task `CANCELLED` |

What this does *not* cover: an approved command runs with your own permissions, and network
isolation of test runs only exists on Linux. See
[ADR 0007](docs/adr/0007-approvals-capabilities-and-isolation.md) for the per-OS table.

Retrieval quality and task success rates have not been measured yet; that is milestone 4
(SWE-bench Lite subset).

## Design decisions

| Decision | ADR |
|---|---|
| SQLite FTS5 + sqlite-vec in one file; local embeddings | [0002](docs/adr/0002-index-storage-and-embeddings.md) |
| AST chunking instead of fixed windows | [0003](docs/adr/0003-ast-chunking.md) |
| SEARCH/REPLACE blocks as the edit format | [0004](docs/adr/0004-edit-format.md) |
| Match tiers, uniqueness rule, stale-file policy | [0005](docs/adr/0005-fuzzy-matching-and-stale-files.md) |
| Model routing by role; budgets enforced in code | [0008](docs/adr/0008-model-routing-and-budgets.md) |
| Shadow git worktree; only new problems fail validation | [0006](docs/adr/0006-shadow-workspace.md) |
| Approvals and capabilities in code; limits of local isolation | [0007](docs/adr/0007-approvals-capabilities-and-isolation.md) |
| Symbol resolution with the index and jedi | [0010](docs/adr/0010-symbol-resolution.md) |

## Install (development)

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"      # Windows: .venv\Scripts\pip
agent index            # first run downloads the embedding model once (~70 MB)
agent search "where are expired tokens rejected"
```

## Configure a model

Configuration lives in `~/.config/code-agent/config.toml` (or the file named by
`CODE_AGENT_CONFIG`), never in the repository. No model names are built in. Set your key, list
what it can use, then pick models:

```bash
export GEMINI_API_KEY=...        # PowerShell: $env:GEMINI_API_KEY = "..."
agent models
```

```toml
[llm.routes]
cheap  = ["gemini:<a small model from `agent models`>"]
strong = ["gemini:<a capable model>", "gemini:<a fallback model>"]

[llm.pricing."gemini:<a capable model>"]   # optional; without it cost shows as "unknown"
input_per_mtok = 0.0
output_per_mtok = 0.0

[budgets]
max_usd = 1.0
max_tool_calls = 60
```

## Tests

```bash
pytest
```

The tests use a scripted fake LLM and a hashing embedder, so they need no API key, no network
and no model download.
