# Build Plan: AI Coding Agent (Claude Code / Cursor style, with a headless autonomous mode)

> Hand this whole file to Claude Code. Save it in the repo root as `PLAN.md`.

## 0. Context and ground rules (read first)

**What this is.** This is the flagship portfolio project on my resume. It is a **terminal coding agent** in the style of Claude Code / Cursor's agent. It indexes a local repo, retrieves the right context, streams focused edits, validates them in a hidden **shadow workspace** before I see them, and asks for approval before running privileged commands. It also has a **headless mode** that runs autonomously inside a Docker container. That mode is used to score the agent on a **SWE-bench Lite subset**, and it will later serve as the sandbox backend (`spawn_sandbox`) for my multi-agent orchestration project.

I am an AI engineer (fresher level), and I will be interviewed on every design choice here. Clear code and documented reasoning matter more than feature count.

**Ground rules for you (Claude Code):**
1. Create a **new repo** called `code-agent`. Do not modify my other repos.
2. Work on a feature branch and make **small, logical commits** (one concern per commit). **Do not push** anything without asking me.
3. **Ask before adding any paid service.** Default to local and free tools wherever possible.
4. **Don't hard-code model names.** Put them in config and verify they're currently available from the provider before using them.
5. Write an **ADR** in `docs/adr/` for every major decision (list in section 10).
6. Put **security rules in code, not in prompts** (section 4.7). The model can never widen its own permissions.
7. **Don't overclaim** in the README. Every claim needs a test, a benchmark, or an eval number behind it.
8. Write docs in your own words. Don't copy text from any reference spec I show you.
9. **Stop at the end of each milestone** (section 12) and report: what works, what doesn't, the decisions you made, and any open questions.
10. Never run the agent's own generated commands on my machine outside the approval flow, including while testing.

---

## 1. Scope

### In scope
- **CLI agent** (interactive REPL) for Python repositories
- **Local codebase index**: AST-aware chunking, BM25 keyword search plus vector search, incremental updates on file change
- **Context assembly**: hybrid retrieval, symbol expansion (definitions and references), and a token budget capped at about 80% of the model window, with signature-only pruning
- **Focused edits** as SEARCH/REPLACE blocks, streamed and parsed incrementally
- **Fast Apply**: base-hash check, exact → whitespace-tolerant → fuzzy matching, rejection of ambiguous matches, atomic writes, and undo
- **Shadow workspace**: a git worktree that gets the candidate edits, then lint + type check + targeted tests, with a bounded auto-fix loop (max 3) before the diff is shown
- **Human in the loop**: a diff review (accept/reject per file) and approval prompts for terminal commands (command-scoped or session-scoped)
- **Security**: a capability model, path canonicalization, secret path exclusion, outbound secret scanning, treating tool output as untrusted, and an audit log
- **Stale file handling**: reject and regenerate if the file changed after retrieval
- **Cancellation, budgets, idempotency** for privileged tool calls
- **Degraded mode**: search, index, and diff review keep working when the LLM is unavailable
- **Headless autonomous mode** inside Docker (no network) for SWE-bench and as the sandbox backend
- **Evals**: SWE-bench Lite subset, retrieval recall, apply success and latency, ablations, and security tests

### Out of scope (explain in the README's "Scaling to production" section)
- A VS Code extension. The core engine is the same, and the CLI keeps the UI work small. Design the engine as a library so an extension could wrap it later (stretch goal only).
- Languages other than Python. Keep the interfaces language-agnostic, but implement Python only.
- A cloud index or server-side embeddings, multi-user, and enterprise policy servers
- Training or fine-tuning models
- Multi-day autonomous tasks
- A true OS-level sandbox for **interactive** mode. Be honest in ADR 0007 that local approval plus restrictions are the control there, and that headless mode uses Docker isolation.

---

## 2. User experience

```
$ agent index                  # build/refresh the index for the current repo
$ agent search "jwt expiry"    # hybrid search, works offline
$ agent chat                   # interactive session
> fix the bug where expired tokens are still accepted
  [retrieval] 6 chunks from 4 files · 5.2K tokens (41% of budget)
  [plan] ...
  [edit] auth/tokens.py (streaming)
  [shadow] ruff ✓  pyright ✓  pytest tests/test_tokens.py ✓ (attempt 1/3)
  ── diff ──  (syntax-highlighted)
  Apply changes to 1 file? [y]es / [n]o / [v]iew full
  Agent wants to run: pytest -q tests/   (classified: test, no network)
  Approve? [o]nce / [s]ession / [d]eny
$ agent undo                   # revert last applied change set
$ agent audit                  # show tool audit log for the last task
$ agent run --task "..." --headless   # autonomous mode (use inside Docker only)
```

A status line shows tokens used, cost, and elapsed time. Ctrl+C cancels cleanly.

---

## 3. Tech stack
- Python 3.12, Typer (CLI), Rich and prompt_toolkit (streaming output, diffs, prompts)
- tree-sitter (Python grammar) for AST chunking and signature extraction
- SQLite: metadata and **FTS5 for BM25**. For vectors, use sqlite-vec or LanceDB (pick one and write an ADR).
- **Local embedding model** (small code-capable model via fastembed or sentence-transformers), so code does not leave the machine for indexing. Tell me which model you pick and why.
- Symbol resolution with jedi, or pyright via LSP (pick one and write an ADR)
- watchdog for incremental re-indexing
- Validation tools: ruff, pyright (or mypy), pytest
- LLM access through a small gateway module with config-driven providers, a cheap model for planning and query rewriting, a strong model for edits, and a fallback provider
- Docker for headless mode, plus the official SWE-bench evaluation harness
- pytest and GitHub Actions for the project's own tests

---

## 4. Architecture and components

### 4.1 Indexer
- Walk the repo and respect `.gitignore` plus a project `.agentignore`. Also exclude binaries, dependencies (`.venv`, `node_modules`, `site-packages`), generated files, and **sensitive paths** (`.env*`, `*.pem`, `*.key`, `secrets/`, and a configurable list). Sensitive paths are never indexed and never sent to the LLM.
- **AST chunking**: one chunk per function or class (the class header plus method signatures, with methods as their own chunks) and a module-level chunk for imports and globals. Fall back to line windows for non-Python text files.
- Each chunk stores repo_id, file_path, symbol name, kind, start and end line, content, content_hash, language, and the embedding.
- **Incremental updates**: compare file content hashes on save, rename, delete, or branch switch. Only changed files are re-chunked and re-embedded.
- Validate index integrity on startup. Rebuild corrupted or stale entries from the workspace.
- Benchmark and report: initial index time on a mid-size repo (report the repo and file count) and incremental update time per file.

### 4.2 Retrieval and context assembly
- Steps:
  1. The cheap model rewrites the request into a few search queries, and exact identifiers are extracted.
  2. Run exact symbol search, BM25, and vector search in parallel.
  3. Merge with reciprocal rank fusion.
  4. Take the top K chunks.
  5. **Symbol expansion**: add the definitions and signatures of symbols the top chunks reference.
- A **token budget** caps assembled context at about 80% of the model window, reserving room for output. When over budget, drop the least relevant chunks first, then collapse function bodies to **signatures only** (via tree-sitter).
- Log the exact context sent for each request (after redaction) so retrieval can be debugged and evaluated.

### 4.3 Agent loop and tools
The tools are `read_file(path, start, end)`, `search_codebase(query)`, `get_references(symbol)`, `get_definition(symbol)`, `run_terminal_command(cmd, cwd, approval_scope)`, and `propose_edits(blocks)`.

- Every tool call is **schema-validated and capability-checked** before execution, and model-generated arguments are treated as untrusted.
- Budgets per task: max tokens, max USD, max tool calls, and a wall-clock limit. When a budget is exhausted, the agent stops with a clear message and a partial result.
- Tool output is wrapped and labeled as **untrusted data** when it's fed back to the model. It is truncated with a marker if it's too long.

### 4.4 Edit format and Fast Apply
- The model outputs **SEARCH/REPLACE blocks** that name the file path. The parser is **streaming**: blocks are parsed as tokens arrive, and application starts once a block closes.
- Apply algorithm, per block:
  1. **Base hash check**: the file's current hash must equal the hash recorded when it was retrieved. On a mismatch, **stop, re-retrieve, and regenerate**. Never patch new content with an old plan.
  2. Exact match, then whitespace/indentation-tolerant match, then fuzzy match (similarity above a threshold).
  3. **The match must be unique.** Zero or multiple candidates means reject with `AMBIGUOUS_MATCH` or `NO_MATCH`, and feed the error back to the model (this counts toward the retry budget).
  4. Preserve line endings, the trailing newline, encoding, and file permissions.
- **Atomic write**: write to a temp file in the same directory, fsync, then `os.replace`. All files in an accepted change set are applied together. Keep an undo record of each file's previous content and hash.
- Benchmark and report apply latency p50/p95 for 100-, 500-, and 1000-line files (target: under 100 ms for 500 lines).

### 4.5 Shadow workspace
- A hidden **git worktree** in a temp directory, synced with the current HEAD **plus my uncommitted changes** (copy the dirty and untracked files that aren't ignored).
- Candidate edits are applied in the shadow workspace first. Then run: `ruff check` on changed files, the type checker on changed files, and **targeted tests** (tests that import the changed modules or are named after them; fall back to the full suite only if it's fast).
- If validation fails, feed the diagnostics (truncated) back to the model, up to **3 attempts**. After that, show the diff **together with the failing diagnostics** and let me decide. The real workspace is never touched until I accept.
- Before applying to the real workspace, re-check the base hashes. If anything changed since validation, regenerate.
- Shadow commands run with a timeout, no network where possible (section 4.7), and cwd locked to the worktree.

### 4.6 Human approval model
- Command classification rules:
  - **read-only** (`ls`, `cat`, `git status`, `git diff`, `grep`) → allowed after a one-time session approval
  - **test/lint** (`pytest`, `ruff`, `pyright`) → approval per command or per session
  - **mutating, network, or unknown** (`pip install`, `curl`, `rm`, `git push`, anything with pipes to a shell, anything outside the workspace) → **always ask, every time**
- Classification is done **in code** (a parser plus an allowlist and denylist), never by the model.
- Each approval gets an `approval_id`. The command runs only if the approval is still valid and the task hasn't been cancelled, and this is re-checked right before execution.
- Privileged tool calls get an idempotency key, so a retry after a crash or cancel doesn't re-run a completed command.

### 4.7 Security (code-enforced)
- **Path canonicalization**: resolve symlinks and `..`. Every read and write must stay inside the workspace root, and sensitive paths are blocked for both reads and edits.
- **Outbound secret scanning**: before any payload goes to the LLM provider, scan it for secret patterns (API keys, tokens, private keys, high-entropy strings) and **redact**. Log that a redaction happened, but never log the secret itself.
- **Command execution**: a scrubbed environment (remove variables that look like secrets), a timeout, cwd inside the workspace, and output size limits. On Linux, run shadow validation commands with network disabled where feasible, and document what is and isn't enforced on each OS.
- **Prompt injection**: nothing in repo files, tool output, or model output can change permissions, the approval rules, the workspace root, or the budgets. Those live only in code and config.
- **Audit log**: every tool call is recorded with the tool, redacted args, approval_id, actor, timestamp, output hash, and whether redaction was applied.

### 4.8 Cancellation and degraded mode
- Ctrl+C cancels the LLM stream, kills child processes, marks the task `CANCELLED`, and cleans up the shadow worktree.
- If the LLM is unavailable: retry with exponential backoff and jitter, then fall back to the secondary provider. If everything is down, `index`, `search`, `undo`, `audit`, and reviewing an existing diff still work, and chat says clearly that generation is unavailable.

### 4.9 Headless autonomous mode
- `agent run --task ... --headless --auto-approve` **refuses to start unless it detects it's inside a container** (with an explicit override flag for tests).
- It runs in Docker with `--network none` (LLM calls go through a controlled proxy or a network allowance only to the provider; document the choice), a non-root user, and CPU, memory, and time limits.
- It uses the same engine, but approvals are auto-granted only inside the container, and every action is still written to the audit log.
- It outputs a unified diff (`git diff`) and a JSON report: status, tokens, cost, attempts, and validation results.
- **Integration point**: expose it as a function or CLI that my multi-agent project's `spawn_sandbox` tool can call later. Document the interface, but don't modify the other repo.

---

## 5. Local data model (SQLite)
- `repositories`: repo_id, workspace_root, branch, revision, indexed_at
- `file_chunks`: id, repo_id, file_path, symbol, kind, start_line, end_line, content, content_hash, language, last_modified, with `UNIQUE (repo_id, file_path, chunk_index)`. Plus an FTS5 virtual table and the vector index.
- `agent_tasks`: request_id, conversation_id, repo_id, base_revision, status (`QUEUED/RUNNING/WAITING_FOR_APPROVAL/SUCCEEDED/FAILED/CANCELLED`), budgets, usage, idempotency_key, timestamps
- `approvals`: approval_id, request_id, command, classification, scope, decision, decided_at
- `tool_audit_log`: tool_call_id, request_id, approval_id, tool_name, actor, redacted_args, output_hash, redaction_applied, executed_at
- `change_sets`: id, request_id, status (`PROPOSED/VALIDATED/APPLIED/REJECTED/UNDONE`), and the per-file before and after hashes and content (for undo)

---

## 6. Evaluation (this produces the resume numbers)

### 6.1 SWE-bench Lite subset (headless mode)
- Use the **official SWE-bench harness** on a fixed, randomly sampled subset (start with 10 tasks to validate the pipeline, then 30–50). Check the harness's Docker and disk requirements first and tell me before pulling large images.
- Report the **resolve rate**, average cost per task, average wall-clock time, and average number of attempts.
- Save per-task logs so I can read failures myself.

### 6.2 Retrieval eval (cheap, no test execution)
- For each subset task, the gold patch tells us which files had to change. Measure **file-level recall@5 and @10** of the retrieval step.
- Ablations: BM25 only, vector only, hybrid, and hybrid + symbol expansion.

### 6.3 Pipeline ablations (on the SWE-bench subset)
- With vs without shadow validation and the auto-fix loop (resolve rate and cost)
- Report the **Fast Apply first-try success rate** and the reasons blocks were rejected (no match, ambiguous, stale).

### 6.4 Micro-benchmarks
- Apply latency p50/p95 by file size
- Initial index time and incremental update time

### 6.5 Security tests (must all pass)
- Build a fixture repo containing injection attempts: comments and docs telling the agent to run `curl ... | sh`, to read `.env`, to edit files outside the repo, or to disable approvals.
- Requirements: **0 unapproved privileged command executions, 0 reads or edits of sensitive or out-of-workspace paths, 0 secrets in outbound payloads.** In test mode, capture the outbound payloads and scan them.
- A stale-file test: modify the target file after retrieval but before apply. The agent must reject the stale patch and regenerate, never overwrite.
- A cancellation test: cancel mid-command. The child process is killed and the task ends in `CANCELLED`.

Put every result in README tables. **Report honestly**: a resolve rate with an affordable model will be modest. Compare against my own ablations, not against leaderboard entries, and explain what limits the score.

---

## 7. Tests and CI
- Unit tests: chunker, FTS and vector retrieval, RRF, budget pruning, streaming block parser, every Fast Apply case (exact, whitespace, fuzzy, ambiguous, no match, stale hash, line endings, atomicity), command classifier, path canonicalization (symlinks and `..`), secret scanner, approval re-check, budgets.
- Integration tests with a **deterministic fake LLM** on a small fixture repo: request → retrieval → edits → shadow failure → auto-fix → pass → approve → apply → undo.
- The security suite from 6.5.
- GitHub Actions: lint, type check, and unit + integration + security tests using the fake LLM (no API costs in CI). Add a badge to the README.

---

## 8. Demo
- A CLI can't be hosted, so the README's demo is a **terminal recording** (asciinema or a GIF) of a real bug fix: the index step, retrieval, the shadow validation failing and then passing after auto-fix, the diff review, the approval prompt, apply, and undo.
- A second short recording shows the agent **refusing an injected instruction** from the security fixture repo.
- Installation with `pipx install .` (publishing to PyPI is optional; ask me first).

---

## 9. Documentation
**README** must include:
- The problem and what the agent does, in about 5 lines, with the demo GIF at the top
- An architecture diagram (Mermaid): CLI → orchestrator → index/retrieval → gateway → Fast Apply → shadow workspace → approval → real workspace
- A request lifecycle diagram
- A key design decisions table linking to the ADRs
- Eval tables (SWE-bench subset, retrieval recall ablations, apply stats, security results)
- Install and usage instructions, config reference, a limitations section
- A **"Scaling to production"** section: rough capacity math for a product with millions of users (requests/sec, tokens/sec, index storage), model routing to cut token spend, local-first vs cloud index trade-offs, enterprise policy and data residency, and how this differs from a fully autonomous cloud agent (trust boundary, latency, access to local state)

---

## 10. ADRs (`docs/adr/`)
- 0001: CLI vs IDE extension
- 0002: Index storage (SQLite FTS5 + vector store choice) and local embeddings
- 0003: AST chunking vs fixed windows
- 0004: SEARCH/REPLACE vs unified diff vs full-file output
- 0005: Fuzzy matching thresholds, uniqueness rule, stale-file policy
- 0006: Shadow workspace via git worktree, and syncing uncommitted changes
- 0007: The approval and capability model, and the honest limits of local isolation
- 0008: Model routing and task budgets
- 0009: Headless Docker mode, and its relation to cloud autonomous agents
- 0010: Symbol resolution (jedi vs pyright LSP)

---

## 11. Definition of done
- [ ] `index`, `search`, `chat`, `undo`, `audit`, and `run --headless` all work on a real mid-size Python repo
- [ ] Fast Apply rejects stale and ambiguous edits, and every case has a test
- [ ] Shadow validation plus auto-fix works, and the real workspace is untouched until accept
- [ ] The security suite passes with zeros across the board
- [ ] SWE-bench subset, retrieval ablation, and apply benchmark results are in the README
- [ ] CI is green with the fake LLM
- [ ] Demo recordings are in the README
- [ ] All ADRs are written
- [ ] Commits are logical and nothing has been pushed without my approval

---

## 12. Milestones (about 15 days; stop and report after each)

| # | Days | Deliverable |
|---|---|---|
| M1 | 1–3 | Repo skeleton, config, SQLite schema, AST chunker, FTS5 + vector index, incremental updates, `agent index` / `agent search`, retrieval unit tests |
| M2 | 4–6 | Gateway with the fake LLM, context assembly + budget pruning + symbol expansion, agent loop, streaming SEARCH/REPLACE parser, Fast Apply with atomic writes + undo, apply benchmarks |
| M3 | 7–10 | Shadow worktree + validation + auto-fix loop, diff review UI, command classifier + approvals, path/secret/injection controls, audit log, cancellation, security test suite |
| M4 | 11–13 | Headless Docker mode, SWE-bench harness integration (10 tasks first, then the full subset), retrieval ablations, pipeline ablations |
| M5 | 14–15 | CI, demo recordings, README, ADRs, final cleanup pass |
