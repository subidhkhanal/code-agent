# ADR 0010: Symbol resolution: index lookup plus jedi, not a pyright language server

- Status: accepted
- Date: 2026-09-28
- Milestone: M2

## Context

The agent needs symbol information in two places:

1. **Symbol expansion** during context assembly (ADR 0003): for the top retrieved chunks, add
   the signatures of what they call. This runs on *every* request, so it must be fast. It only
   adds signatures, so an imperfect guess costs a few tokens, not correctness.
2. **The `get_definition` / `get_references` tools**: the model asks explicitly, often right
   before editing ("who calls `verify_token`? will my change break them?"). Here precision
   matters more than speed.

## Options

| | jedi | pyright via LSP |
|---|---|---|
| Runtime | pure Python, in-process | Node.js language server process |
| Startup | none (library call) | seconds (server boot + project analysis) |
| Integration | a function call | JSON-RPC over stdio: initialize, didOpen, request/response lifecycle |
| Inference quality | good for untyped code: follows assignments (`store = TokenStore(); store.verify_token(...)`) | best with type annotations; excellent overall |
| Headless Docker (M4) | nothing extra | Node in the image and a server per task |

## Decision

- **Expansion uses the index's symbol table** (the `name` column from AST chunking): a
  microsecond SQL lookup, same file preferred, and names with more than three definitions
  skipped as too ambiguous to guess (`get`, `run`, `__init__`).
- **`get_definition` also uses the index.** It is the same table search already uses, so
  what the model can find and what it can open are consistent.
- **`get_references` uses jedi** (`Script.get_references(scope="project")`), seeded at the
  definition's exact line and column from the index. On the fixture repo it correctly finds
  `store.verify_token(...)` calls in another file by following `store = TokenStore()`, in about
  0.5 s. Results outside the workspace (stdlib, site-packages) and in sensitive paths are
  filtered out.

pyright stays in the project as the **type checker** for shadow validation (M3), where it runs
as a one-shot CLI with no server lifecycle to manage.

## Consequences

- Name-based lookup can show the wrong `verify` if two classes define one. The tools print
  every candidate with its file and qualified name, and the model can disambiguate with a
  qualified symbol (`TokenStore.verify`).
- jedi's references are inference-based. Dynamic dispatch, `getattr`, and heavy
  metaprogramming can hide call sites, so the tool's answer is "likely complete", not "proven
  complete". Targeted tests in the shadow workspace (M3) are the real backstop.
- If M4's evaluation shows reference misses causing failed tasks, a pyright LSP adapter can
  sit behind the same tool interface without touching the agent loop.
