# ADR 0011: A hosted playground on Claude, with limits and key protection in code

- Status: accepted
- Date: 2026-10-09
- Milestone: post-M5

## Context

The project page showed recorded demos. Recordings prove little: the model's replies were
scripted so the recording was reproducible. The goal now is for visitors to run the real agent
themselves, from a browser, with no install and no API key of their own.

That turns a CLI into a public service, with three new problems:

1. **Money.** Every run spends the owner's prepaid API credit (USD 100). Strangers, scripts, and
   people who share the link all draw on the same balance.
2. **Untrusted code on the server.** Validation runs the sample repo's tests after the model's
   edits, so model-written code executes on the server. A visitor can ask the model to write a
   test that prints environment variables, reads `/proc`, or opens a socket.
3. **One secret worth stealing.** The server holds the API key. Everything else on the box is
   disposable.

The owner chose a single shared server over per-visitor sandboxes ("it's a resume demo"), and a
free host. So isolation between visitors and the key's protection must come from the process
itself, not from extra infrastructure.

## Decision

**Provider.** Claude via the official `anthropic` SDK (`llm/claude.py`), behind the existing
gateway. The model is configuration (`deploy/playground.toml`), checked against the live model
list at startup like every other route. The adapter:

- replays the assistant's raw content blocks (thinking blocks with signatures) verbatim;
- sets `prefix_mismatch_behavior: drop_block`, because the loop elides old tool output to fit
  the history budget, which changes the prefix that earlier thinking blocks were produced under;
- opts into the server-side refusal fallback (`fallbacks: "default"`); a final `refusal` ends the
  task cleanly instead of planning edits from partial output;
- turns prompt caching on; an agent task resends the same system prompt, tools and retrieved
  code on every turn, which then bill at the cache-read rate;
- leaves eager tool-input streaming off: tool inputs here are short (paths, queries), edits stream
  as text, and leaving it off keeps the API's own schema validation of tool inputs;
- turns the SDK's own retries off, so the gateway's retry, fallback and budget logic stay the
  single source of truth for every provider.

Cost accounting now distinguishes cache reads and writes. An unconfigured cache price is charged
at the *higher* rate, so a missing price can only overstate cost.

**Limits (`playground/guard.py`), all enforced before a model call:**

| Limit | Default | Why |
|---|---|---|
| Global daily budget | USD 3 | A shared link can't drain the credit: at most ~USD 90 a month even if every day is used up |
| Per-run reservation | 2 × the per-run cap | Concurrent runs reserve worst case up front, so they can't jointly overshoot the day |
| Per-run cost cap | USD 0.50 | Checked before every model call (existing `BudgetConfig.max_usd`) |
| Runs per visitor per day | 3 | Fairness; visitors are salted IP hashes, never stored raw |
| Concurrent runs | 2 | Bounds CPU on a free host |

Unknown cost is charged as the full reservation. Spend persists to a file so a restart doesn't
reset the day. A spend limit on the API key in the provider's console is the final backstop.

**What a run may do (`playground/engine.py`).** Each run copies a curated sample repo into a temp
dir, runs the normal engine, and deletes the copy. The approver allows test/lint commands only;
read-only shell commands (the model has file and search tools) and privileged commands are
denied, and there is no prompt a visitor could be talked into answering. Edits are validated and
shown as a diff, never applied anywhere persistent.

**Key protection (`playground/__main__.py`), in layers:**

1. The launcher reads the key from the environment, writes it into a pipe and `execve`s itself
   with an environment that no longer contains it. After `execve`, `/proc/<pid>/environ` shows
   only the new environment; the key exists in process memory alone.
2. The server calls `prctl(PR_SET_DUMPABLE, 0)`. Its `/proc/<pid>/environ` and `/proc/<pid>/mem`
   become root-owned, so processes of the same user (the tests it spawns) can't read them.
   Verified in the image: `cat /proc/1/environ` as uid 1000 gives "Permission denied"; a Linux CI
   test checks the same against a child process.
3. Commands get an allowlisted environment (existing `security/runner.py`).
4. Every byte streamed to the browser is scrubbed of the key by exact match, then by the outbound
   secret scanner.
5. Code and samples are installed as root; the app runs as uid 1000 and can't modify them.

**Hosting.** Hugging Face Spaces (Docker SDK, free CPU tier). The image is built from a folder
assembled from the repo (`deploy/huggingface/assemble.sh`), so the image tested locally is the
image that is deployed. A GitHub workflow uploads that folder after CI passes.

## Consequences

- Visitors run the real agent: real retrieval, real model, real ruff, pyright and pytest.
- The owner pays only for API usage, bounded per day in code and per month at the provider.
- **Not defended:** model-written code can still use the network and the CPU of the shared
  container, and can disturb a concurrent run (for example by filling `/tmp`). The container is
  disposable and restarts clean; the samples are tiny; runs are short and capped. Per-visitor
  sandboxes (the headless Docker mode of ADR 0009) are the path if this ever matters.
- **Per-visitor limits depend on the proxy.** `PLAYGROUND_PROXY_HOPS` picks the address the
  host's proxy appended to `X-Forwarded-For`. If it is set wrong, visitors share or can spoof an
  identity. The global budget does not depend on it.
- The free tier sleeps after two days without visitors; the first visit after that waits about a
  minute for the container to start.
