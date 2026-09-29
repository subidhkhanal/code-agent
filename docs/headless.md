# Headless mode and the `spawn_sandbox` contract

Headless mode runs one task end to end with no human: retrieve, edit, validate in the shadow
workspace, apply, and report. It exists for two consumers: the SWE-bench evaluation, and an
external orchestrator (for example a multi-agent system's `spawn_sandbox` tool) that wants a
disposable coding agent.

## Two ways to call it

### 1. From Python (host side): `code_agent.sandbox.spawn_sandbox`

```python
from pathlib import Path
from code_agent.sandbox import SandboxConfig, spawn_sandbox

report = spawn_sandbox(
    repo=Path("/path/to/checkout"),        # mounted read-write at /work
    task="fix the bug where expired tokens are still accepted",
    out_dir=Path("/tmp/agent-run-1"),      # receives report.json, patch.diff, container.log
    cfg=SandboxConfig(),                   # image, limits, network names, API key env var
    config_file=Path("agent.toml"),        # routes, prices, budgets (mounted read-only)
)
```

It creates (idempotently) an internal Docker network and the egress proxy, runs the agent image
on that network, waits (bounded by `SandboxConfig.timeout_s`), and returns the parsed report.
The patch is `out_dir/patch.diff`. The repository itself is also modified in place.

### 2. From inside a container: the CLI

```bash
agent run --task "..." --headless --auto-approve -p /work --out /out
```

This refuses to start unless it detects a container (`/.dockerenv`, `/run/.containerenv`, or a
container cgroup). An environment variable is deliberately *not* accepted as proof.

## Outputs

`report.json` (schema version 1, `code_agent.headless.HeadlessReport`):

| field | meaning |
|---|---|
| `status` | `SUCCEEDED`, `FAILED` or `CANCELLED` |
| `applied` | whether edits were written to the repo |
| `message` | why the task ended (e.g. `edits validated`, `token budget exhausted (...)`) |
| `files_changed` | workspace-relative paths |
| `input_tokens`, `output_tokens`, `cost_usd`, `llm_calls`, `tool_calls` | usage; `cost_usd` is `null` when a model has no configured price |
| `edit_attempts`, `fix_attempts`, `rejections` | rejected edit rounds, shadow-validation rounds, Fast Apply rejection codes |
| `validation` | the last shadow validation: `ok`, summary, new diagnostics, regressions, fixed and still-failing tests, skipped checks and why |
| `events` | a compact trace of retrieval, tool calls, edits and validations |

`patch.diff` is `git diff` of the workspace after the task, including new files. Exit code: 0
if `SUCCEEDED`, 1 otherwise, 3 if headless mode refused to start.

## Isolation (what the caller gets)

| | |
|---|---|
| Network | internal Docker network, no route out; HTTPS only via the egress proxy, which allows only the configured `host:port` (default `generativelanguage.googleapis.com:443`) |
| User | uid 1000, `no-new-privileges` |
| Filesystem | read-only root, tmpfs for `/tmp`; only the repo and the output dir are writable |
| Resources | `--cpus 2 --memory 4g --pids-limit 512`, plus a wall-clock timeout |
| Credentials | the API key is passed by name (`-e GEMINI_API_KEY`); child processes the agent starts get an allowlisted environment without it |
| Approvals | every command auto-approved *once* and audited in `/work/.agent/index.db` |

## Images

- `docker/agent.Dockerfile` → `code-agent:latest`: the agent plus a baked-in embedding model and
  pyright runtime (no downloads at run time). Also runs the egress proxy.
- `docker/runtime.Dockerfile` → `code-agent-runtime:latest`: the same agent as a relocatable
  `/opt/agent` directory, mounted into *other* images (each SWE-bench task image) so the task's
  own Python and dependencies are used unchanged.
- `docker/harness.Dockerfile` → the official SWE-bench harness, run on Linux.
