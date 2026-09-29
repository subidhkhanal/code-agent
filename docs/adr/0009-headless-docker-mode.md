# ADR 0009: Headless mode runs only in a container, behind an allowlisting egress proxy

- Status: accepted
- Date: 2026-09-29
- Milestone: M4

## Context

Autonomous mode removes the human from the approval loop. In interactive mode that human is the
main control over what runs (ADR 0007). Without them, the operating system has to provide the
isolation instead. Headless mode is also the engine for SWE-bench and for an external
orchestrator's `spawn_sandbox`, so it must be reproducible and callable as a unit.

## Decision

1. **Refuse to start outside a container.** Auto-approval is only enabled when a container is
   detected from filesystem markers (`/.dockerenv`, `/run/.containerenv`, container cgroups).
   An environment variable is not accepted, since anyone can set one. A hidden
   `--allow-outside-container` flag exists for tests only.
2. **Same engine, different approver.** Headless mode uses the chat session code unchanged,
   with an approver that grants each command *once*. Every execution therefore still has its own
   approval record and audit entry, and all the code-level controls (classification, path
   policy, env allowlist, budgets, redaction) still apply.
3. **Network: internal network + allowlisting egress proxy**, not `--network none`. The agent
   must reach the LLM API, but nothing else. The agent container sits on an `--internal` Docker
   network with no route out. A small `CONNECT` proxy (`code_agent/egress.py`) sits on both
   networks and tunnels only to allowlisted `host:port` pairs. TLS stays end to end, so the
   proxy sees host names, not content, and holds no key. Tests the agent runs therefore get no
   network either.

   | Option | Why not |
   |---|---|
   | `--network none` | the agent can't reach the model at all |
   | normal bridge network | tests and commands could reach anything; the plan asks for no network |
   | key held by the proxy (auth-injecting reverse proxy) | would require terminating TLS in the proxy; more code and a MITM certificate for little gain, since child processes already can't see the key |
4. **Container hardening**: uid 1000, read-only root filesystem with tmpfs scratch,
   `no-new-privileges`, CPU/memory/pids limits and a wall-clock timeout.
5. **For SWE-bench, the agent visits the task's image** rather than the other way around. Each
   task image already contains the repo at the base commit and a (often old, e.g. Python 3.6)
   environment with its dependencies. The agent runtime is a relocatable `/opt/agent`
   (python-build-standalone CPython 3.12 + venv) mounted with `--volumes-from`. The entrypoint
   copies `/testbed`, then drops from root to uid 1000 with `setpriv`.

## Findings from running it

- Headless in Docker with the real model, through the proxy, on the fixture repo: validated and
  applied; the proxy log shows only `generativelanguage.googleapis.com:443` tunnels.
- The first in-container run exposed that the image had no pytest *and* that the validator
  reported "pytest ok" when pytest couldn't start. Both were fixed (the latter with a regression
  test: a check that didn't run is never a pass).
- `git diff` omits untracked files, so a patch that adds a module would have lost it. New files
  are now marked intent-to-add first.
- The official SWE-bench harness misbehaves on a Windows host (CRLF eval scripts, cp1252
  patches). It now runs unchanged in a Linux container that drives the host Docker daemon.

## How this relates to cloud autonomous agents

A hosted autonomous agent (the "assign an issue, get a PR" kind) runs in the provider's
sandbox. Its trust boundary is the provider's infrastructure, it has only what's in the repo
and its configured secrets, and a round trip is minutes. This design has the same shape, but
runs locally:

| | this headless mode | cloud autonomous agent |
|---|---|---|
| Trust boundary | your Docker daemon; code and index never leave the machine except LLM calls | provider's VMs; the repo is cloned there |
| Local state | can operate on uncommitted work (mount the checkout) | sees what was pushed |
| Latency | container start in seconds; no clone | queueing + clone + environment build |
| Scale | one laptop, a few tasks at a time | elastic |
| Environment fidelity | whatever image you give it (e.g. SWE-bench's exact per-task image) | the provider's generic images or a configured setup script |

The engine is the same one the interactive CLI uses; only the approver and the isolation differ.
That's the main design point.
