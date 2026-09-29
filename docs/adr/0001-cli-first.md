# ADR 0001: A CLI first, with the engine as a library an IDE could wrap

- Status: accepted
- Date: 2026-09-28 (written up at M5)
- Milestone: M1

## Context

The same core (index, retrieval, edits, validation, approvals) could be delivered as a terminal
tool or as an editor extension. The interesting problems in this project are in the core, and
the time budget is about 15 days.

## Options

| | CLI | VS Code extension |
|---|---|---|
| UI work | a REPL, streamed text, a diff, a few prompts | webviews, inline diffs, editor decorations, extension packaging |
| Where it runs | any terminal, SSH, containers (headless mode reuses it directly) | inside one editor |
| Language | Python end to end | TypeScript UI + a Python backend over a protocol |
| Testability | `CliRunner` end-to-end tests with a scripted LLM | UI tests need an editor host |
| What users lose | inline diagnostics, click-to-accept hunks, editor-native undo | nothing, but it costs weeks |

## Decision

A CLI (`agent index | search | chat | run | undo | audit | models`) over an engine that has no
UI in it:

- `AgentSession` wires index, retrieval, gateway, tools and the loop, and exposes plain calls:
  `run_task()`, `apply()`, `decline()`.
- The loop reports progress as typed **events** (`RetrievalDone`, `ToolStarted`, `EditProposed`,
  `ValidationDone`, ...) to a callback. The CLI renders them. An extension would subscribe to
  the same events.
- Human decisions come in through injected callbacks: the approver for commands, and the
  review step for diffs.

## Evidence the boundary holds

The engine already has two front ends that share it unchanged:
- `agent chat` (interactive: approver = terminal prompt, review = diff + y/n/per-file)
- `agent run --headless` (autonomous: approver = auto-approve once, review = apply)

A VS Code extension would be a third: a thin TypeScript client talking to a small JSON-RPC
server around `AgentSession`, mapping events to webviews and review to the editor's diff view.
That's the stretch goal the plan mentions, and it isn't built.

## Consequences

- No inline, per-hunk review inside an editor. Review is per file in the terminal.
- Terminal output needs care: model- and repo-controlled text is escaped before printing, so it
  can't inject Rich markup (a real bug found by a test).
