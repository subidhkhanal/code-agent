# ADR 0008: Model routing by role, and per-task budgets enforced in code

- Status: accepted
- Date: 2026-09-28
- Milestone: M2

## Context

A task makes several model calls of very different difficulty. Turning "fix the bug where
expired tokens are accepted" into three search queries is easy. Writing a correct edit to
unfamiliar code is hard. Sending both to the strongest model wastes money; sending both to the
cheapest wastes the task. Separately, an agent loop can run away (tool-call ping-pong,
repeated rejected edits, a huge file read into every turn), so spend has to be bounded by
something the model cannot talk its way around.

## Decision: route by role, not by model name

- Code asks the gateway for a **role**: `cheap` (query rewriting, and later planning) or `strong`
  (the edit loop). Config maps each role to an ordered list of `provider:model` routes; later
  entries are fallbacks.
- **No model name exists in the code.** Routes come from the user's config and are checked
  against the provider's live model list before first use (`verify_models`), so a retired or
  misspelled model fails at startup with the list of available models, not mid-task. That
  check also yields each model's context window, which sizes the prompt budget.
- The provider is Google Gemini (the user's choice), over its REST API, with a scripted fake
  provider for tests, CI, and demos. Adding a provider is one adapter; the loop never changes.

## Decision: retries and fallback live in the gateway

- Retryable failures (timeouts, connection errors, 408/429/5xx) back off exponentially with
  **full jitter** (`uniform(0, min(cap, base · 2^attempt))`). Jitter spreads retries so many
  clients don't hammer a recovering provider in lockstep.
- Non-retryable failures (400, 401/403, 404) skip straight to the next route. Retrying a bad
  request only burns time.
- A call that already **streamed part of its answer is not retried transparently**: the user
  has seen partial text, and silently restarting would duplicate it. The error surfaces to the
  loop instead.
- When every route fails, the task ends with "generation unavailable", while `index`, `search`,
  and `undo` keep working (tested).

## Decision: budgets are checked in code, before every model and tool call

Per task (user config, `[budgets]`): max tokens, max USD, max tool calls, max wall-clock seconds,
and max rejected-edit rounds. `Meter.check()` runs before every model call and every tool call.
Exhausting a budget ends the task with a reason and whatever partial result exists. These
numbers live in config and code only. The prompt doesn't mention them, and no model output can
change them.

**Unknown prices are reported as unknown, not guessed.** Model prices come from config
(`[llm.pricing]`). A call to an unpriced model makes the task's cost `None`: the status line
says "cost unknown", and the USD budget can't be enforced for that task. The token budget still
bounds spend. The alternative, a built-in price table, would silently go stale.

## Decision: the prompt budget is bounded by cost, not just the window

The plan says to cap context at about 80% of the model window. With Gemini windows around 1M
tokens, that alone would permit roughly 800k-token prompts, which are technically valid and
financially absurd for a bug fix. The budget is therefore
`min(max_prompt_tokens, 0.8 · window − max_output_tokens)`, and retrieved code gets at most
`max_context_tokens` (default 24k) of that. As a conversation grows, the oldest tool outputs are
elided first. The system prompt, the task with its retrieved code, and the latest turn are always
kept (tested).

## Consequences

- `cheap` currently does only query rewriting. When it's unavailable, retrieval falls back to
  the raw request, so a missing cheap route degrades quality, not availability.
- Token counts in the budget are estimates (about 3.5 characters per token), because there's no
  local Gemini tokenizer. Provider-reported usage, which is authoritative, is what the token and
  cost budgets count.
- Model choices and their measured effect on cost and resolve rate are reported in M4, per
  route, rather than asserted here.
