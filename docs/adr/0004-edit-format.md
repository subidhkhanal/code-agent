# ADR 0004: SEARCH/REPLACE blocks as the edit format

- Status: accepted
- Date: 2026-09-28
- Milestone: M2

## Context

The model has to tell us how to change files. Whatever format it uses has to be (a) easy for a
model to produce correctly, (b) cheap in output tokens, which are the slowest and most expensive
tokens, (c) checkable before anything touches disk, and (d) parseable while it streams, so edits
can be validated as soon as each one is complete.

## Options

| Format | Output tokens | Typical failure | Detectable before writing? |
|---|---|---|---|
| Whole file | the entire file, even for a one-line fix | silently drops or "summarizes" untouched code (`# ... rest unchanged`) | only by diffing, and a dropped function looks like an intentional deletion |
| Unified diff | small | wrong line numbers and hunk counts; models are bad at counting | yes, but most failures are formatting, not intent |
| **SEARCH/REPLACE** | small (the changed region plus a little context) | quoted text doesn't match the file | **yes**, and the failure mode is precise: no match, or more than one match |

## Decision

The model emits blocks like:

```
path/to/file.py
<<<<<<< SEARCH
<exact lines currently in the file>
=======
<lines to put there instead>
>>>>>>> REPLACE
```

- The path goes on the line before the block. Common decoration (backticks, `**bold**`, `File:`)
  is stripped, and a block with no path reuses the previous block's path.
- An empty SEARCH creates a new file. On an existing file it is rejected (`EMPTY_SEARCH`).
  "Replace the whole file" is never implied.
- Blocks are parsed line by line as tokens stream in (`edits/parser.py`). A block is emitted the
  moment its `>>>>>>> REPLACE` line arrives. A test splits a sample response at every possible
  character position and checks the events are identical, so parsing cannot depend on how the
  provider fragments tokens.

## Why SEARCH/REPLACE

- **It fails loudly.** A SEARCH that doesn't match is a clear, local error we can feed back to
  the model with the closest actual lines ("similarity 0.78 at lines 40-45: ..."). Whole-file
  output fails silently. Unified-diff output fails on arithmetic the model shouldn't be doing.
- **Line numbers aren't needed.** The quoted text *is* the address. That's also what makes the
  stale-file check meaningful: the text must still be there, exactly once (ADR 0005).
- **Retrieval produces quotable text.** Chunks are verbatim slices of the file (ADR 0003), so the
  model can copy lines it was shown.

## Edits travel in the reply text, not in a `propose_edits` tool call

PLAN.md lists `propose_edits(blocks)` among the tools. It is implemented as the SEARCH/REPLACE
channel in the model's *streamed reply text*, not as a JSON function call:

- Function-call arguments arrive only when the call is complete, so they can't be parsed while
  streaming. The plan's requirement to parse blocks "as tokens arrive" rules them out.
- Putting code inside a JSON string means escaping every quote, backslash, and newline. That
  adds output tokens, and escaping mistakes silently change the code.

Everything else about the edit step is unchanged: blocks go through the same parser, the same
Fast Apply checks, and the same retry budget.

## Consequences

- A SEARCH or REPLACE body containing a line that is exactly `=======` (e.g. RST headings, merge
  markers in a fixture) will be mis-parsed. That's rare in Python source. The parser reports
  unexpected markers rather than guessing.
- The model must reproduce existing code exactly. The whitespace and fuzzy tiers in ADR 0005
  absorb small copy errors, and anything bigger comes back as feedback.
- Deleting a whole file is not expressible. That's deliberate: file deletion should go through
  an explicit, approved tool (M3), not a text convention.
