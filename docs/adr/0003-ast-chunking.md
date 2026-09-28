# ADR 0003: AST chunking instead of fixed line windows

- Status: accepted
- Date: 2026-09-28
- Milestone: M1

## Context

Retrieval returns *chunks*, not files. Chunk boundaries decide three things downstream:

1. **What the retriever can match.** A window that cuts a function in half splits its name from
   its body, so neither half matches a query well.
2. **What the model sees.** Context is paid for in tokens. Half a function forces the model to
   ask for the rest (another tool call), and it may never see the lines that matter.
3. **Whether edits apply.** The edit format (SEARCH/REPLACE, ADR 0004) requires the model to
   quote existing code exactly. If the chunk it saw is not a verbatim slice of the file, the
   quote will not match.

## Options

| Option | Pros | Cons |
|---|---|---|
| Fixed line windows (e.g. 60 lines, 10 overlap) | Trivial, language-agnostic | Cuts through definitions; overlap duplicates tokens; no symbol names for exact lookup |
| Python `ast` module | Stdlib, exact | Fails on any syntax error (common mid-edit); no byte/column precision for decorators and comments |
| **tree-sitter** | Error-tolerant (partial trees on broken code), fast, same API for other languages later | Native dependency; grammar version pinning needed |

## Decision

Use tree-sitter's Python grammar with these rules:

- **Top-level function**: one chunk, including decorators and comments directly above it (no
  blank line in between).
- **Class**: a *header* chunk (decorators, `class` line, docstring, class attributes, and each
  method's signature with the body replaced by `...`), plus **one chunk per method** with a
  qualified symbol such as `TokenStore.verify_token`. Nested classes recurse.
- **Module-level code** (imports, constants, `if __name__ == "__main__"`): grouped into
  *contiguous runs* between definitions. Each run is one chunk.
- Very long runs (more than 150 lines) and all non-Python text files use overlapping line windows.

Every chunk except class headers is a **verbatim, contiguous slice** of the file, and a test
checks this against every Python file in the repo and the fixtures
(`test_non_class_chunks_are_verbatim_slices_of_the_file`).

## Why these specific rules

- **Methods as their own chunks, classes as signature summaries.** A 600-line class as one
  chunk would blow the context budget. With this split, the header answers "what does this class
  offer?" cheaply, and the method chunk answers "how does this method work?". The header costs
  little and doubles as the signature-only fallback used by budget pruning (M2).
- **Contiguous module runs instead of one "imports and globals" chunk.** One module chunk would
  have to stitch together non-adjacent lines (imports at the top, `__main__` at the bottom), so
  its text would not exist anywhere in the file. A model quoting it in a SEARCH block would get
  `NO_MATCH`. Contiguous runs avoid that.
- **Attaching adjacent comments.** Comments right above a function usually describe it. A
  comment separated by a blank line usually describes the section, so it stays with the module
  code.
- **Qualified symbols** (`Class.method`) make exact-symbol lookup work for queries that name the
  code, which is the cheapest and most precise retriever.

## Consequences

- Chunks vary in size (a one-line function vs. a 300-line method). BM25 normalizes for length,
  embeddings truncate at `max_embed_chars`, and the context budget (M2) handles the rest.
- Class headers are synthetic (they contain `...`). They are labeled `kind = class`, so the
  context renderer can tell the model they are a summary and not quotable.
- Newlines are normalized to `\n` before parsing, so CRLF files chunk identically (tested).
  Fast Apply (M2) is responsible for writing the file's original line endings back.

## Incident: tree-sitter 0.26.0 segfault on Windows

py-tree-sitter **0.26.0** crashed the process with an access violation inside
`Node.child_by_field_name` after a few hundred parses on Windows. A pure-tree-sitter repro with
none of this project's code crashed too, while 0.25.2 ran 100 × 19 files cleanly. (0.24 cannot
load the ABI-15 grammar.) The dependency is pinned to `tree-sitter>=0.25,<0.26` and
`tree-sitter-python>=0.25,<0.26`. Revisit when a 0.26.x fix is released.

Separately, the chunker never uses `Node.text`. It slices names from its own copy of the source
bytes, which it keeps alive for the tree's lifetime. That removes a dependency on the binding's
buffer ownership, which is exactly where native crashes come from.
