# ADR 0006: Validate edits in a shadow git worktree before the user sees them

- Status: accepted
- Date: 2026-09-29
- Milestone: M3

## Context

A diff can look right and still break the build: an undefined name, a wrong attribute, a test
that used to pass. Showing such diffs wastes the user's review time, and so does applying them
and discovering the breakage afterwards. The fix is to check candidate edits *before* review,
but the checks must never touch the user's working tree. The user may be editing files, and
nothing reaches their files until they accept.

## Options

| | Shadow via `git worktree` | Copy the tree to a temp dir | Validate in place and revert |
|---|---|---|---|
| Setup cost | cheap: shares the object store, checks out HEAD | copies everything, including large files | none |
| Touches the user's files | no | no | **yes**: breaks the "untouched until accept" rule, and races with their editor |
| Includes uncommitted work | copied on top (below) | yes | yes |
| Cleanup | `git worktree remove` + prune | rmtree | revert (fragile after a crash) |

## Decision

- A detached worktree of HEAD is created in the system temp directory, once per session, and
  reused across tasks. `reset()` restores it to the user's *current* state before every
  validation: checkout HEAD, reset, clean (untracked files only, so ignored caches survive),
  then **copy the user's uncommitted state on top**. Modified and untracked-but-not-ignored
  files are copied in, and deleted files are removed. The candidate change set is then written
  into the shadow.
- Checks, only on the files the change touches:
  - `ruff check` (lint) and `pyright` (type errors). Both are static, never execute repository
    code, and always run.
  - **Targeted tests**: test files named after a changed module or importing it, plus changed
    test files. If none match, the whole suite runs only if it's small (≤ 30 test files by
    default). Running tests executes repository code, including the model's edits, so it
    requires the same approval as any test/lint command (ADR 0007).
- **Only new problems fail validation.** The same checks first run on the unmodified shadow as a
  baseline. Lint and type errors are compared by (file, rule, message); line numbers are
  ignored because edits shift them. Tests are classified as **regression** (passed before, fails
  now), **still failing** (fails both before and after; shown, not blocking), or **fixed**.
  Without a baseline, any repository with existing lint debt could never validate.
- **Editable-install trap:** tests run with `PYTHONPATH` pointing at the shadow (and its
  `src/`), so an editable install of the project can't silently make tests import the user's
  real checkout instead of the edited shadow copy.
- **Bounded auto-fix loop:** on failure, the diagnostics (truncated) go back to the model, up to
  `max_fix_attempts` (default 3). The model's edits stay in an in-memory *overlay*: the next
  round's SEARCH/REPLACE blocks apply on top of them, and `read_file` shows the edited content,
  so the model sees a consistent state. The rounds are composed into one change set, from the
  original content to the final content. If validation still fails, the diff is shown *with* the
  failing diagnostics, and the user decides.
- Before the real write, base hashes are re-checked (the atomic write's phase 0), so a file the
  user changed during validation is never overwritten.

## Measured

On the fixture repo, a full validation (baseline plus after-change ruff, pyright, and one
targeted test file) takes about 5 s, dominated by pyright's startup. The end-to-end CLI tests run
real ruff, pyright, and pytest. A typo'd attribute (`time.tme()`) is caught by pyright and fixed
in the second round (`test_chat_auto_fixes_after_a_failed_validation`).

## Consequences and limits

- Non-git workspaces aren't validated (the diff says so). Copying arbitrary trees wasn't worth
  the complexity for a git-centric tool.
- Test selection is heuristic (names and imports). A change whose only coverage is through a
  distant integration test won't trigger that test. The full-suite fallback covers small
  projects; larger ones rely on the targeted set.
- Network isolation for test runs is only attempted on Linux (`unshare --net`). See ADR 0007
  for the per-OS table.
- A cancelled validation can leave the shadow dirty; the next `reset()` cleans it, and the
  worktree is removed when the session ends (a crash leaves one that is pruned next time).
