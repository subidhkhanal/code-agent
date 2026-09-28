# ADR 0005: Match tiers, fuzzy threshold, uniqueness rule, and stale-file policy

- Status: accepted
- Date: 2026-09-28
- Milestone: M2

## Context

A SEARCH block (ADR 0004) has to be located in the file. Models copy code almost, but not quite,
perfectly: trailing spaces go missing, a snippet gets re-indented, an identifier gets
mis-remembered. Being too strict wastes a round-trip (tokens, latency, money). Being too lenient
edits the wrong place, which is far worse, because the change can look plausible in review.

## Decision

### Three tiers, strictest first (`edits/matching.py`)

1. **Exact**: identical lines.
2. **Whitespace-tolerant**: trailing whitespace ignored, internal runs of whitespace collapsed,
   and one *consistent* indentation shift allowed across the whole block. The replacement gets
   the same shift, so a snippet the model dedented still lands at the right depth. An
   *inconsistent* shift (some lines moved, others not) is not treated as a whitespace
   difference.
3. **Fuzzy**: similarity `2·LCS / (len(a) + len(b))` ≥ **0.90** between the SEARCH and a window of
   the file with the same number of lines. Only allowed when the SEARCH has **at least 3
   non-blank lines**.

The first tier that finds any candidate decides. We never fall through to a looser tier to break
a tie.

### Uniqueness: two candidates means reject, never "pick the best"

- Exact or whitespace: more than one candidate gives `AMBIGUOUS_MATCH`, with the candidate
  line numbers.
- Fuzzy: if another *non-overlapping* window also scores ≥ 0.90, the result is
  `AMBIGUOUS_MATCH`, even if one candidate scores higher.

The error goes back to the model ("matches 14 places at lines ...; include more surrounding
lines") and counts toward the task's retry budget.

**Evidence that this matters.** In the apply benchmark's first version, the synthetic file was
many near-identical functions differing only by a number (`handler_3`, `handler_4`, ...). A
5-line SEARCH with one typo scored ≥ 0.90 against **14** windows. A "pick the highest score"
rule would have chosen correctly there, but only because the typo happened not to fall on the
distinguishing identifier. If it had, the edit would have landed in the wrong function and
still produced a reasonable-looking diff. Rejecting costs one retry; a wrong edit costs a bug.

### Why 0.90 and 3 lines

- Measured over 200 random 5-line blocks (median 161 characters) from the benchmark generator:
  a one-character typo scores **0.994–1.000**, and rewriting one of the five lines entirely scores
  **0.765–0.891** (median 0.850). 0.90 sits between the two, so it accepts copy errors and rejects
  "the model means different code". The margin on the rewrite side is thin (0.891), which is one
  more reason the uniqueness rule, not the threshold alone, is the real safety net.
- For 1–2 line blocks, a single edit moves similarity by 5–20% and there are usually many
  similar lines in a file (`return None`). Fuzzy matching them is guessing, so they must match
  exactly or whitespace-tolerantly. When they don't, the closest lines are still reported as a
  hint.
- These are starting values. The M4 pipeline ablation reports the first-try apply rate and the
  distribution of rejection reasons, and the thresholds should be revisited against that data,
  not intuition.

### Stale-file policy

- Every file the model sees (via retrieval or `read_file`) has its content hash recorded. An
  edit to a file whose current hash differs is rejected with `STALE_FILE`: **re-read and
  regenerate; never patch new content with an old plan.** The SEARCH text might still match,
  but the model's reasoning was about content that no longer exists.
- Editing a file the model never read is rejected with `NOT_READ`.
- The check runs twice: when planning, and again immediately before the atomic write (the user
  can save a file while the diff is on screen). The second check raises `WriteConflictError`
  and nothing is written.
- Undo uses the same idea in reverse. A file is restored only if it still has exactly the bytes
  the agent wrote. Otherwise undo refuses rather than discard the user's later edits.

## Performance

`benchmarks/bench_apply.py`, plan + atomic write (fsync + rename), 50 runs per cell, i5-13420H,
Windows 11:

| File lines | exact p50 / p95 | whitespace p50 / p95 | fuzzy p50 / p95 |
|---|---|---|---|
| 100 | 2.3 / 3.3 ms | 2.1 / 2.6 ms | 2.1 / 2.7 ms |
| 500 | 2.4 / 3.2 ms | 2.3 / 3.3 ms | **3.0 / 3.7 ms** |
| 1000 | 2.7 / 4.4 ms | 2.8 / 3.5 ms | 3.9 / 4.9 ms |

The first fuzzy implementation used `difflib.SequenceMatcher` and took **154 ms p50 at 500
lines**, over the 100 ms target. Scanning every window is inherent to the approach, so the fix
was the similarity kernel: `rapidfuzz.fuzz.ratio` computes the exact LCS ratio in C++ (difflib's
`ratio()` approximates the same quantity). On the same 500-line file it took 0.4 ms instead of
172 ms, with the same best window and score. Most of the remaining ~2 ms is fsync.

## Consequences

- Some correct-but-imprecise edits are rejected and cost a retry. That's the intended trade.
- Line endings, a missing final newline, a UTF-8 BOM, and permission bits are preserved per line
  and per file (tested). Non-UTF-8 files are rejected with `ENCODING` rather than re-encoded.
