"""Locate a SEARCH block inside a file: exact, then whitespace-tolerant, then fuzzy.

All matching is line-based. A SEARCH block always covers whole lines, which is what the model
sees in its context (chunks are whole-line slices) and makes "which lines changed" unambiguous.

Tiers stop at the first tier that finds *any* candidate:

1. exact            identical lines
2. whitespace       same lines after collapsing internal whitespace and ignoring trailing
                    whitespace, with one *consistent* indentation shift across the block
                    (the model re-indented the snippet). The replacement gets the same shift.
3. fuzzy            LCS similarity >= threshold over a window of the same line count.
                    Only for blocks with >= 3 non-blank lines: short blocks are too generic
                    to match approximately without hitting the wrong place.

Uniqueness: two or more candidates in the winning tier is AMBIGUOUS, never "pick the first".
For fuzzy, a second non-overlapping window above the threshold is also AMBIGUOUS.
See docs/adr/0005-fuzzy-matching-and-stale-files.md for the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from rapidfuzz import fuzz

DEFAULT_FUZZY_THRESHOLD = 0.90
FUZZY_MIN_LINES = 3


class MatchKind(StrEnum):
    EXACT = "exact"
    WHITESPACE = "whitespace"
    FUZZY = "fuzzy"


@dataclass(frozen=True)
class Match:
    start: int  # 0-based index of first matched line
    end: int  # exclusive
    kind: MatchKind
    similarity: float
    indent_add: str = ""  # whitespace tier: prefix the file has that the SEARCH lacks
    indent_remove: str = ""  # whitespace tier: prefix the SEARCH has that the file lacks

    def reindent(self, lines: list[str]) -> list[str]:
        """Apply the same indentation shift to replacement lines."""
        if not (self.indent_add or self.indent_remove):
            return lines
        out = []
        for line in lines:
            if not line.strip():
                out.append(line)
            elif self.indent_add:
                out.append(self.indent_add + line)
            elif line.startswith(self.indent_remove):
                out.append(line[len(self.indent_remove) :])
            else:
                out.append(line.lstrip())
        return out


@dataclass(frozen=True)
class NoMatch:
    closest_start: int | None  # best fuzzy window, for feedback to the model
    closest_similarity: float


@dataclass(frozen=True)
class Ambiguous:
    kind: MatchKind
    starts: list[int]  # 0-based start lines of the competing candidates


MatchOutcome = Match | NoMatch | Ambiguous


def _leading(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _squash(line: str) -> str:
    return " ".join(line.split())


def _exact(lines: list[str], needle: list[str]) -> list[int]:
    n, first = len(needle), needle[0]
    return [
        i for i in range(len(lines) - n + 1) if lines[i] == first and lines[i : i + n] == needle
    ]


def _indent_shift(window: list[str], needle: list[str]) -> tuple[str, str] | None:
    """(add, remove) prefix that maps needle indentation onto the window, or None."""
    shift: tuple[str, str] | None = None
    for have, want in zip(window, needle, strict=True):
        if not have.strip() and not want.strip():
            continue
        if _squash(have) != _squash(want):
            return None
        ih, iw = _leading(have), _leading(want)
        if ih.endswith(iw):
            candidate = (ih[: len(ih) - len(iw)], "")
        elif iw.endswith(ih):
            candidate = ("", iw[: len(iw) - len(ih)])
        else:
            return None
        if shift is None:
            shift = candidate
        elif shift != candidate:
            return None  # inconsistent re-indentation: not a whitespace-only difference
    return shift


def _whitespace(lines: list[str], needle: list[str]) -> list[tuple[int, tuple[str, str]]]:
    n = len(needle)
    anchor = next((j for j, s in enumerate(needle) if s.strip()), None)
    if anchor is None:
        return []
    anchor_text = _squash(needle[anchor])
    found = []
    for i in range(len(lines) - n + 1):
        if _squash(lines[i + anchor]) != anchor_text:
            continue
        shift = _indent_shift(lines[i : i + n], needle)
        if shift is not None:
            found.append((i, shift))
    return found


def _fuzzy_scores(lines: list[str], needle: list[str]) -> list[tuple[float, int]]:
    """Similarity of every same-length window to the needle, as 2*LCS / (len(a) + len(b)).

    rapidfuzz computes the exact LCS-based ratio in C++ (difflib's `ratio()` approximates the
    same quantity in pure Python and was ~400x slower here: 172 ms vs 0.4 ms on a 500-line file).
    Windows scoring below 50% are dropped; they are useless even as a "closest lines" hint.
    """
    n = len(needle)
    target = "\n".join(_squash(s) for s in needle)
    squashed = [_squash(s) for s in lines]
    scores = []
    for i in range(len(lines) - n + 1):
        score = fuzz.ratio("\n".join(squashed[i : i + n]), target, score_cutoff=50)
        if score:
            scores.append((score / 100, i))
    return scores


def find_block(
    lines: list[str], needle: list[str], *, fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD
) -> MatchOutcome:
    """Find `needle` (the SEARCH lines) in `lines` (the file). Lines exclude terminators."""
    n = len(needle)
    if n == 0 or n > len(lines):
        return NoMatch(None, 0.0)

    exact = _exact(lines, needle)
    if len(exact) == 1:
        return Match(exact[0], exact[0] + n, MatchKind.EXACT, 1.0)
    if exact:
        return Ambiguous(MatchKind.EXACT, exact)

    loose = _whitespace(lines, needle)
    if len(loose) == 1:
        start, (add, remove) = loose[0]
        return Match(start, start + n, MatchKind.WHITESPACE, 1.0, add, remove)
    if loose:
        return Ambiguous(MatchKind.WHITESPACE, [s for s, _ in loose])

    scores = sorted(_fuzzy_scores(lines, needle), reverse=True)
    if not scores:
        return NoMatch(None, 0.0)
    best_score, best = scores[0]
    eligible = sum(1 for s in needle if s.strip()) >= FUZZY_MIN_LINES
    if not eligible or best_score < fuzzy_threshold:
        return NoMatch(best, best_score)
    rivals = [i for s, i in scores[1:] if s >= fuzzy_threshold and abs(i - best) >= n]
    if rivals:
        return Ambiguous(MatchKind.FUZZY, [best, *rivals])
    return Match(best, best + n, MatchKind.FUZZY, best_score)
