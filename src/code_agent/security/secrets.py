"""Secret detection and redaction for everything that leaves the machine.

Runs as the gateway's outbound filter (before the request log), so neither the LLM provider nor
our own logs ever receive a detected secret. Only the *kind* of each redaction is recorded,
never the value.

Detection layers, most specific first:
1. Known token formats (cloud keys, VCS tokens, LLM API keys, chat tokens, JWTs, private keys).
2. Credentials embedded in URLs (`scheme://user:password@host`).
3. Secret-looking assignments: a key named like password/secret/token/api_key assigned a literal.
4. High-entropy strings: long base64-ish literals with Shannon entropy above a threshold. Pure
   hex is exempt (git SHAs and content hashes are everywhere in code and almost never secret).

False positives cost a little context; false negatives leak a credential. Thresholds lean
toward redacting.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, replace

from code_agent.llm.types import Request, ToolCall

PLACEHOLDER = "[REDACTED:{kind}]"
PLACEHOLDER_PATTERN = re.compile(r"\[REDACTED:[a-z_]+\]")

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key", re.compile(
        r"-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----[\s\S]*?"
        r"(?:-----END (?:[A-Z]+ )?PRIVATE KEY-----|\Z)"
    )),
    ("cloud_access_key", re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})\b"
    )),
    ("gitlab_token", re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("google_token", re.compile(r"\bAQ\.[A-Za-z0-9_\-]{30,}")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
    ("stripe_key", re.compile(r"\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
)  # fmt: skip

_URL_CREDENTIALS = re.compile(
    r"(?P<prefix>\b[a-z][a-z0-9+.\-]*://[^\s:/@]+:)(?P<secret>[^\s@/]+)(?=@)"
)
_ASSIGNMENT = re.compile(
    r"""(?P<prefix>\b[\w.\-]*(?:password|passwd|pwd|secret|token|api[_\-]?key|access[_\-]?key|
         auth[_\-]?key|private[_\-]?key|client[_\-]?secret)[\w.\-]*["']?\s*[:=]\s*["'])
        (?P<secret>[^"'\s]{8,})(?=["'])""",
    re.IGNORECASE | re.VERBOSE,
)
_QUERY_SECRET = re.compile(
    r"(?P<prefix>[?&](?:access_?token|token|api_?key|key|secret|password|sig|signature)=)"
    r"(?P<secret>[^&#\s\"']{8,})",
    re.IGNORECASE,
)
_URL = re.compile(
    r"\b[a-z][a-z0-9+.\-]*://[^\s\"'<>)]+|\b(?:www\.)?[a-z0-9\-]+\.(?:org|com|io|net)/[^\s\"'<>)]+",
    re.IGNORECASE,
)
_CANDIDATE = re.compile(r"[A-Za-z0-9+/_\-=]{32,}")
_HEX = re.compile(r"[0-9a-fA-F]+")
ENTROPY_THRESHOLD = 4.2  # bits/char; random base64 is ~6, English-like identifiers are ~3-4


def shannon_entropy(text: str) -> float:
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


@dataclass(frozen=True)
class Finding:
    kind: str
    start: int
    end: int


def _is_wordish(token: str) -> bool:
    """Identifiers and paths built from short word segments
    (`test_should_do_markup_PY_COLORS_eq_1`, `pytest-dev/2019-May`) mix character classes like a
    random token does, but a random token does not split into mostly alphabetic segments."""
    segments = [s for s in re.split(r"[_\-/.]+", token) if s]
    if len(segments) < 3:
        return False
    plain = sum(1 for s in segments if s.isalpha() or s.isdigit())
    return plain / len(segments) >= 0.7 and max(map(len, segments)) <= 16


def _looks_random(token: str) -> bool:
    if _HEX.fullmatch(token) or _is_wordish(token):
        return False
    classes = sum(bool(re.search(p, token)) for p in (r"[a-z]", r"[A-Z]", r"[0-9]"))
    return classes >= 3 and shannon_entropy(token) >= ENTROPY_THRESHOLD


def scan(text: str) -> list[Finding]:
    """Non-overlapping findings, earliest-longest first."""
    found: list[Finding] = []
    for kind, pattern in _PATTERNS:
        found += [Finding(kind, m.start(), m.end()) for m in pattern.finditer(text)]
    found += [
        Finding("url_credentials", m.start("secret"), m.end("secret"))
        for m in _URL_CREDENTIALS.finditer(text)
    ]
    found += [
        Finding("assigned_secret", m.start("secret"), m.end("secret"))
        for m in _ASSIGNMENT.finditer(text)
        if not PLACEHOLDER_PATTERN.fullmatch(m.group("secret"))
    ]
    found += [
        Finding("url_query_secret", m.start("secret"), m.end("secret"))
        for m in _QUERY_SECRET.finditer(text)
    ]
    # URL paths mix letters, digits and slashes and look random to an entropy test
    # (`org/pipermail/pytest-dev/2019-May/004716`); credentials in URLs are covered by the
    # user:password and query-parameter rules above instead.
    url_spans = [(m.start(), m.end()) for m in _URL.finditer(text)]
    found += [
        Finding("high_entropy", m.start(), m.end())
        for m in _CANDIDATE.finditer(text)
        if _looks_random(m.group())
        and not any(start <= m.start() < end for start, end in url_spans)
    ]
    found.sort(key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[Finding] = []
    for f in found:
        if merged and f.start < merged[-1].end:
            continue  # overlaps an earlier (more specific or longer) finding
        merged.append(f)
    return merged


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted text, kinds redacted). Never returns or logs the secret values."""
    findings = scan(text)
    if not findings:
        return text, []
    parts: list[str] = []
    pos = 0
    for f in findings:
        parts.append(text[pos : f.start])
        parts.append(PLACEHOLDER.format(kind=f.kind))
        pos = f.end
    parts.append(text[pos:])
    return "".join(parts), [f.kind for f in findings]


def contains_placeholder(text: str) -> bool:
    return bool(PLACEHOLDER_PATTERN.search(text))


class OutboundRedactor:
    """Gateway outbound filter: redacts every message and tool-call argument."""

    def __init__(self) -> None:
        self.redactions: Counter[str] = Counter()  # kind -> count, for the audit log

    def _text(self, text: str) -> str:
        clean, kinds = redact(text)
        self.redactions.update(kinds)
        return clean

    def __call__(self, request: Request) -> Request:
        messages = []
        for m in request.messages:
            calls = tuple(
                ToolCall(c.id, c.name, {k: self._text(v) if isinstance(v, str) else v
                                        for k, v in c.arguments.items()})
                for c in m.tool_calls
            )  # fmt: skip
            messages.append(replace(m, content=self._text(m.content), tool_calls=calls))
        return replace(request, messages=tuple(messages))

    @property
    def applied(self) -> bool:
        return bool(self.redactions)
