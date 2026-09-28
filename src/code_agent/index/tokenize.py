"""Code-aware text helpers shared by BM25 indexing and query parsing.

SQLite's unicode61 tokenizer already splits `snake_case` and `a.b.c` (underscore and dot are
separators), but it keeps `parseHttpResponse` as one token. We therefore append the camelCase
pieces of each identifier to the indexed text, and split queries the same way.
"""

from __future__ import annotations

import re

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[A-Za-z0-9]+")
_DOTTED = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
_BACKTICKED = re.compile(r"`([^`]+)`")
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Small English stopword list: NL requests ("fix the bug where ...") otherwise match every chunk.
_STOPWORD_TEXT = """
a an and are as at be been but by can could do does for from has have how i if in into is it its
me my no not of on or our should so that the their them then there these this those to was we
were what when where which while who why will with would you your please fix bug issue make add
change update implement
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def split_identifiers(text: str) -> str:
    """`parseHTTPResponse` -> `parse HTTP Response`."""
    return _CAMEL_BOUNDARY.sub(" ", text)


def camel_parts(text: str) -> list[str]:
    """Extra tokens for BM25: the pieces of every camelCase identifier in `text`."""
    extra: list[str] = []
    for ident in set(_IDENT.findall(text)):
        pieces = split_identifiers(ident)
        if pieces != ident:
            extra.append(pieces)
    return extra


def query_terms(query: str) -> list[str]:
    """Lower-cased, de-duplicated, stopword-free terms of a free-text query."""
    seen: dict[str, None] = {}
    for word in _WORD.findall(split_identifiers(query.replace("_", " "))):
        w = word.lower()
        if len(w) > 1 and w not in STOPWORDS:
            seen.setdefault(w, None)
    return list(seen)


def fts_match_expression(query: str) -> str | None:
    """Safe FTS5 MATCH string. Each term is quoted (no FTS syntax injection) and OR-ed: natural
    language requests rarely have every word present in the relevant code, and BM25 ranking
    already rewards chunks that match more terms."""
    terms = query_terms(query)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def _looks_like_identifier(token: str, position: int) -> bool:
    if "_" in token.strip("_") or _CAMEL_BOUNDARY.search(token):
        return True
    # A capitalized word mid-sentence is usually a class name ("where is Indexer built");
    # the first word of a sentence is capitalized for grammar, not because it is code.
    return position > 0 and token[:1].isupper() and len(token) > 2


def extract_identifiers(query: str) -> list[str]:
    """Tokens in a query that are probably code symbols, for exact symbol lookup.

    Counts as an identifier: anything in backticks, dotted names (`TokenStore.verify`), names
    followed by `(`, snake_case / camelCase tokens, and capitalized words after the first word.
    A single-token query is always treated as a possible symbol (`agent search verify_token`).
    """
    found: dict[str, None] = {}
    for inner in _BACKTICKED.findall(query):
        for ident in _DOTTED.findall(inner) or _IDENT.findall(inner):
            found.setdefault(ident, None)
    for ident in _DOTTED.findall(query):
        found.setdefault(ident, None)
    for ident in _CALL.findall(query):
        found.setdefault(ident, None)
    tokens = _IDENT.findall(query)
    for position, token in enumerate(tokens):
        if _looks_like_identifier(token, position):
            found.setdefault(token, None)
    if len(tokens) == 1:
        found.setdefault(tokens[0], None)
    return [f for f in found if f.lower() not in STOPWORDS]
