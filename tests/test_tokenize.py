import sqlite3

import pytest

from code_agent.index.tokenize import (
    camel_parts,
    extract_identifiers,
    fts_match_expression,
    query_terms,
    split_identifiers,
)


def test_split_identifiers():
    assert split_identifiers("parseHttpResponse") == "parse Http Response"
    assert split_identifiers("HTTPError") == "HTTP Error"
    assert split_identifiers("snake_case") == "snake_case"


def test_camel_parts_only_for_camel_identifiers():
    assert camel_parts("def calculateTotal(self): x_y = 1") == ["calculate Total"]


def test_query_terms_drop_stopwords_and_split():
    assert query_terms("Fix the bug where expired tokens are still accepted") == [
        "expired",
        "tokens",
        "still",
        "accepted",
    ]
    assert query_terms("calculateTotal") == ["calculate", "total"]
    assert query_terms("verify_token") == ["verify", "token"]


@pytest.mark.parametrize(
    "hostile",
    ['foo" OR "bar', "NEAR(a b)", "a* AND b", "col:value", "^start", '"', "(", "-x"],
)
def test_fts_expression_is_always_valid_and_quoted(hostile: str):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
    conn.execute("INSERT INTO t VALUES ('foo bar start value near')")
    expression = fts_match_expression(hostile)
    if expression is not None:
        conn.execute("SELECT * FROM t WHERE t MATCH ?", (expression,)).fetchall()  # no error


def test_fts_expression_none_for_stopword_only_query():
    assert fts_match_expression("the of and") is None


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("verify_token", ["verify_token"]),
        ("where is `TokenStore` defined", ["TokenStore"]),
        ("TokenStore.verify_token is wrong", ["TokenStore.verify_token"]),
        ("calls parse() twice", ["parse"]),
        ("why does calculateTotal add tax", ["calculateTotal"]),
        ("Expired tokens are accepted", []),  # sentence-initial capital is not a symbol
        ("the Indexer skips files", ["Indexer"]),
        ("retry", ["retry"]),  # single token: always tried as a symbol
    ],
)
def test_extract_identifiers(query: str, expected: list[str]):
    found = extract_identifiers(query)
    for ident in expected:
        assert ident in found
    if not expected:
        assert found == []
