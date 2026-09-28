from __future__ import annotations

import pytest

from code_agent.index.embeddings import HashingEmbedder
from code_agent.retrieval.search import Mode, Searcher, reciprocal_rank_fusion

from .conftest import IndexedRepo


def symbols(result) -> list[str | None]:
    return [h.symbol for h in result.hits]


# -- RRF ------------------------------------------------------------------------------------


def test_rrf_scores_match_formula():
    fused = dict(reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1]}, k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)


def test_rrf_rewards_agreement_over_single_top_rank():
    # Doc 7 is only #1 in one list; doc 9 is #2 in both. Agreement wins.
    order = [d for d, _ in reciprocal_rank_fusion({"a": [7, 9], "b": [8, 9]})]
    assert order[0] == 9


def test_rrf_single_list_preserves_order_and_ties_break_by_id():
    assert [d for d, _ in reciprocal_rank_fusion({"a": [5, 3, 9]})] == [5, 3, 9]
    assert [d for d, _ in reciprocal_rank_fusion({"a": [4], "b": [2]})] == [2, 4]


def test_rrf_empty():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []


# -- individual retrievers ------------------------------------------------------------------


def test_symbol_search_exact_name(indexed: IndexedRepo):
    result = indexed.searcher.search("verify_token", mode=Mode.SYMBOL)
    assert symbols(result)[0] == "TokenStore.verify_token"


def test_symbol_search_qualified_and_backticked(indexed: IndexedRepo):
    # The qualified match ranks first; the owning class follows as useful context.
    assert symbols(indexed.searcher.search("TokenStore.issue", mode=Mode.SYMBOL)) == [
        "TokenStore.issue",
        "TokenStore",
    ]
    top = indexed.searcher.search("where is `TokenStore` built", mode=Mode.SYMBOL).hits[0]
    assert (top.symbol, top.kind) == ("TokenStore", "class")


def test_symbol_search_case_insensitive_fallback(indexed: IndexedRepo):
    assert symbols(indexed.searcher.search("calculatetotal", mode=Mode.SYMBOL)) == [
        "InvoiceCalculator.calculateTotal"
    ]


def test_symbol_search_with_underscore_is_not_a_like_wildcard(indexed: IndexedRepo):
    # `_` is a LIKE wildcard; "apply.discount" must not match "apply_discount" by accident.
    assert indexed.searcher.search("x.apply_discoun", mode=Mode.SYMBOL).hits == []


def test_bm25_splits_camel_case(indexed: IndexedRepo):
    result = indexed.searcher.search("calculate total", mode=Mode.BM25)
    assert result.hits[0].symbol == "InvoiceCalculator.calculateTotal"
    result = indexed.searcher.search("parse http response", mode=Mode.BM25)
    assert result.hits[0].symbol == "parseHttpResponse"


def test_bm25_natural_language(indexed: IndexedRepo):
    result = indexed.searcher.search("expired tokens are still accepted", mode=Mode.BM25)
    assert result.hits[0].file_path in ("auth/tokens.py", "tests/test_tokens.py", "README.md")
    assert any(h.symbol == "TokenStore.verify_token" for h in result.hits[:5])


@pytest.mark.parametrize("query", ['" OR 1=1 --', "NEAR(", "*", "col:x", "the and of"])
def test_bm25_hostile_or_empty_queries_do_not_raise(indexed: IndexedRepo, query: str):
    indexed.searcher.search(query, mode=Mode.BM25)


def test_vector_search_uses_embeddings(indexed: IndexedRepo):
    result = indexed.searcher.search("retry attempts connection error", mode=Mode.VECTOR)
    assert result.hits[0].symbol == "retry"
    assert result.notes == []


# -- hybrid + degraded mode ---------------------------------------------------------------


def test_hybrid_fuses_all_retrievers(indexed: IndexedRepo):
    result = indexed.searcher.search("verify_token expiry")
    top = result.hits[0]
    assert top.symbol == "TokenStore.verify_token"
    assert set(top.ranks) == {"symbol", "bm25", "vector"}
    assert set(result.timings_ms) == {"symbol", "bm25", "vector"}
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_respected(indexed: IndexedRepo):
    assert len(indexed.searcher.search("token", k=3).hits) == 3


def test_no_embedder_degrades_to_keyword_search(indexed: IndexedRepo):
    searcher = Searcher(
        indexed.workspace.index_path, indexed.workspace.repo_id, indexed.searcher.cfg, None
    )
    result = searcher.search("verify_token expiry")
    assert any("vector search unavailable" in n for n in result.notes)
    assert result.hits and result.hits[0].symbol == "TokenStore.verify_token"


def test_model_mismatch_is_reported_not_silently_wrong(indexed: IndexedRepo):
    searcher = Searcher(
        indexed.workspace.index_path,
        indexed.workspace.repo_id,
        indexed.searcher.cfg,
        HashingEmbedder(dim=32),  # index was built with dim=64
    )
    result = searcher.search("token", mode=Mode.VECTOR)
    assert result.hits == []
    assert "embedded with" in result.notes[0]


def test_sensitive_content_is_never_returned(indexed: IndexedRepo):
    (indexed.root / ".env").write_text("STRIPE_KEY=zzsupersecretzz\n")
    indexed.indexer.sync()
    for mode in Mode:
        hits = indexed.searcher.search("zzsupersecretzz STRIPE_KEY", mode=mode).hits
        assert all(h.file_path != ".env" and "zzsupersecretzz" not in h.content for h in hits)


def test_results_reflect_edits_after_reindex(indexed: IndexedRepo):
    path = indexed.root / "billing/invoice.py"
    path.write_text(path.read_text().replace("apply_discount", "apply_coupon"))
    indexed.indexer.update_paths(["billing/invoice.py"])
    assert indexed.searcher.search("apply_discount", mode=Mode.SYMBOL).hits == []
    assert symbols(indexed.searcher.search("apply_coupon", mode=Mode.SYMBOL)) == ["apply_coupon"]
