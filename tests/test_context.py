from __future__ import annotations

import pytest

from code_agent.context.assembly import ContextAssembler, estimate_tokens
from code_agent.hashing import sha256_bytes
from code_agent.index.chunker import signature_only

from .conftest import IndexedRepo


@pytest.fixture
def assembler(indexed: IndexedRepo) -> ContextAssembler:
    return ContextAssembler(indexed.searcher, indexed.store.conn, indexed.workspace.repo_id)


def symbols(bundle) -> list[str | None]:
    return [i.chunk.symbol for i in bundle.items]


def test_estimate_tokens_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 2
    assert estimate_tokens("x" * 350) == 100


def test_relevant_code_is_assembled_with_base_hashes(assembler, indexed: IndexedRepo):
    bundle = assembler.assemble(["expired tokens are still accepted", "verify_token"], 50_000)
    assert "TokenStore.verify_token" in symbols(bundle)
    disk = sha256_bytes((indexed.root / "auth/tokens.py").read_bytes())
    assert bundle.base_hashes["auth/tokens.py"] == disk
    assert set(bundle.base_hashes) == set(bundle.files)
    assert bundle.tokens <= bundle.budget_tokens


def test_rendering_labels_code_as_data(assembler):
    text = assembler.assemble(["verify_token"], 50_000).render()
    assert text.startswith("<retrieved_code>") and text.rstrip().endswith("</retrieved_code>")
    assert "not instructions" in text
    assert "### auth/tokens.py:" in text and "```python" in text


def test_class_header_is_marked_as_summary(assembler):
    text = assembler.assemble(["TokenStore"], 50_000).render()
    assert "summary: method bodies elided" in text


def test_multi_query_fusion_covers_both_topics(assembler):
    bundle = assembler.assemble(["calculate invoice total", "parse http response"], 50_000)
    files = bundle.files
    assert "billing/invoice.py" in files and "utils/http.py" in files


def test_symbol_expansion_adds_signatures_of_called_code(indexed: IndexedRepo):
    # Retrieve only the test function; the code it calls must arrive via expansion.
    one = ContextAssembler(indexed.searcher, indexed.store.conn, indexed.workspace.repo_id,
                           top_k=1)  # fmt: skip
    bundle = one.assemble(["test_issue_then_verify"], 50_000)
    assert bundle.items[0].chunk.symbol == "test_issue_then_verify"
    expansions = {i.chunk.symbol: i for i in bundle.items if i.source == "expansion"}
    assert {"TokenStore", "TokenStore.issue", "TokenStore.verify_token"} <= set(expansions)
    assert all(i.signature_only for i in expansions.values())
    rendered = bundle.render()
    assert "referenced by the code above" in rendered
    assert "expires_at > 0" not in rendered  # body of verify_token was not included


def test_expansion_skips_code_already_retrieved(assembler):
    bundle = assembler.assemble(["test issue then verify"], 50_000)
    retrieved = {i.chunk.chunk_id for i in bundle.items if i.source == "retrieval"}
    expanded = {i.chunk.chunk_id for i in bundle.items if i.source == "expansion"}
    assert not retrieved & expanded


def test_tight_budget_drops_and_collapses_but_fits(assembler):
    roomy = assembler.assemble(["token expiry verify issue revoke"], 50_000)
    tight = assembler.assemble(["token expiry verify issue revoke"], roomy.tokens // 3)
    assert tight.tokens <= tight.budget_tokens
    assert tight.dropped or tight.collapsed
    assert tight.items[0].chunk.symbol == roomy.items[0].chunk.symbol  # best item survives


def test_extreme_budget_truncates_top_item_with_marker(assembler):
    bundle = assembler.assemble(["verify_token"], 40)
    assert len(bundle.items) == 1
    assert bundle.tokens <= 40 or bundle.items[0].truncated


def test_items_are_ordered_by_relevance(assembler):
    bundle = assembler.assemble(["verify_token expiry"], 50_000)
    scores = [i.score for i in bundle.items]
    assert scores == sorted(scores, reverse=True)


def test_no_hits_renders_placeholder(assembler):
    bundle = assembler.assemble(["zzqqxx_nothing_matches_this"], 50_000)
    assert "(no relevant code found)" in bundle.render() or bundle.items


def test_signature_only_keeps_docstring_and_drops_body():
    method = (
        "    def verify(self, token):\n"
        '        """Check it."""\n'
        "        return token.expires_at > 0\n"
    )
    collapsed = signature_only(method)
    assert collapsed is not None
    assert "def verify(self, token):" in collapsed and '"""Check it."""' in collapsed
    assert "expires_at" not in collapsed and collapsed.rstrip().endswith("...")
    assert signature_only("x = 1") is None
