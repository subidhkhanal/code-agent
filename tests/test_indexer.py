from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import numpy as np

from code_agent.config import AgentConfig
from code_agent.index.embeddings import EmbeddingUnavailableError, HashingEmbedder
from code_agent.index.indexer import Indexer
from code_agent.index.store import VEC_TABLE, open_index
from code_agent.workspace import Workspace

from .conftest import IndexedRepo, git


def chunk_rows(ix: IndexedRepo, file_path: str) -> list[tuple]:
    return ix.store.conn.execute(
        "SELECT symbol, kind, start_line, end_line, has_vector FROM file_chunks "
        "WHERE file_path = ? ORDER BY chunk_index",
        (file_path,),
    ).fetchall()


def edit(path: Path, old: str, new: str) -> None:
    # Bytes, not text: write_text on Windows turns \n into \r\n, which silently changes the
    # file size (and made a same-size-edit test pass for the wrong reason).
    data = path.read_bytes()
    assert old.encode() in data
    path.write_bytes(data.replace(old.encode(), new.encode()))


def bump_mtime(path: Path, seconds: int = 10) -> None:
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def age_all_files(root: Path) -> None:
    """Pretend files were written long before indexing (outside the racy-mtime window)."""
    old = 1_600_000_000_000_000_000
    for path in root.rglob("*"):
        if path.is_file() and ".git" not in path.parts and ".agent" not in path.parts:
            os.utime(path, ns=(old, old))


def test_initial_sync_indexes_and_embeds_everything(indexed: IndexedRepo):
    counts = indexed.store.counts()
    assert counts["files"] == 8
    assert counts["chunks"] > 15
    assert counts["vectors"] == counts["chunks"]
    symbols = {r[0] for r in chunk_rows(indexed, "auth/tokens.py")}
    assert {"Token", "TokenStore", "TokenStore.verify_token", "TokenStore.issue"} <= symbols


def test_second_sync_changes_nothing(indexed: IndexedRepo):
    stats = indexed.indexer.sync()
    assert stats.changed == 0
    assert stats.unchanged == 8
    assert stats.chunks_written == 0 and stats.vectors_embedded == 0


def test_unchanged_files_are_not_even_read(indexed: IndexedRepo, monkeypatch):
    age_all_files(indexed.root)
    indexed.indexer.sync()  # records the aged mtimes

    def fail(*_a, **_k):
        raise AssertionError("file was read although size+mtime were unchanged")

    monkeypatch.setattr(Path, "read_bytes", fail)
    assert indexed.indexer.sync().unchanged == 8


def test_editing_one_method_reembeds_only_that_chunk(indexed: IndexedRepo):
    edit(
        indexed.root / "auth/tokens.py",
        "return token.expires_at > 0  # BUG: expiry is never compared with the current time",
        "return token.expires_at > time.time()",
    )
    stats = indexed.indexer.sync()
    assert stats.updated == 1 and stats.added == 0
    total = len(chunk_rows(indexed, "auth/tokens.py"))
    assert stats.vectors_embedded == 1
    assert stats.vectors_reused == total - 1
    assert all(row[4] == 1 for row in chunk_rows(indexed, "auth/tokens.py"))


def test_touch_without_content_change_is_not_reindexed(indexed: IndexedRepo):
    bump_mtime(indexed.root / "billing/invoice.py")
    stats = indexed.indexer.sync()
    assert stats.changed == 0 and stats.chunks_written == 0


def test_same_size_edit_within_racy_window_is_detected(indexed: IndexedRepo):
    """The racy case: the file was modified within the mtime-resolution window before it was
    hashed, then edited again without changing size or mtime. The fast path must not trust it.

    (This test used to pass on Windows by accident: NTFS doesn't restore a nanosecond mtime
    exactly, so the edit was always seen. On Linux the mtime is restored exactly.)"""
    path = indexed.root / "billing/invoice.py"
    now = time.time_ns()
    os.utime(path, ns=(now, now))  # modified "just now"...
    indexed.indexer.sync()  # ...and hashed immediately: within the racy window
    edit(path, 'TAX_RATE = Decimal("0.2")', 'TAX_RATE = Decimal("0.3")')  # same size
    os.utime(path, ns=(now, now))  # and exactly the same mtime
    assert indexed.indexer.sync().updated == 1


def test_same_size_same_mtime_edit_outside_window_is_a_known_limit(indexed: IndexedRepo):
    """Outside the racy window, an edit that keeps size and mtime identical is not seen, by
    design (like git's index). Documented, not a bug: normal editors always move the mtime."""
    path = indexed.root / "billing/invoice.py"
    old = 1_600_000_000_000_000_000
    os.utime(path, ns=(old, old))
    indexed.indexer.sync()
    edit(path, 'TAX_RATE = Decimal("0.2")', 'TAX_RATE = Decimal("0.3")')
    os.utime(path, ns=(old, old))
    assert indexed.indexer.sync().updated == 0


def test_deleted_file_removes_chunks_bm25_rows_and_vectors(indexed: IndexedRepo):
    conn = indexed.store.conn
    ids = [r[0] for r in conn.execute("SELECT id FROM file_chunks WHERE file_path='utils/http.py'")]
    assert ids
    (indexed.root / "utils/http.py").unlink()
    stats = indexed.indexer.sync()
    assert stats.removed == 1
    marks = ",".join("?" * len(ids))

    def count(sql: str) -> int:
        return conn.execute(sql.format(marks=marks), ids).fetchone()[0]

    assert count("SELECT COUNT(*) FROM chunks_fts WHERE rowid IN ({marks})") == 0
    assert count(f"SELECT COUNT(*) FROM {VEC_TABLE} WHERE chunk_id IN ({{marks}})") == 0


def test_rename_is_delete_plus_add(indexed: IndexedRepo):
    git(indexed.root, "mv", "utils/http.py", "utils/web.py")
    stats = indexed.indexer.sync()
    assert (stats.added, stats.removed) == (1, 1)
    assert chunk_rows(indexed, "utils/http.py") == []
    assert chunk_rows(indexed, "utils/web.py")


def test_branch_switch_is_picked_up(indexed: IndexedRepo):
    root = indexed.root
    git(root, "checkout", "-q", "-b", "feature")
    edit(root / "billing/invoice.py", "def apply_discount", "def apply_coupon")
    git(root, "commit", "-q", "-am", "rename")
    assert indexed.indexer.sync().updated == 1
    git(root, "checkout", "-q", "main")
    stats = indexed.indexer.sync()
    assert stats.updated == 1
    symbols = {r[0] for r in chunk_rows(indexed, "billing/invoice.py")}
    assert "apply_discount" in symbols and "apply_coupon" not in symbols
    revision = indexed.store.conn.execute("SELECT branch FROM repositories").fetchone()[0]
    assert revision == "main"


def test_new_sensitive_file_is_never_indexed(indexed: IndexedRepo):
    (indexed.root / ".env").write_text("OPENAI_API_KEY=sk-not-real\n")
    (indexed.root / "secrets").mkdir()
    (indexed.root / "secrets" / "keys.py").write_text("KEY = 'x'\n")
    indexed.indexer.sync()
    indexed.indexer.update_paths([".env", "secrets/keys.py"])
    paths = {r[0] for r in indexed.store.conn.execute("SELECT file_path FROM file_chunks")}
    assert ".env" not in paths and "secrets/keys.py" not in paths
    hits = indexed.store.conn.execute(
        "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH '\"sk\"'"
    ).fetchone()[0]
    assert hits == 0


def test_update_paths_handles_create_modify_delete_and_ignored(indexed: IndexedRepo):
    root = indexed.root
    (root / "new.py").write_text("def brand_new():\n    return 1\n")
    edit(root / "billing/invoice.py", "percent: int", "percent: float")
    (root / "utils/http.py").unlink()
    (root / "generated").mkdir()
    (root / "generated" / "gen.py").write_text("def generated():\n    pass\n")
    stats = indexed.indexer.update_paths(
        ["new.py", "billing/invoice.py", "utils/http.py", "generated/gen.py", "missing.py"]
    )
    assert (stats.added, stats.updated, stats.removed) == (1, 1, 1)
    assert chunk_rows(indexed, "generated/gen.py") == []


def test_index_without_embeddings_then_embed_later(repo: Path, cfg: AgentConfig):
    ws = Workspace.discover(repo)
    conn, _ = open_index(ws)
    embedder = HashingEmbedder()
    indexer = Indexer(ws, conn, cfg, embedder)
    stats = indexer.sync(embed=False)
    assert stats.vectors_embedded == 0 and indexer.store.counts()["vectors"] == 0
    stats = indexer.sync()
    assert stats.changed == 0
    assert stats.vectors_embedded == indexer.store.counts()["chunks"]
    conn.close()


class FlakyEmbedder(HashingEmbedder):
    """Works for `ok_calls` document batches, then reports the model as unavailable."""

    def __init__(self, ok_calls: int) -> None:
        super().__init__(dim=64)
        self.ok_calls = ok_calls

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if self.ok_calls <= 0:
            raise EmbeddingUnavailableError("model went away")
        self.ok_calls -= 1
        return super().embed_documents(texts)


def test_embedding_failure_leaves_keyword_index_usable_and_resumes(
    repo: Path, cfg: AgentConfig, monkeypatch
):
    monkeypatch.setattr("code_agent.index.indexer.EMBED_FETCH_SIZE", 5)
    ws = Workspace.discover(repo)
    conn, _ = open_index(ws)
    stats = Indexer(ws, conn, cfg, FlakyEmbedder(ok_calls=1)).sync()
    assert stats.embedding_error == "model went away"
    counts = Indexer(ws, conn, cfg, None).store.counts()
    assert counts["vectors"] == 5 and counts["chunks"] > 5  # first batch was committed

    stats = Indexer(ws, conn, cfg, HashingEmbedder()).sync()
    assert stats.embedding_error is None
    assert stats.vectors_embedded == counts["chunks"] - 5
    conn.close()


def test_changing_embedding_model_reembeds_everything(indexed: IndexedRepo, cfg: AgentConfig):
    total = indexed.store.counts()["chunks"]
    other = Indexer(indexed.workspace, indexed.store.conn, cfg, HashingEmbedder(dim=32))
    stats = other.sync()
    assert any("embedding model changed" in r for r in stats.repairs)
    assert stats.vectors_embedded == total


def test_integrity_check_repairs_drift(indexed: IndexedRepo):
    conn = indexed.store.conn
    total = indexed.store.counts()["chunks"]
    conn.execute("DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM file_chunks LIMIT 3)")
    conn.execute(f"DELETE FROM {VEC_TABLE} WHERE chunk_id IN (SELECT id FROM file_chunks LIMIT 2)")
    conn.commit()

    stats = indexed.indexer.sync()
    assert any("rebuilt BM25" in r for r in stats.repairs)
    assert any("lost their vectors" in r for r in stats.repairs)
    assert stats.vectors_embedded == 2
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == total


def test_corrupt_index_file_is_quarantined_and_rebuilt(repo: Path, cfg: AgentConfig):
    ws = Workspace.discover(repo)
    ws.ensure_state_dir()
    ws.index_path.write_bytes(b"this is not a sqlite database" * 100)
    conn, repairs = open_index(ws)
    assert repairs and "rebuilt" in repairs[0]
    assert any(p.name.startswith("index.db.bad-") for p in ws.state_dir.iterdir())
    stats = Indexer(ws, conn, cfg, HashingEmbedder()).sync()
    assert stats.added == 8
    conn.close()


def test_state_dir_ignores_itself(indexed: IndexedRepo):
    assert "?? .agent" not in git(indexed.root, "status", "--porcelain")


def test_copying_repo_elsewhere_gets_a_distinct_repo_id(repo: Path, tmp_path: Path):
    copy = tmp_path / "copy"
    shutil.copytree(repo, copy)
    assert Workspace.discover(repo).repo_id != Workspace.discover(copy).repo_id
