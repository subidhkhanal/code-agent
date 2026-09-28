"""Index persistence: chunk rows, the FTS5 (BM25) table and the sqlite-vec table, kept in sync.

Every per-file change goes through `replace_file` / `remove_file`, which update all three
structures inside the caller's transaction, so a crash never leaves BM25 and vectors disagreeing
about which chunks exist. `check_integrity` detects and repairs drift anyway (e.g. an index
copied mid-write, or a process killed between embedding batches).
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from code_agent.db import SchemaMismatchError, connect, get_meta, init_schema, set_meta, utcnow
from code_agent.index.chunker import Chunk
from code_agent.index.files import SourceFile
from code_agent.index.tokenize import camel_parts, split_identifiers
from code_agent.workspace import Workspace

log = logging.getLogger(__name__)

VEC_TABLE = "chunks_vec"


# Filesystems store mtimes at limited resolution (FAT: 2 s). A file modified within this window
# before we hashed it could change again without its (size, mtime) changing, so we don't trust
# the metadata fast path for it. Same idea as git's "racily clean" index entries.
RACY_WINDOW_NS = 2_000_000_000


@dataclass(frozen=True)
class FileRecord:
    file_path: str
    content_hash: str
    size: int
    mtime_ns: int
    hashed_at_ns: int

    def metadata_matches(self, size: int, mtime_ns: int) -> bool:
        """True if the file can be assumed unchanged without reading it."""
        return (
            self.size == size
            and self.mtime_ns == mtime_ns
            and mtime_ns < self.hashed_at_ns - RACY_WINDOW_NS
        )


@dataclass(frozen=True)
class PendingChunk:
    id: int
    file_path: str
    symbol: str | None
    kind: str
    content: str


def open_index(workspace: Workspace) -> tuple[sqlite3.Connection, list[str]]:
    """Open (or create) the workspace index. A corrupt or incompatible DB is moved aside and
    rebuilt from the workspace, which is always the source of truth. Returns (conn, repairs)."""
    workspace.ensure_state_dir()
    path = workspace.index_path
    repairs: list[str] = []
    try:
        conn = _open_checked(path)
    except (sqlite3.DatabaseError, SchemaMismatchError) as exc:
        quarantine = path.with_name(f"{path.name}.bad-{int(time.time())}")
        for suffix in ("", "-wal", "-shm"):
            side = Path(str(path) + suffix)
            if side.exists():
                shutil.move(side, Path(str(quarantine) + suffix))
        log.warning("index at %s unusable (%s); moved to %s and rebuilding", path, exc, quarantine)
        repairs.append(f"index was unusable ({exc}); rebuilt from scratch")
        conn = connect(path)
        init_schema(conn)
    return conn, repairs


def _open_checked(path: Path) -> sqlite3.Connection:
    conn = connect(path)
    try:
        init_schema(conn)
        status = conn.execute("PRAGMA quick_check").fetchone()[0]
        if status != "ok":
            raise sqlite3.DatabaseError(f"quick_check: {status}")
    except BaseException:
        conn.close()
        raise
    return conn


class IndexStore:
    def __init__(self, conn: sqlite3.Connection, repo_id: str) -> None:
        self.conn = conn
        self.repo_id = repo_id

    # -- repository ---------------------------------------------------------------------------

    def upsert_repository(self, root: Path, branch: str | None, revision: str | None) -> None:
        self.conn.execute(
            """INSERT INTO repositories(repo_id, workspace_root, branch, revision)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(repo_id) DO UPDATE SET
                 workspace_root = excluded.workspace_root,
                 branch = excluded.branch, revision = excluded.revision""",
            (self.repo_id, str(root), branch, revision),
        )

    def mark_indexed(self) -> None:
        self.conn.execute(
            "UPDATE repositories SET indexed_at = ? WHERE repo_id = ?", (utcnow(), self.repo_id)
        )

    # -- files --------------------------------------------------------------------------------

    def file_records(self) -> dict[str, FileRecord]:
        rows = self.conn.execute(
            "SELECT file_path, content_hash, size, mtime_ns, hashed_at_ns FROM indexed_files "
            "WHERE repo_id = ?",
            (self.repo_id,),
        )
        return {r["file_path"]: FileRecord(*r) for r in rows}

    def file_record(self, file_path: str) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT file_path, content_hash, size, mtime_ns, hashed_at_ns FROM indexed_files "
            "WHERE repo_id = ? AND file_path = ?",
            (self.repo_id, file_path),
        ).fetchone()
        return None if row is None else FileRecord(*row)

    def touch_file(self, source: SourceFile) -> None:
        """Content unchanged but mtime/size moved (e.g. `git checkout` of an identical file)."""
        self.conn.execute(
            "UPDATE indexed_files SET size = ?, mtime_ns = ?, hashed_at_ns = ? "
            "WHERE repo_id = ? AND file_path = ?",
            (source.size, source.mtime_ns, time.time_ns(), self.repo_id, source.rel_path),
        )

    def replace_file(
        self,
        source: SourceFile,
        content_hash: str,
        chunks: Sequence[Chunk],
        embed_hashes: Sequence[str],
    ) -> int:
        """Replace all chunks of one file. Returns how many vectors were reused.

        Vectors are reused per chunk when the embedded text is byte-identical to a chunk that
        existed before (same `embed_hash`). Editing one function in a 50-function file therefore
        re-embeds one chunk, not fifty.
        """
        reusable = self._vectors_by_embed_hash(source.rel_path)
        self._delete_chunks(source.rel_path)
        self.conn.execute(
            """INSERT INTO indexed_files(repo_id, file_path, content_hash, size, mtime_ns,
                                         hashed_at_ns, language, indexed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(repo_id, file_path) DO UPDATE SET
                 content_hash = excluded.content_hash, size = excluded.size,
                 mtime_ns = excluded.mtime_ns, hashed_at_ns = excluded.hashed_at_ns,
                 language = excluded.language, indexed_at = excluded.indexed_at""",
            (
                self.repo_id, source.rel_path, content_hash, source.size, source.mtime_ns,
                time.time_ns(), source.language, utcnow(),
            ),
        )  # fmt: skip
        reused = 0
        for index, (chunk, e_hash) in enumerate(zip(chunks, embed_hashes, strict=True)):
            vector = reusable.get(e_hash)
            cur = self.conn.execute(
                """INSERT INTO file_chunks(repo_id, file_path, chunk_index, symbol, name, kind,
                       start_line, end_line, content, content_hash, embed_hash, language,
                       last_modified, has_vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    self.repo_id, source.rel_path, index, chunk.symbol, chunk.name, chunk.kind,
                    chunk.start_line, chunk.end_line, chunk.content, chunk.content_hash, e_hash,
                    source.language, source.mtime_ns, int(vector is not None),
                ),
            )  # fmt: skip
            chunk_id = cur.lastrowid
            assert chunk_id is not None  # always set after a successful INSERT
            self._insert_fts(chunk_id, source.rel_path, chunk.symbol, chunk.content)
            if vector is not None:
                self.conn.execute(
                    f"INSERT INTO {VEC_TABLE}(chunk_id, embedding) VALUES (?, ?)",
                    (chunk_id, vector),
                )
                reused += 1
        return reused

    def remove_file(self, file_path: str) -> None:
        self._delete_chunks(file_path)
        self.conn.execute(
            "DELETE FROM indexed_files WHERE repo_id = ? AND file_path = ?",
            (self.repo_id, file_path),
        )

    def _chunk_ids(self, file_path: str) -> list[int]:
        rows = self.conn.execute(
            "SELECT id FROM file_chunks WHERE repo_id = ? AND file_path = ?",
            (self.repo_id, file_path),
        )
        return [r["id"] for r in rows]

    def _delete_chunks(self, file_path: str) -> None:
        ids = [(i,) for i in self._chunk_ids(file_path)]
        if not ids:
            return
        self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", ids)
        if self.has_vector_table():
            self.conn.executemany(f"DELETE FROM {VEC_TABLE} WHERE chunk_id = ?", ids)
        self.conn.execute(
            "DELETE FROM file_chunks WHERE repo_id = ? AND file_path = ?",
            (self.repo_id, file_path),
        )

    def _insert_fts(self, chunk_id: int, file_path: str, symbol: str | None, content: str) -> None:
        symbol_text = f"{symbol} {split_identifiers(symbol)}" if symbol else ""
        body = content + "\n" + " ".join(camel_parts(content))
        self.conn.execute(
            "INSERT INTO chunks_fts(rowid, symbol, path, body) VALUES (?, ?, ?, ?)",
            (chunk_id, symbol_text, file_path, body),
        )

    # -- vectors ------------------------------------------------------------------------------

    def has_vector_table(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (VEC_TABLE,)
        ).fetchone()
        return row is not None

    def _vectors_by_embed_hash(self, file_path: str) -> dict[str, bytes]:
        if not self.has_vector_table():
            return {}
        rows = self.conn.execute(
            f"""SELECT c.embed_hash, v.embedding FROM file_chunks c
                JOIN {VEC_TABLE} v ON v.chunk_id = c.id
                WHERE c.repo_id = ? AND c.file_path = ? AND c.has_vector = 1""",
            (self.repo_id, file_path),
        )
        return {r["embed_hash"]: bytes(r["embedding"]) for r in rows}

    def reconcile_embedding_model(self, model_id: str) -> bool:
        """Drop all vectors if they were made by a different model. Returns True if dropped."""
        stored = get_meta(self.conn, "embedding_model")
        if stored is None or stored == model_id:
            return False
        self.conn.execute(f"DROP TABLE IF EXISTS {VEC_TABLE}")
        self.conn.execute("UPDATE file_chunks SET has_vector = 0")
        self.conn.execute("DELETE FROM meta WHERE key IN ('embedding_model', 'embedding_dim')")
        return True

    def ensure_vector_table(self, model_id: str, dim: int) -> None:
        if not self.has_vector_table():
            self.conn.execute(
                f"CREATE VIRTUAL TABLE {VEC_TABLE} USING vec0("
                f"chunk_id INTEGER PRIMARY KEY, embedding float[{int(dim)}] distance_metric=cosine)"
            )
        set_meta(self.conn, "embedding_model", model_id)
        set_meta(self.conn, "embedding_dim", str(dim))

    def count_missing_vectors(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM file_chunks WHERE repo_id = ? AND has_vector = 0",
            (self.repo_id,),
        ).fetchone()[0]

    def chunks_missing_vectors(self, limit: int) -> list[PendingChunk]:
        rows = self.conn.execute(
            """SELECT id, file_path, symbol, kind, content FROM file_chunks
               WHERE repo_id = ? AND has_vector = 0 ORDER BY id LIMIT ?""",
            (self.repo_id, limit),
        )
        return [PendingChunk(*r) for r in rows]

    def store_vectors(self, chunk_ids: Sequence[int], vectors: np.ndarray) -> None:
        blobs = [np.asarray(v, dtype=np.float32).tobytes() for v in vectors]
        self.conn.executemany(
            f"INSERT OR REPLACE INTO {VEC_TABLE}(chunk_id, embedding) VALUES (?, ?)",
            list(zip(chunk_ids, blobs, strict=True)),
        )
        self.conn.executemany(
            "UPDATE file_chunks SET has_vector = 1 WHERE id = ?", [(i,) for i in chunk_ids]
        )

    # -- integrity ----------------------------------------------------------------------------

    def check_integrity(self) -> list[str]:
        """Repair drift between chunks, BM25 rows and vectors. Returns what was repaired."""
        repairs: list[str] = []
        c = self.conn

        orphan_chunks = c.execute(
            """DELETE FROM file_chunks WHERE repo_id = ? AND file_path NOT IN
               (SELECT file_path FROM indexed_files WHERE repo_id = ?)""",
            (self.repo_id, self.repo_id),
        ).rowcount
        if orphan_chunks:
            repairs.append(f"removed {orphan_chunks} chunks with no file record")

        chunk_ids = {r[0] for r in c.execute("SELECT id FROM file_chunks")}
        fts_ids = {r[0] for r in c.execute("SELECT rowid FROM chunks_fts")}
        if chunk_ids != fts_ids:
            c.execute("DELETE FROM chunks_fts")
            rows = c.execute("SELECT id, file_path, symbol, content FROM file_chunks").fetchall()
            for r in rows:
                self._insert_fts(r["id"], r["file_path"], r["symbol"], r["content"])
            repairs.append(f"rebuilt BM25 index ({len(rows)} chunks)")

        flagged = {r[0] for r in c.execute("SELECT id FROM file_chunks WHERE has_vector = 1")}
        if self.has_vector_table():
            vec_ids = {r[0] for r in c.execute(f"SELECT chunk_id FROM {VEC_TABLE}")}
            orphans = vec_ids - chunk_ids
            if orphans:
                c.executemany(
                    f"DELETE FROM {VEC_TABLE} WHERE chunk_id = ?", [(i,) for i in orphans]
                )
                repairs.append(f"removed {len(orphans)} orphan vectors")
            unflagged = (vec_ids & chunk_ids) - flagged
            if unflagged:
                c.executemany(
                    "UPDATE file_chunks SET has_vector = 1 WHERE id = ?", [(i,) for i in unflagged]
                )
            missing = flagged - vec_ids
        else:
            missing = flagged
        if missing:
            c.executemany(
                "UPDATE file_chunks SET has_vector = 0 WHERE id = ?", [(i,) for i in missing]
            )
            repairs.append(f"{len(missing)} chunks lost their vectors; queued for re-embedding")
        c.commit()
        return repairs

    def clear(self) -> None:
        """Drop all index data (chunks, BM25, vectors) but keep tasks, audit log and undo
        history, which live in the same database."""
        with self.conn:
            self.conn.execute("DELETE FROM chunks_fts")
            self.conn.execute("DELETE FROM file_chunks")
            self.conn.execute("DELETE FROM indexed_files")
            self.conn.execute(f"DROP TABLE IF EXISTS {VEC_TABLE}")
            self.conn.execute("DELETE FROM meta WHERE key IN ('embedding_model', 'embedding_dim')")

    # -- reporting ----------------------------------------------------------------------------

    def counts(self) -> dict[str, int]:
        c = self.conn
        return {
            "files": c.execute(
                "SELECT COUNT(*) FROM indexed_files WHERE repo_id = ?", (self.repo_id,)
            ).fetchone()[0],
            "chunks": c.execute(
                "SELECT COUNT(*) FROM file_chunks WHERE repo_id = ?", (self.repo_id,)
            ).fetchone()[0],
            "vectors": c.execute(
                "SELECT COUNT(*) FROM file_chunks WHERE repo_id = ? AND has_vector = 1",
                (self.repo_id,),
            ).fetchone()[0],
        }
