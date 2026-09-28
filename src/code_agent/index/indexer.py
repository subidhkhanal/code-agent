"""Incremental indexing.

Change detection per file, cheapest check first:

1. size + mtime unchanged           -> skip without reading the file (unless the mtime is too
                                       close to when we last hashed it; see RACY_WINDOW_NS)
2. content hash unchanged           -> only refresh size/mtime (branch switch, `touch`, re-save)
3. content changed                  -> re-chunk; re-embed only chunks whose text changed
4. file gone / now ignored          -> drop its chunks, BM25 rows and vectors

Chunks are written (and BM25-searchable) first; embeddings are computed afterwards in committed
batches. If embedding is interrupted or the model is unavailable, the index is still usable for
keyword search and the next run embeds whatever is missing.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from code_agent.config import AgentConfig
from code_agent.hashing import sha256_bytes
from code_agent.index.chunker import chunk_file
from code_agent.index.embeddings import (
    Embedder,
    EmbeddingUnavailableError,
    embed_hash,
    embedding_text,
)
from code_agent.index.files import FileSelector, SourceFile, read_source_text
from code_agent.index.store import FileRecord, IndexStore
from code_agent.workspace import Workspace

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]  # (done, total)

EMBED_FETCH_SIZE = 1024  # rows pulled from SQLite per round, then sorted by length
EMBED_COMMIT_SIZE = 128  # rows embedded + committed together (unit of resumability)


@dataclass
class IndexStats:
    files_seen: int = 0
    added: int = 0
    updated: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: int = 0
    chunks_written: int = 0
    vectors_reused: int = 0
    vectors_embedded: int = 0
    seconds: float = 0.0
    repairs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    embedding_error: str | None = None

    @property
    def changed(self) -> int:
        return self.added + self.updated + self.removed


class Indexer:
    def __init__(
        self,
        workspace: Workspace,
        conn: sqlite3.Connection,
        cfg: AgentConfig,
        embedder: Embedder | None,
    ) -> None:
        self.workspace = workspace
        self.cfg = cfg
        self.embedder = embedder
        self.store = IndexStore(conn, workspace.repo_id)
        self.selector = FileSelector(workspace, cfg.index)
        # Vectors are tagged with the model that actually produced them.
        self.model_id = embedder.model_id if embedder else cfg.index.embedding_model

    @property
    def conn(self) -> sqlite3.Connection:
        return self.store.conn

    # -- public API ---------------------------------------------------------------------------

    def sync(self, *, embed: bool = True, progress: ProgressCallback | None = None) -> IndexStats:
        """Bring the index in line with the whole workspace."""
        started = time.perf_counter()
        stats = IndexStats()
        self._prepare(stats)

        # Rebuilt each full sync so edits to .agentignore / .gitignore take effect.
        self.selector = FileSelector(self.workspace, self.cfg.index)
        files = self.selector.scan()
        known = self.store.file_records()
        stats.files_seen = len(files)
        for source in files:
            self._sync_file(source, known.get(source.rel_path), stats)
        for gone in known.keys() - {f.rel_path for f in files}:
            self.store.remove_file(gone)
            stats.removed += 1
        self._finish(stats, embed, progress, started)
        return stats

    def update_paths(
        self,
        rel_paths: Iterable[str],
        *,
        embed: bool = True,
        progress: ProgressCallback | None = None,
    ) -> IndexStats:
        """Re-index just these paths (from the file watcher). Handles create/modify/delete."""
        started = time.perf_counter()
        stats = IndexStats()
        self._prepare(stats)

        paths = sorted({p.replace("\\", "/").lstrip("/") for p in rel_paths if p})
        candidates = [p for p in paths if self.selector.is_candidate(p)]
        ignored = self.selector.vcs_ignored(candidates)
        for path in paths:
            source = None
            if path in candidates and path not in ignored:
                source = self.selector.stat(path)
            record = self.store.file_record(path)
            if source is not None:
                stats.files_seen += 1
                self._sync_file(source, record, stats)
            elif record is not None:
                self.store.remove_file(path)
                stats.removed += 1
        self._finish(stats, embed, progress, started)
        return stats

    def embed_missing(self, stats: IndexStats, progress: ProgressCallback | None = None) -> None:
        total = self.store.count_missing_vectors()
        if total == 0 or self.embedder is None:
            return
        try:
            dim = self.embedder.dim  # loads the model
        except EmbeddingUnavailableError as exc:
            stats.embedding_error = str(exc)
            log.warning("%s; index is keyword-search only until embeddings succeed", exc)
            return
        self.store.ensure_vector_table(self.embedder.model_id, dim)
        self.conn.commit()

        done = 0
        max_chars = self.cfg.index.max_embed_chars
        while pending := self.store.chunks_missing_vectors(EMBED_FETCH_SIZE):
            texts = [
                embedding_text(p.file_path, p.symbol, p.kind, p.content, max_chars) for p in pending
            ]
            # The model pads every batch to its longest input, so a batch mixing a 3-line
            # function with a 200-line test pays for 200 lines of attention on every row.
            # Sorting by length keeps batches homogeneous.
            order = sorted(range(len(pending)), key=lambda i: len(texts[i]))
            for start in range(0, len(order), EMBED_COMMIT_SIZE):
                batch = order[start : start + EMBED_COMMIT_SIZE]
                try:
                    vectors = self.embedder.embed_documents([texts[i] for i in batch])
                except EmbeddingUnavailableError as exc:
                    stats.embedding_error = str(exc)
                    return
                self.store.store_vectors([pending[i].id for i in batch], vectors)
                self.conn.commit()  # durable per batch; an interrupted run resumes from here
                done += len(batch)
                stats.vectors_embedded += len(batch)
                if progress:
                    progress(done, total)

    # -- internals ----------------------------------------------------------------------------

    def _prepare(self, stats: IndexStats) -> None:
        stats.repairs.extend(self.store.check_integrity())
        if self.store.reconcile_embedding_model(self.model_id):
            stats.repairs.append("embedding model changed; all chunks will be re-embedded")
        branch, revision = self.workspace.git_revision()
        self.store.upsert_repository(self.workspace.root, branch, revision)

    def _finish(
        self, stats: IndexStats, embed: bool, progress: ProgressCallback | None, started: float
    ) -> None:
        self.store.mark_indexed()
        self.conn.commit()
        if embed:
            self.embed_missing(stats, progress)
        stats.seconds = time.perf_counter() - started

    def _sync_file(self, source: SourceFile, record: FileRecord | None, stats: IndexStats) -> None:
        if record and record.metadata_matches(source.size, source.mtime_ns):
            stats.unchanged += 1
            return
        try:
            data = source.abs_path.read_bytes()
        except OSError as exc:
            stats.errors.append(f"{source.rel_path}: {exc}")
            return
        content_hash = sha256_bytes(data)
        if record and record.content_hash == content_hash:
            self.store.touch_file(source)
            stats.unchanged += 1
            return
        text = read_source_text(data)
        if text is None:  # binary or not UTF-8
            if record:
                self.store.remove_file(source.rel_path)
                stats.removed += 1
            stats.skipped += 1
            return

        idx = self.cfg.index
        chunks = chunk_file(
            text, source.language, max_module_lines=idx.max_module_chunk_lines,
            window=idx.text_window_lines, overlap=idx.text_window_overlap,
        )  # fmt: skip
        hashes = [
            embed_hash(
                self.model_id,
                embedding_text(source.rel_path, c.symbol, c.kind, c.content, idx.max_embed_chars),
            )
            for c in chunks
        ]
        stats.vectors_reused += self.store.replace_file(source, content_hash, chunks, hashes)
        stats.chunks_written += len(chunks)
        if record:
            stats.updated += 1
        else:
            stats.added += 1
