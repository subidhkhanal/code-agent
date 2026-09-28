"""Hybrid search: exact symbol lookup + BM25 + vector similarity, merged with reciprocal rank
fusion (RRF).

Why three retrievers:
* symbol lookup nails queries that name the code ("where is `verify_token`"),
* BM25 handles rare identifiers and error strings that embeddings blur together,
* vectors handle paraphrase ("expired tokens are accepted" vs. `if exp < now`).

Why RRF: the three scores live on incomparable scales (exact/no-exact, negative BM25, cosine
distance). RRF only uses each retriever's *rank*, so no score normalization or weight tuning is
needed, and a chunk that several retrievers agree on rises to the top.

The retrievers run concurrently on separate read-only SQLite connections (WAL mode allows this).
The slow part is embedding the query, which overlaps with the SQL lookups.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from code_agent.config import RetrievalConfig
from code_agent.db import connect, get_meta
from code_agent.index.embeddings import Embedder, EmbeddingUnavailableError
from code_agent.index.store import VEC_TABLE
from code_agent.index.tokenize import extract_identifiers, fts_match_expression


class Mode(StrEnum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    VECTOR = "vector"
    SYMBOL = "symbol"


RETRIEVERS_FOR_MODE: dict[Mode, tuple[str, ...]] = {
    Mode.HYBRID: ("symbol", "bm25", "vector"),
    Mode.BM25: ("bm25",),
    Mode.VECTOR: ("vector",),
    Mode.SYMBOL: ("symbol",),
}

# Among exact symbol matches, prefer definitions that carry the most information.
_KIND_ORDER = "CASE kind WHEN 'class' THEN 0 WHEN 'function' THEN 1 WHEN 'method' THEN 2 ELSE 3 END"


class RetrieverUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True)
class SearchHit:
    chunk_id: int
    file_path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    content: str
    score: float
    ranks: dict[str, int]  # retriever -> 1-based rank in that retriever's list


@dataclass
class SearchResult:
    query: str
    mode: Mode
    hits: list[SearchHit]
    notes: list[str] = field(default_factory=list)  # degraded-mode explanations
    timings_ms: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    ranked_lists: Mapping[str, Sequence[int]], k: int = 60
) -> list[tuple[int, float]]:
    """score(d) = sum over lists of 1 / (k + rank(d)). Ties broken by id for determinism."""
    scores: dict[int, float] = defaultdict(float)
    for ids in ranked_lists.values():
        for rank, doc_id in enumerate(ids, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def _dedupe(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(ids))


class Searcher:
    def __init__(
        self,
        db_path: Path,
        repo_id: str,
        cfg: RetrievalConfig,
        embedder: Embedder | None,
    ) -> None:
        self.db_path = db_path
        self.repo_id = repo_id
        self.cfg = cfg
        self.embedder = embedder

    def search(self, query: str, *, k: int | None = None, mode: Mode = Mode.HYBRID) -> SearchResult:
        k = k or self.cfg.top_k
        n = max(self.cfg.candidates_per_retriever, k)
        result = SearchResult(query=query, mode=mode, hits=[])
        runners: dict[str, Callable[[str, int], list[int]]] = {
            "symbol": self._symbol,
            "bm25": self._bm25,
            "vector": self._vector,
        }
        names = RETRIEVERS_FOR_MODE[mode]

        ranked: dict[str, list[int]] = {}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {name: pool.submit(self._timed, runners[name], query, n) for name in names}
            for name, future in futures.items():
                try:
                    ids, elapsed_ms = future.result()
                except RetrieverUnavailableError as exc:
                    result.notes.append(f"{name} search unavailable: {exc}")
                    continue
                ranked[name] = _dedupe(ids)
                result.timings_ms[name] = elapsed_ms

        fused = reciprocal_rank_fusion(ranked, k=self.cfg.rrf_k)[:k]
        rank_of = {name: {cid: r for r, cid in enumerate(ids, 1)} for name, ids in ranked.items()}
        rows = self._load_chunks([cid for cid, _ in fused])
        for chunk_id, score in fused:
            row = rows.get(chunk_id)
            if row is None:  # deleted between retrieval and load (concurrent re-index)
                continue
            result.hits.append(
                SearchHit(
                    chunk_id=chunk_id, file_path=row["file_path"], symbol=row["symbol"],
                    kind=row["kind"], start_line=row["start_line"], end_line=row["end_line"],
                    content=row["content"], score=score,
                    ranks={n: rank_of[n][chunk_id] for n in rank_of if chunk_id in rank_of[n]},
                )
            )  # fmt: skip
        return result

    # -- retrievers ---------------------------------------------------------------------------

    @staticmethod
    def _timed(fn: Callable[[str, int], list[int]], query: str, n: int) -> tuple[list[int], float]:
        started = time.perf_counter()
        ids = fn(query, n)
        return ids, (time.perf_counter() - started) * 1000

    def _connect(self) -> sqlite3.Connection:
        return connect(self.db_path, read_only=True)

    def _symbol(self, query: str, n: int) -> list[int]:
        identifiers = extract_identifiers(query)
        if not identifiers:
            return []
        ids: list[int] = []
        conn = self._connect()
        try:
            for ident in identifiers:
                if "." in ident:
                    # Full qualified symbol, or a suffix of one at a "." boundary. substr() rather
                    # than LIKE, because `_` in identifiers is a LIKE wildcard.
                    sql = f"""SELECT id FROM file_chunks WHERE repo_id = ? AND (symbol = ?
                                OR substr(symbol, -length(?) - 1) = '.' || ?)
                              ORDER BY {_KIND_ORDER}, file_path LIMIT ?"""
                    params: tuple = (self.repo_id, ident, ident, ident, n)
                else:
                    sql = f"""SELECT id FROM file_chunks WHERE repo_id = ? AND name = ?
                              ORDER BY {_KIND_ORDER}, file_path LIMIT ?"""
                    params = (self.repo_id, ident, n)
                rows = conn.execute(sql, params).fetchall()
                if not rows and "." not in ident:
                    rows = conn.execute(
                        sql.replace("name = ?", "name = ? COLLATE NOCASE"), params
                    ).fetchall()
                ids.extend(r["id"] for r in rows)
        finally:
            conn.close()
        return ids[:n]

    def _bm25(self, query: str, n: int) -> list[int]:
        expression = fts_match_expression(query)
        if expression is None:
            return []
        conn = self._connect()
        try:
            # Column weights: symbol name > file path > body text.
            rows = conn.execute(
                """SELECT c.id FROM chunks_fts f JOIN file_chunks c ON c.id = f.rowid
                   WHERE chunks_fts MATCH ? AND c.repo_id = ?
                   ORDER BY bm25(chunks_fts, 4.0, 2.0, 1.0) LIMIT ?""",
                (expression, self.repo_id, n),
            ).fetchall()
        finally:
            conn.close()
        return [r["id"] for r in rows]

    def _vector(self, query: str, n: int) -> list[int]:
        if self.embedder is None:
            raise RetrieverUnavailableError("no embedding model configured")
        conn = self._connect()
        try:
            indexed_model = get_meta(conn, "embedding_model")
            if indexed_model is None:
                raise RetrieverUnavailableError("index has no embeddings yet (run `agent index`)")
            if indexed_model != self.embedder.model_id:
                raise RetrieverUnavailableError(
                    f"index was embedded with {indexed_model!r}, query model is "
                    f"{self.embedder.model_id!r} (run `agent index`)"
                )
            try:
                vector = np.asarray(self.embedder.embed_query(query), dtype=np.float32)
            except EmbeddingUnavailableError as exc:
                raise RetrieverUnavailableError(str(exc)) from exc
            rows = conn.execute(
                f"""SELECT chunk_id FROM {VEC_TABLE}
                    WHERE embedding MATCH ? AND k = ? ORDER BY distance""",
                (vector.tobytes(), n),
            ).fetchall()
        finally:
            conn.close()
        return [r["chunk_id"] for r in rows]

    def _load_chunks(self, ids: list[int]) -> dict[int, sqlite3.Row]:
        if not ids:
            return {}
        conn = self._connect()
        try:
            placeholders = ",".join("?" * len(ids))
            rows = conn.execute(
                f"""SELECT id, file_path, symbol, kind, start_line, end_line, content
                    FROM file_chunks WHERE id IN ({placeholders})""",
                ids,
            ).fetchall()
        finally:
            conn.close()
        return {r["id"]: r for r in rows}
