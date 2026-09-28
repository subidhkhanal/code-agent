"""Local embeddings. Code never leaves the machine to be indexed.

`FastEmbedEmbedder` runs an ONNX model on CPU via fastembed. The weights are downloaded once into
`model_cache_dir` (that download is the only network access indexing ever does).
`HashingEmbedder` is a deterministic bag-of-words stand-in for tests: no download, and
similarity still tracks token overlap, so retrieval tests stay meaningful.
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol

import numpy as np

from code_agent.hashing import sha256_text
from code_agent.index.tokenize import split_identifiers


class EmbeddingUnavailableError(RuntimeError):
    """The embedding model cannot be loaded (not downloaded and offline, corrupt cache, ...)."""


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def embedding_text(
    file_path: str, symbol: str | None, kind: str, content: str, max_chars: int
) -> str:
    """What actually gets embedded for a chunk. The path and symbol give the model context that
    the code body alone often lacks (e.g. `auth/tokens.py` for a function named `check`)."""
    header = f"# file: {file_path}\n"
    if symbol:
        header += f"# {kind}: {symbol}\n"
    return (header + content)[:max_chars]


def embed_hash(model_id: str, text: str) -> str:
    return sha256_text(f"{model_id}\x00{text}")


class FastEmbedEmbedder:
    def __init__(self, model_id: str, cache_dir: Path, batch_size: int = 16) -> None:
        self._model_id = model_id
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._model = None
        self._dim: int | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    def _load(self):
        if self._model is None:
            try:
                from fastembed import TextEmbedding

                self._cache_dir.mkdir(parents=True, exist_ok=True)
                self._model = TextEmbedding(self._model_id, cache_dir=str(self._cache_dir))
            except Exception as exc:  # fastembed raises a variety of errors for download/IO
                raise EmbeddingUnavailableError(
                    f"could not load embedding model {self._model_id!r}: {exc}"
                ) from exc
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = int(self.embed_query("dimension probe").shape[0])
        return self._dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        model = self._load()
        vectors = list(model.passage_embed(texts, batch_size=self._batch_size))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        model = self._load()
        return np.asarray(next(iter(model.query_embed(text))), dtype=np.float32)


_TOKEN = re.compile(r"[A-Za-z]+|\d+")


class HashingEmbedder:
    """Feature-hashed bag of lower-cased word pieces, L2-normalized. Tests only."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def model_id(self) -> str:
        return f"test/hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        for token in _TOKEN.findall(split_identifiers(text).lower()):
            bucket = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=4).digest(), "big")
            vec[bucket % self._dim] += 1.0
        norm = math.sqrt(float(vec @ vec))
        return vec / norm if norm else vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self._dim))

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)
