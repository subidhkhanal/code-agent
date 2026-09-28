"""Configuration.

Trust boundary: configuration is read from a *user-level* file only
(`~/.config/code-agent/config.toml`, or the path in `CODE_AGENT_CONFIG`). There is deliberately no
repo-level config file, because a repository is untrusted input: a cloned repo must not be able to
change budgets, approval rules, or the sensitive-path policy. The one per-repo file honoured is
`.agentignore`, and it can only *exclude* more files from the index.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    # Typos in config keys should fail loudly instead of silently falling back to defaults.
    model_config = ConfigDict(extra="forbid")


# File types indexed with line-window chunking. Python gets AST chunking.
DEFAULT_TEXT_EXTENSIONS: tuple[str, ...] = (
    ".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yaml", ".yml", ".json",
    ".sh", ".sql", ".html", ".css", ".js", ".ts", ".tsx", ".jsx",
)  # fmt: skip
DEFAULT_TEXT_FILENAMES: tuple[str, ...] = ("Makefile", "Dockerfile", "README", "LICENSE")


class IndexConfig(_Strict):
    # Default chosen on measured CPU throughput; see docs/adr/0002. The code-trained alternative
    # is "jinaai/jina-embeddings-v2-base-code" (about 3-7x slower to index on CPU).
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="fastembed model id. Changing it triggers a re-embed on the next index run.",
    )
    embedding_batch_size: int = Field(default=16, ge=1, le=512)
    max_embed_chars: int = Field(
        default=1500,
        ge=200,
        description="Chunk text beyond this is truncated before embedding (path, symbol, "
        "signature and docstring come first, so they are always kept).",
    )
    max_file_bytes: int = Field(default=1_000_000, ge=1_000)
    text_window_lines: int = Field(default=60, ge=5)
    text_window_overlap: int = Field(default=10, ge=0)
    max_module_chunk_lines: int = Field(
        default=150, ge=10, description="Top-level code runs longer than this are windowed."
    )
    text_extensions: list[str] = Field(default_factory=lambda: list(DEFAULT_TEXT_EXTENSIONS))
    text_filenames: list[str] = Field(default_factory=lambda: list(DEFAULT_TEXT_FILENAMES))
    extra_ignore: list[str] = Field(default_factory=list, description="gitignore-style patterns")
    extra_sensitive: list[str] = Field(
        default_factory=list, description="Added to the built-in sensitive patterns (never removes)"
    )


class RetrievalConfig(_Strict):
    top_k: int = Field(default=10, ge=1, le=200)
    candidates_per_retriever: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, description="RRF damping constant (60 is the usual value)")


class AgentConfig(_Strict):
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "code-agent" / "models",
        description="Where embedding model weights are cached (downloaded once).",
    )


def user_config_path() -> Path:
    override = os.environ.get("CODE_AGENT_CONFIG")
    if override:
        return Path(override)
    return Path.home() / ".config" / "code-agent" / "config.toml"


def load_config(path: Path | None = None) -> AgentConfig:
    """Load config from `path` (or the user config path). Missing file means all defaults."""
    path = path or user_config_path()
    if not path.is_file():
        return AgentConfig()
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    return AgentConfig.model_validate(data)
