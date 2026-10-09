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
from typing import Literal

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
    embeddings: bool = Field(
        default=True,
        description="Compute vector embeddings. Off = keyword + symbol search only (e.g. for "
        "large repos indexed from scratch inside a CPU-limited container).",
    )
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
    max_context_tokens: int = Field(
        default=24_000,
        ge=1_000,
        description="Cap on retrieved code per request. The real limit is min(this, share of the "
        "model window): with ~1M-token windows, cost binds long before the window does.",
    )


class ProviderConfig(_Strict):
    kind: Literal["gemini", "anthropic", "fake"] = "gemini"
    api_key_env: str | None = Field(
        default=None,
        description="Name of the env var holding the key (not the key). Default: GEMINI_API_KEY "
        "for gemini, ANTHROPIC_API_KEY for anthropic.",
    )
    base_url: str | None = Field(default=None, description="Override the provider's API URL")
    timeout_s: float = Field(default=120.0, gt=0)
    script: Path | None = Field(
        default=None, description="kind='fake' only: JSON file of scripted turns (demos, CI)"
    )
    # kind='anthropic' only. Defaults suit the current Claude models (see docs/adr/0011).
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = Field(
        default=None, description="anthropic: output_config.effort (None = the model's default)"
    )
    refusal_fallback: bool = Field(
        default=True,
        description="anthropic: let the API re-run a safety-declined request on its "
        "recommended fallback model instead of returning the refusal",
    )
    drop_mismatched_thinking: bool = Field(
        default=True,
        description="anthropic: when history trimming changes the conversation prefix, drop "
        "the invalidated thinking blocks instead of failing the request",
    )


class PriceConfig(_Strict):
    input_per_mtok: float = Field(ge=0, description="USD per million input tokens")
    output_per_mtok: float = Field(ge=0, description="USD per million output tokens")
    cache_read_per_mtok: float | None = Field(
        default=None, ge=0, description="Prompt-cache reads (unset: charged as normal input)"
    )
    cache_write_per_mtok: float | None = Field(
        default=None, ge=0, description="Prompt-cache writes (unset: 1.25x input)"
    )


class LLMConfig(_Strict):
    """No model names are built in: routes must be configured, and are checked against the
    provider's live model list before use (`agent models` lists what is available)."""

    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {"gemini": ProviderConfig()}
    )
    routes: dict[str, list[str]] = Field(
        default_factory=dict,
        description="role -> ['provider:model', <fallbacks>...]. Roles: 'cheap' (planning, "
        "query rewriting) and 'strong' (edits).",
    )
    pricing: dict[str, PriceConfig] = Field(
        default_factory=dict, description="'provider:model' -> price; unpriced cost is unknown"
    )
    max_attempts: int = Field(default=4, ge=1, le=10)
    base_delay_s: float = Field(default=1.0, ge=0)
    max_delay_s: float = Field(default=20.0, ge=0)
    context_fraction: float = Field(
        default=0.8, gt=0.1, le=0.95, description="Share of the context window for the prompt"
    )
    max_prompt_tokens: int = Field(
        default=64_000, ge=4_000, description="Cap on the whole prompt, incl. tool results"
    )
    max_output_tokens: int = Field(default=8_192, ge=256)


class BudgetConfig(_Strict):
    """Per-task limits. Enforced in code by the agent loop; nothing the model says changes them."""

    max_tokens: int = Field(default=400_000, ge=1_000)
    max_usd: float | None = Field(default=1.0, ge=0)
    max_tool_calls: int = Field(default=60, ge=1)
    max_seconds: int = Field(default=900, ge=10)
    max_edit_attempts: int = Field(default=3, ge=1, le=10)
    max_fix_attempts: int = Field(
        default=3, ge=1, le=10, description="Shadow-validation rounds before asking the user"
    )


class ValidationConfig(_Strict):
    """Shadow-workspace validation (ADR 0006)."""

    enabled: bool = True
    lint: bool = True
    type_check: bool = True
    tests: bool = Field(default=True, description="Targeted tests; asks for approval first")
    python: Path | None = Field(
        default=None, description="Interpreter for tests (default: the repo's .venv, if any)"
    )
    test_timeout_s: int = Field(default=300, ge=10)
    full_suite_max_test_files: int = Field(default=30, ge=0)


class AgentConfig(_Strict):
    index: IndexConfig = Field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budgets: BudgetConfig = Field(default_factory=BudgetConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
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
