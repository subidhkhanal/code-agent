from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from code_agent.config import AgentConfig
from code_agent.index.embeddings import HashingEmbedder
from code_agent.index.indexer import Indexer
from code_agent.index.store import IndexStore, open_index
from code_agent.retrieval.search import Searcher
from code_agent.workspace import Workspace

FIXTURES = Path(__file__).parent / "fixtures"


def git(root: Path, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return proc.stdout


def make_git_repo(root: Path) -> Path:
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    git(root, "config", "core.autocrlf", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "initial")
    return root


@pytest.fixture(scope="session")
def _repo_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # `git init` + commit costs ~1 s on Windows; build once, copy per test.
    root = tmp_path_factory.mktemp("template") / "repo"
    shutil.copytree(FIXTURES / "sample_repo", root)
    return make_git_repo(root)


@pytest.fixture
def repo(tmp_path: Path, _repo_template: Path) -> Path:
    """A fresh git copy of tests/fixtures/sample_repo (safe to mutate)."""
    root = tmp_path / "repo"
    shutil.copytree(_repo_template, root)
    return root


@pytest.fixture
def cfg(tmp_path: Path) -> AgentConfig:
    return AgentConfig(model_cache_dir=tmp_path / "models")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=64)


@dataclass
class IndexedRepo:
    root: Path
    workspace: Workspace
    indexer: Indexer
    store: IndexStore
    searcher: Searcher


@pytest.fixture
def indexed(repo: Path, cfg: AgentConfig, embedder: HashingEmbedder) -> Iterator[IndexedRepo]:
    ws = Workspace.discover(repo)
    conn, _ = open_index(ws)
    indexer = Indexer(ws, conn, cfg, embedder)
    indexer.sync()
    searcher = Searcher(ws.index_path, ws.repo_id, cfg.retrieval, embedder)
    yield IndexedRepo(repo, ws, indexer, indexer.store, searcher)
    conn.close()
