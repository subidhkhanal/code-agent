from pathlib import Path

from code_agent.config import AgentConfig, IndexConfig
from code_agent.index.files import FileSelector, read_source_text
from code_agent.workspace import Workspace

from .conftest import git


def scanned(root: Path, cfg: IndexConfig | None = None) -> set[str]:
    selector = FileSelector(Workspace.discover(root), cfg or AgentConfig().index)
    return {f.rel_path for f in selector.scan()}


def write(root: Path, rel: str, content: str | bytes = "x = 1\n") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def test_fixture_repo_selection(repo: Path):
    # .gitignore/.agentignore have no indexable extension; docs/drafts.md is in .agentignore.
    assert scanned(repo) == {
        "README.md",
        "auth/__init__.py",
        "auth/tokens.py",
        "billing/__init__.py",
        "billing/invoice.py",
        "utils/__init__.py",
        "utils/http.py",
        "tests/test_tokens.py",
    }


def test_gitignored_and_untracked_files(repo: Path):
    write(repo, "generated/out.py")  # .gitignore: generated/
    write(repo, "debug.log", "log\n")  # .gitignore: *.log
    write(repo, "new_module.py")  # untracked but not ignored -> indexed
    files = scanned(repo)
    assert "new_module.py" in files
    assert "generated/out.py" not in files and "debug.log" not in files


def test_dependency_and_cache_dirs_excluded_even_if_not_gitignored(repo: Path):
    for rel in [".venv/lib/x.py", "node_modules/p/index.js", "pkg/__pycache__/m.py",
                "build/lib/m.py", "pkg.egg-info/PKG-INFO.txt", "site-packages/m.py"]:  # fmt: skip
        write(repo, rel)
    files = scanned(repo)
    assert not any(
        part in f
        for f in files
        for part in (".venv", "node_modules", "__pycache__", "build/", "egg-info", "site-packages")
    )


def test_sensitive_files_excluded_even_when_tracked(repo: Path):
    write(repo, ".env", "API_KEY=sk-test-not-a-real-key\n")
    write(repo, "config/prod.env", "X=1\n")
    write(repo, "secrets/settings.py", "PASSWORD = 'hunter2'\n")
    write(repo, "certs/server.pem", "-----BEGIN PRIVATE KEY-----\n")
    git(repo, "add", "-f", ".")
    git(repo, "commit", "-q", "-m", "oops, committed secrets")
    files = scanned(repo)
    assert not {".env", "config/prod.env", "secrets/settings.py", "certs/server.pem"} & files


def test_extra_ignore_and_extra_sensitive_from_config(repo: Path):
    cfg = IndexConfig(extra_ignore=["billing/"], extra_sensitive=["utils/http.py"])
    files = scanned(repo, cfg)
    assert not any(f.startswith("billing/") for f in files)
    assert "utils/http.py" not in files


def test_agentignore_negations_are_ignored(repo: Path):
    (repo / ".agentignore").write_text("docs/drafts.md\n!.env\n", encoding="utf-8")
    write(repo, ".env", "X=1\n")
    assert ".env" not in scanned(repo)


def test_unknown_extensions_and_oversized_files_skipped(repo: Path):
    write(repo, "image.png", b"\x89PNG\r\n\x1a\n\x00\x00")
    write(repo, "big.py", "x = 1\n" * 50_000)
    files = scanned(repo, IndexConfig(max_file_bytes=100_000))
    assert "image.png" not in files and "big.py" not in files


def test_binary_and_non_utf8_content_detected():
    assert read_source_text(b"abc\x00def") is None
    assert read_source_text("café".encode("latin-1")) is None
    assert read_source_text("café".encode()) == "café"
    assert read_source_text(b"\xef\xbb\xbfx = 1") == "x = 1"  # BOM stripped


def test_non_git_directory_uses_root_gitignore(tmp_path: Path):
    root = tmp_path / "plain"
    write(root, "keep.py")
    write(root, "skip/me.py")
    write(root, ".venv/lib.py")
    write(root, ".gitignore", "skip/\n")
    ws = Workspace.discover(root)
    assert not ws.is_git
    assert scanned(root) == {"keep.py"}


def test_vcs_ignored_for_explicit_paths(repo: Path):
    selector = FileSelector(Workspace.discover(repo), AgentConfig().index)
    assert selector.vcs_ignored(["generated/x.py", "auth/tokens.py", "a.log"]) == {
        "generated/x.py",
        "a.log",
    }
