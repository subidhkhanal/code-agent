import os
from pathlib import Path

import pytest

from code_agent.security.paths import (
    InvalidPatternError,
    PathOutsideWorkspaceError,
    SensitivePathPolicy,
    resolve_in_workspace,
)


@pytest.mark.parametrize(
    "path",
    [
        ".env", "config/.env", ".env.local", "prod.env", ".ENV",
        "certs/server.pem", "deploy/app.key", "id_rsa", "home/.ssh/config", ".aws/credentials",
        "secrets/db.txt", "app/secrets/token", ".npmrc", ".git-credentials",
        "service-account-prod.json", "infra/terraform.tfstate",
        ".git/config", ".git/hooks/pre-commit", ".agent/index.db",
    ],
)  # fmt: skip
def test_default_sensitive_paths(path: str):
    assert SensitivePathPolicy().is_sensitive(path)


@pytest.mark.parametrize(
    "path",
    ["env.py", "src/environment.py", "src/secretsauce.py", "keys.py", "README.md", "docs/pem.md"],
)
def test_ordinary_paths_are_not_sensitive(path: str):
    assert not SensitivePathPolicy().is_sensitive(path)


def test_windows_separators_are_normalized():
    assert SensitivePathPolicy().is_sensitive("app\\secrets\\token")


def test_extra_patterns_add_to_defaults():
    policy = SensitivePathPolicy(["*.sqlite", "private/"])
    assert policy.is_sensitive("data/app.sqlite")
    assert policy.is_sensitive("private/notes.md")
    assert policy.is_sensitive(".env")  # defaults still apply


def test_negated_patterns_cannot_unblock_defaults():
    with pytest.raises(InvalidPatternError):
        SensitivePathPolicy(["!.env.example"])


def test_resolve_relative_path_inside_workspace(tmp_path: Path):
    (tmp_path / "pkg").mkdir()
    assert resolve_in_workspace(tmp_path, "pkg/mod.py") == (tmp_path / "pkg" / "mod.py").resolve()
    assert resolve_in_workspace(tmp_path, "pkg/../pkg/mod.py").name == "mod.py"


@pytest.mark.parametrize("escape", ["../outside.py", "pkg/../../outside.py", "../../etc/passwd"])
def test_dotdot_escape_is_rejected(tmp_path: Path, escape: str):
    root = tmp_path / "ws"
    (root / "pkg").mkdir(parents=True)
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_in_workspace(root, escape)


def test_absolute_path_outside_is_rejected(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_in_workspace(root, tmp_path / "other.py")


def test_nul_byte_is_rejected(tmp_path: Path):
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_in_workspace(tmp_path, "a\x00b.py")


def _symlink_or_skip(link: Path, target: Path, is_dir: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=is_dir)
    except OSError as exc:  # Windows without Developer Mode / admin
        pytest.skip(f"cannot create symlinks here: {exc}")


def test_symlink_pointing_outside_is_rejected(tmp_path: Path):
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    _symlink_or_skip(root / "link", outside, is_dir=True)
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_in_workspace(root, "link/secret.txt")


def test_symlink_inside_workspace_is_allowed(tmp_path: Path):
    (tmp_path / "real.py").write_text("x = 1\n")
    _symlink_or_skip(tmp_path / "alias.py", tmp_path / "real.py")
    assert resolve_in_workspace(tmp_path, "alias.py") == (tmp_path / "real.py").resolve()


@pytest.mark.skipif(os.name != "nt", reason="NTFS-specific path forms")
@pytest.mark.parametrize("path", ["file.py:stream", "C:relative.py"])
def test_windows_alternate_streams_and_drive_relative_rejected(tmp_path: Path, path: str):
    with pytest.raises(PathOutsideWorkspaceError):
        resolve_in_workspace(tmp_path, path)
