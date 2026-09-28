from __future__ import annotations

from pathlib import Path

import pytest

from code_agent.edits.apply import ApplyPlan, Planner
from code_agent.edits.parser import EditBlock
from code_agent.hashing import sha256_bytes
from code_agent.shadow.validate import ShadowValidator, module_names, select_tests
from code_agent.shadow.worktree import ShadowUnavailableError, ShadowWorktree
from code_agent.workspace import Workspace

from .conftest import git

BUGGY = (
    "        return token.expires_at > 0  # BUG: expiry is never compared with the current time\n"
)
FIXED = "        return token.expires_at > time.time()\n"


def plan_for(root: Path, *blocks: tuple[str, str, str]) -> ApplyPlan:
    edits = [EditBlock(path, s, r, i) for i, (path, s, r) in enumerate(blocks)]
    hashes = {b.path: sha256_bytes((root / b.path).read_bytes()) for b in edits
              if (root / b.path).exists()}  # fmt: skip
    plan = Planner(root).plan(edits, hashes)
    assert plan.ok, plan.feedback()
    return plan


@pytest.fixture
def shadow(repo: Path):
    worktree = ShadowWorktree(Workspace.discover(repo))
    worktree.create()
    yield worktree
    worktree.close()


# -- worktree ------------------------------------------------------------------------------


def test_worktree_mirrors_head_plus_uncommitted_state(repo: Path):
    (repo / "billing" / "invoice.py").write_text("# modified, not committed\n")
    (repo / "notes.py").write_text("untracked = True\n")
    (repo / "utils" / "http.py").unlink()
    (repo / "debug.log").write_text("ignored\n")  # .gitignore: *.log
    with ShadowWorktree(Workspace.discover(repo)) as shadow:
        root = shadow.root
        assert root != repo and (root / "auth" / "tokens.py").exists()
        assert (root / "billing" / "invoice.py").read_text() == "# modified, not committed\n"
        assert (root / "notes.py").exists()
        assert not (root / "utils" / "http.py").exists()
        assert not (root / "debug.log").exists()


def test_shadow_writes_never_touch_the_real_workspace(repo: Path, shadow: ShadowWorktree):
    before = (repo / "auth" / "tokens.py").read_bytes()
    shadow.write("auth/tokens.py", b"# rewritten in the shadow\n")
    shadow.write("brand_new.py", b"x = 1\n")
    assert (repo / "auth" / "tokens.py").read_bytes() == before
    assert not (repo / "brand_new.py").exists()
    assert git(repo, "status", "--porcelain") == ""


def test_reset_discards_shadow_edits_and_picks_up_new_user_edits(repo: Path, shadow):
    shadow.write("auth/tokens.py", b"# junk\n")
    shadow.write("junk.py", b"x = 1\n")
    (repo / "README.md").write_text("user edited this meanwhile\n")
    shadow.reset()
    assert (shadow.root / "auth/tokens.py").read_bytes() == (repo / "auth/tokens.py").read_bytes()
    assert not (shadow.root / "junk.py").exists()
    assert (shadow.root / "README.md").read_text() == "user edited this meanwhile\n"


def test_close_removes_the_worktree(repo: Path):
    shadow = ShadowWorktree(Workspace.discover(repo))
    path = shadow.create()
    assert str(path).replace("\\", "/") in git(repo, "worktree", "list").replace("\\", "/")
    shadow.close()
    assert not path.exists()
    assert git(repo, "worktree", "list").count("\n") == 1


def test_non_git_workspace_is_refused(tmp_path: Path):
    (tmp_path / "a.py").write_text("x = 1\n")
    with pytest.raises(ShadowUnavailableError):
        ShadowWorktree(Workspace.discover(tmp_path))


# -- test selection ------------------------------------------------------------------------


def test_module_names():
    assert module_names("src/pkg/auth/tokens.py") == ["pkg.auth.tokens", "auth.tokens", "tokens"]
    assert module_names("pkg/__init__.py") == ["pkg"]


def test_select_tests_by_name_and_import(repo: Path):
    assert select_tests(repo, ["auth/tokens.py"]) == ["tests/test_tokens.py"]
    assert select_tests(repo, ["tests/test_tokens.py"]) == ["tests/test_tokens.py"]
    assert select_tests(repo, ["billing/invoice.py"]) == []


# -- validation ----------------------------------------------------------------------------


def test_correct_fix_validates_and_reports_the_fixed_test(repo: Path, shadow):
    plan = plan_for(repo, ("auth/tokens.py", BUGGY, FIXED))
    report = ShadowValidator(shadow).validate(plan)
    assert report.ok, report.feedback()
    assert set(report.attempted) >= {"ruff", "pyright", "pytest"}
    assert report.tests_run == ["tests/test_tokens.py"]
    assert any("test_expired_token_rejected" in t for t in report.fixed)
    assert report.regressions == [] and report.still_failing == []
    assert (repo / "auth/tokens.py").read_text().count("> 0  # BUG") == 1  # real file untouched


def test_new_lint_error_fails_validation(repo: Path, shadow):
    plan = plan_for(repo, ("auth/tokens.py", BUGGY, "        return token.expires_at > now()\n"))
    report = ShadowValidator(shadow).validate(plan)
    assert not report.ok
    assert any(d.tool == "ruff" and d.rule == "F821" for d in report.new_diagnostics)
    assert "F821" in report.feedback() and "auth/tokens.py" in report.feedback()


def test_breaking_a_passing_test_is_a_regression(repo: Path, shadow):
    plan = plan_for(repo, ("auth/tokens.py", "        return token_id\n", "        return None\n"))
    report = ShadowValidator(shadow).validate(plan)
    assert not report.ok
    assert any("test_issue_then_verify" in t for t in report.regressions)
    # Returning None makes verify_token(None) False, so the expiry test passes by accident:
    # "fixed" is an observation about test outcomes, not a claim that the change is right.
    assert any("test_expired_token_rejected" in t for t in report.fixed)


def test_pre_existing_problems_do_not_fail_validation(repo: Path):
    target = repo / "billing" / "invoice.py"
    target.write_text("import os  # unused: a pre-existing lint error\n" + target.read_text())
    git(repo, "commit", "-qam", "add a lint error")
    with ShadowWorktree(Workspace.discover(repo)) as shadow:
        plan = plan_for(repo, ("billing/invoice.py", 'TAX_RATE = Decimal("0.2")\n',
                               'TAX_RATE = Decimal("0.25")\n'))  # fmt: skip
        report = ShadowValidator(shadow).validate(plan)
    assert report.ok, report.feedback()


def test_tests_are_skipped_when_the_gate_says_no(repo: Path, shadow):
    plan = plan_for(repo, ("auth/tokens.py", BUGGY, FIXED))
    report = ShadowValidator(shadow, test_gate=lambda tests: "user declined").validate(plan)
    assert "pytest" not in report.attempted
    assert any("user declined" in s for s in report.skipped)


def test_new_files_are_validated_too(repo: Path, shadow):
    plan = plan_for(repo, ("auth/helpers.py", "", "def f():\n    return undefined_thing\n"))
    report = ShadowValidator(shadow).validate(plan)
    assert any(d.file == "auth/helpers.py" and d.rule == "F821" for d in report.new_diagnostics)


def test_pytest_that_cannot_run_is_never_reported_as_ok(repo: Path, shadow, tmp_path: Path):
    # Found running headless in Docker: the interpreter had no pytest, and validation said
    # "pytest ok". A run with no per-test results must be reported as not validated.
    plan = plan_for(repo, ("auth/tokens.py", BUGGY, FIXED))
    fake_python = tmp_path / ("python.bat" if __import__("os").name == "nt" else "python.sh")
    if fake_python.suffix == ".bat":
        fake_python.write_text("@echo No module named pytest\r\n@exit /b 1\r\n")
    else:
        fake_python.write_text("#!/bin/sh\necho 'No module named pytest'\nexit 1\n")
        fake_python.chmod(0o755)
    report = ShadowValidator(shadow, python=str(fake_python)).validate(plan)
    assert "pytest" not in report.attempted
    assert any("could not run" in s and "No module named pytest" in s for s in report.skipped)
    assert "pytest ok" not in report.summary()
