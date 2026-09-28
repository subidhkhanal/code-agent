from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from code_agent.cli import app
from code_agent.config import AgentConfig, ValidationConfig
from code_agent.headless import HeadlessRefusedError, in_container, run_headless
from code_agent.index.embeddings import HashingEmbedder
from code_agent.llm.gateway import Gateway, RetryPolicy, Route
from code_agent.llm.providers import FakeProvider, FakeTurn
from code_agent.llm.types import ToolCall
from code_agent.workspace import Workspace

BUGGY = (
    "        return token.expires_at > 0  # BUG: expiry is never compared with the current time\n"
)
FIXED = "        return token.expires_at > time.time()\n"
FIX = (f"Compare with the current time.\n\nauth/tokens.py\n<<<<<<< SEARCH\n{BUGGY}=======\n"
       f"{FIXED}>>>>>>> REPLACE\n")  # fmt: skip
SCRIPT = [
    FakeTurn("token expiry"),
    FakeTurn("", (ToolCall("r1", "read_file", {"path": "auth/tokens.py"}),)),
    FakeTurn(FIX),
]


def gateway(script) -> tuple[Gateway, FakeProvider]:
    fake = FakeProvider(script)
    return Gateway(
        providers={"fake": fake},
        routes={"cheap": [Route("fake", "fake-cheap")], "strong": [Route("fake", "fake-strong")]},
        retry=RetryPolicy(1, 0, 0),
    ), fake


def config(tmp_path: Path, *, validation: bool) -> AgentConfig:
    return AgentConfig.model_validate({
        "model_cache_dir": str(tmp_path / "models"),
        "llm": {"routes": {"cheap": ["fake:fake-cheap"], "strong": ["fake:fake-strong"]}},
        "validation": ValidationConfig(enabled=validation).model_dump(),
    })  # fmt: skip


@pytest.mark.skipif(Path("/.dockerenv").exists(), reason="running inside a container")
def test_this_host_is_not_detected_as_a_container():
    assert in_container() is False


def test_refuses_outside_a_container(repo: Path, tmp_path: Path):
    gw, _ = gateway(SCRIPT)
    with pytest.raises(HeadlessRefusedError, match="only runs inside a container"):
        run_headless(Workspace.discover(repo), config(tmp_path, validation=False), "fix it",
                     out_dir=tmp_path / "out", embedder=HashingEmbedder(), gateway=gw)  # fmt: skip
    assert FIXED not in (repo / "auth/tokens.py").read_text()


def test_headless_applies_and_reports(repo: Path, tmp_path: Path):
    gw, _ = gateway(SCRIPT)
    out = tmp_path / "out"
    report = run_headless(
        Workspace.discover(repo), config(tmp_path, validation=True),
        "fix the bug where expired tokens are still accepted", out_dir=out,
        embedder=HashingEmbedder(), gateway=gw, allow_outside_container=True,
    )  # fmt: skip
    assert report.status == "SUCCEEDED" and report.applied
    assert report.files_changed == ["auth/tokens.py"]
    assert report.validation is not None and report.validation["ok"]
    assert "pytest" in report.validation["attempted"]  # tests were auto-approved
    patch = (out / "patch.diff").read_text()
    assert "+        return token.expires_at > time.time()" in patch
    saved = json.loads((out / "report.json").read_text())
    assert saved["status"] == "SUCCEEDED" and saved["llm_calls"] == 3
    assert any(e.startswith("ValidationDone") for e in saved["events"])

    from code_agent.db import connect

    conn = connect(repo / ".agent" / "index.db")
    decisions = conn.execute("SELECT scope, decision FROM approvals").fetchall()
    conn.close()
    assert decisions and all(tuple(d) == ("once", "approved") for d in decisions)


def test_headless_failure_still_writes_a_report(repo: Path, tmp_path: Path):
    wrong = FakeTurn("auth/tokens.py\n<<<<<<< SEARCH\nnot in file\n=======\nx\n>>>>>>> REPLACE\n")
    gw, _ = gateway([SCRIPT[0], wrong, wrong, wrong])
    out = tmp_path / "out"
    report = run_headless(Workspace.discover(repo), config(tmp_path, validation=False), "fix it",
                          out_dir=out, embedder=HashingEmbedder(), gateway=gw,
                          allow_outside_container=True)  # fmt: skip
    assert report.status == "FAILED" and not report.applied
    assert report.rejections == ["NO_MATCH"] * 3
    assert (out / "patch.diff").read_text() == ""
    assert json.loads((out / "report.json").read_text())["status"] == "FAILED"


def test_cli_requires_both_flags_and_a_container(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    runner = CliRunner()
    result = runner.invoke(app, ["run", "--task", "x", "-p", str(repo)])
    assert result.exit_code == 2 and "--headless --auto-approve" in result.output
    monkeypatch.setattr("code_agent.cli._embedder", lambda cfg: HashingEmbedder())
    result = runner.invoke(app, ["run", "--task", "x", "--headless", "--auto-approve",
                                 "-p", str(repo), "--out", str(tmp_path / "o")])  # fmt: skip
    assert result.exit_code == 3 and "only runs inside a container" in result.output
