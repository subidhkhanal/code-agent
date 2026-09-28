import json
from pathlib import Path

from typer.testing import CliRunner

from code_agent.cli import app

runner = CliRunner()


def test_index_then_search_json(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    result = runner.invoke(app, ["index", "--no-embed", "-p", str(repo)])
    assert result.exit_code == 0, result.output
    assert "+8" in result.output

    result = runner.invoke(
        app, ["search", "verify_token", "-m", "symbol", "--json", "-p", str(repo)]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["hits"][0]["symbol"] == "TokenStore.verify_token"
    assert payload["hits"][0]["file_path"] == "auth/tokens.py"


def test_search_table_output(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    runner.invoke(app, ["index", "--no-embed", "-p", str(repo)])
    result = runner.invoke(app, ["search", "calculate total", "-m", "bm25", "-p", str(repo)])
    assert result.exit_code == 0
    assert "billing/invoice.py" in result.output


def test_search_without_index_fails_clearly(repo: Path):
    result = runner.invoke(app, ["search", "anything", "-p", str(repo)])
    assert result.exit_code == 1
    assert "agent index" in result.output


def test_rebuild_discards_index(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    runner.invoke(app, ["index", "--no-embed", "-p", str(repo)])
    result = runner.invoke(app, ["index", "--no-embed", "--rebuild", "-p", str(repo)])
    assert "+8" in result.output


def _apply_fix(repo: Path) -> None:
    from code_agent.edits.apply import Planner
    from code_agent.edits.changesets import ChangeSetStore
    from code_agent.edits.parser import EditBlock
    from code_agent.hashing import sha256_bytes
    from code_agent.index.store import open_index
    from code_agent.workspace import Workspace

    ws = Workspace.discover(repo)
    target = repo / "auth/tokens.py"
    edit = EditBlock("auth/tokens.py", "        return token.expires_at > 0  # BUG: expiry is never"
                     " compared with the current time\n",
                     "        return token.expires_at > time.time()\n", 0)  # fmt: skip
    plan = Planner(repo).plan([edit], {"auth/tokens.py": sha256_bytes(target.read_bytes())})
    assert plan.ok, plan.feedback()
    conn, _ = open_index(ws)
    ChangeSetStore(conn, ws.root).apply(plan)
    conn.close()


def test_undo_reverts_last_change_set(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    original = (repo / "auth/tokens.py").read_bytes()
    _apply_fix(repo)
    assert (repo / "auth/tokens.py").read_bytes() != original
    result = runner.invoke(app, ["undo", "-p", str(repo)])
    assert result.exit_code == 0, result.output
    assert "restored auth/tokens.py" in result.output
    assert (repo / "auth/tokens.py").read_bytes() == original
    result = runner.invoke(app, ["undo", "-p", str(repo)])
    assert result.exit_code == 1 and "Nothing to undo" in result.output


def test_rebuild_keeps_undo_history(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    original = (repo / "auth/tokens.py").read_bytes()
    _apply_fix(repo)
    runner.invoke(app, ["index", "--no-embed", "--rebuild", "-p", str(repo)])
    assert runner.invoke(app, ["undo", "-p", str(repo)]).exit_code == 0
    assert (repo / "auth/tokens.py").read_bytes() == original


def test_undo_refuses_when_user_edited_afterwards(repo: Path, tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    _apply_fix(repo)
    (repo / "auth/tokens.py").write_text("# my own rewrite\n")
    result = runner.invoke(app, ["undo", "-p", str(repo)])
    assert result.exit_code == 1 and "Undo refused" in result.output
    assert (repo / "auth/tokens.py").read_text() == "# my own rewrite\n"
