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


def test_models_without_api_key_fails_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    result = runner.invoke(app, ["models"])
    assert result.exit_code == 1
    assert "GEMINI_API_KEY is not set" in result.output


# -- chat -----------------------------------------------------------------------------------------

BUGGY = (
    "        return token.expires_at > 0  # BUG: expiry is never compared with the current time\n"
)
FIXED = "        return token.expires_at > time.time()\n"


def _fake_llm_config(tmp_path: Path, monkeypatch, turns: list[dict]) -> None:
    import json

    from code_agent.index.embeddings import HashingEmbedder

    script = tmp_path / "script.json"
    script.write_text(json.dumps(turns), encoding="utf-8")
    config = tmp_path / "config.toml"
    config.write_text(
        f'[llm.providers.fake]\nkind = "fake"\nscript = "{script.as_posix()}"\n'
        '[llm.routes]\ncheap = ["fake:fake-cheap"]\nstrong = ["fake:fake-strong"]\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(config))
    monkeypatch.setattr("code_agent.cli._embedder", lambda cfg: HashingEmbedder())


FIX_TURNS = [
    {"text": "verify_token expiry"},
    {"tool_calls": [{"name": "read_file", "arguments": {"path": "auth/tokens.py"}}]},
    {"text": f"Compare with the current time.\n\nauth/tokens.py\n<<<<<<< SEARCH\n{BUGGY}"
             f"=======\n{FIXED}>>>>>>> REPLACE\n"},
]  # fmt: skip


def test_chat_one_shot_review_apply_and_undo(repo: Path, tmp_path: Path, monkeypatch):
    _fake_llm_config(tmp_path, monkeypatch, FIX_TURNS)
    original = (repo / "auth/tokens.py").read_bytes()
    result = runner.invoke(
        app, ["chat", "-m", "fix expired tokens being accepted", "-p", str(repo)], input="y\n"
    )
    assert result.exit_code == 0, result.output
    out = result.output
    assert "[retrieval]" in out and "read_file(path='auth/tokens.py')" in out
    assert "[edit] auth/tokens.py" in out
    assert "+        return token.expires_at > time.time()" in out  # the diff was shown
    assert "Applied" in out and "cost unknown" in out
    assert FIXED.encode() in (repo / "auth/tokens.py").read_bytes()

    assert runner.invoke(app, ["undo", "-p", str(repo)]).exit_code == 0
    assert (repo / "auth/tokens.py").read_bytes() == original


def test_chat_declined_changes_nothing(repo: Path, tmp_path: Path, monkeypatch):
    _fake_llm_config(tmp_path, monkeypatch, FIX_TURNS)
    original = (repo / "auth/tokens.py").read_bytes()
    result = runner.invoke(app, ["chat", "-m", "fix it", "-p", str(repo)], input="n\n")
    assert result.exit_code == 0, result.output
    assert "Discarded; no files were changed." in result.output
    assert (repo / "auth/tokens.py").read_bytes() == original

    from code_agent.db import connect

    conn = connect(repo / ".agent" / "index.db")
    assert conn.execute("SELECT status FROM agent_tasks").fetchone()[0] == "CANCELLED"
    conn.close()


def test_chat_without_models_explains_and_keeps_offline_features(
    repo: Path, tmp_path: Path, monkeypatch
):
    from code_agent.index.embeddings import HashingEmbedder

    monkeypatch.setenv("CODE_AGENT_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setattr("code_agent.cli._embedder", lambda cfg: HashingEmbedder())
    result = runner.invoke(app, ["chat", "-m", "anything", "-p", str(repo)])
    assert result.exit_code == 1
    assert "Generation unavailable" in result.output and "no models configured" in result.output
    assert "agent undo" in result.output


def test_chat_with_unknown_model_lists_alternatives(repo: Path, tmp_path: Path, monkeypatch):
    _fake_llm_config(tmp_path, monkeypatch, [])
    config = Path(__import__("os").environ["CODE_AGENT_CONFIG"])
    config.write_text(config.read_text().replace('"fake:fake-strong"', '"fake:retired-model"'))
    result = runner.invoke(app, ["chat", "-m", "anything", "-p", str(repo)])
    assert result.exit_code == 1
    assert "retired-model" in result.output and "fake-strong" in result.output


def test_model_controlled_text_is_not_interpreted_as_markup(
    repo: Path, tmp_path: Path, monkeypatch
):
    hostile = "[link=http://evil.example]click[/link].py"
    turns = [
        {"text": "q"},
        {"tool_calls": [{"name": "read_file", "arguments": {"path": hostile}}]},
        {"text": "done"},
    ]
    _fake_llm_config(tmp_path, monkeypatch, turns)
    result = runner.invoke(app, ["chat", "-m", "x", "-p", str(repo)])
    assert result.exit_code == 0, result.output
    assert hostile in result.output  # printed literally in the [tool] line, not rendered
