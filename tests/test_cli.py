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
