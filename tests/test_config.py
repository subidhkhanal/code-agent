from pathlib import Path

import pytest
from pydantic import ValidationError

from code_agent.config import AgentConfig, load_config


def test_defaults_when_no_file(tmp_path: Path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg == AgentConfig()
    assert cfg.retrieval.rrf_k == 60


def test_load_from_toml(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        '[index]\nembedding_model = "BAAI/bge-small-en-v1.5"\nextra_sensitive = ["*.sqlite"]\n'
        "[retrieval]\ntop_k = 5\n",
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg.index.embedding_model == "BAAI/bge-small-en-v1.5"
    assert cfg.index.extra_sensitive == ["*.sqlite"]
    assert cfg.retrieval.top_k == 5


def test_env_var_selects_config_file(tmp_path: Path, monkeypatch):
    path = tmp_path / "alt.toml"
    path.write_text("[retrieval]\ntop_k = 3\n", encoding="utf-8")
    monkeypatch.setenv("CODE_AGENT_CONFIG", str(path))
    assert load_config().retrieval.top_k == 3


def test_unknown_keys_are_rejected(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[retrieval]\ntopk = 5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_out_of_range_values_are_rejected(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text("[retrieval]\ntop_k = 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)
