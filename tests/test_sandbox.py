from pathlib import Path

from code_agent.sandbox import SandboxConfig, agent_run_args


def test_agent_container_is_isolated_and_never_gets_the_key_on_the_command_line(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "super-secret-value")
    args = agent_run_args(tmp_path / "repo", tmp_path / "out", "fix it", SandboxConfig(),
                          config_file=tmp_path / "agent.toml")  # fmt: skip
    joined = " ".join(args)
    assert "--network code-agent-internal" in joined  # internal network: no internet route
    assert "--user 1000:1000" in joined and "--read-only" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--memory 4g" in joined and "--cpus 2" in joined and "--pids-limit 512" in joined
    assert "HTTPS_PROXY=http://code-agent-egress:3128" in joined
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        assert args[args.index(name) - 1] == "-e"  # name only: the value comes from the env
    assert "super-secret-value" not in joined
    assert args[-9:] == ["run", "--task", "fix it", "--headless", "--auto-approve",
                         "-p", "/work", "--out", "/out"]  # fmt: skip
