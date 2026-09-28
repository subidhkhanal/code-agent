"""Runner tests. The commands here are the test's own fixed Python snippets, not model output."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from code_agent.llm.types import CancelToken
from code_agent.security.runner import run_command, scrubbed_env

PY = sys.executable


def test_captures_output_and_exit_code(tmp_path: Path):
    result = run_command([PY, "-c", "import sys; print('hello'); sys.exit(3)"], tmp_path)
    assert result.exit_code == 3 and "hello" in result.output
    assert not (result.timed_out or result.cancelled or result.truncated)
    assert "[exit code 3" in result.render()


def test_runs_without_a_shell(tmp_path: Path):
    # With a shell, `;` would start a second command. Without one it is just an argument.
    result = run_command([PY, "-c", "import sys; print(sys.argv[1:])", "a; echo pwned"], tmp_path)
    assert "['a; echo pwned']" in result.output and "pwned\n" not in result.output


def test_environment_is_allowlisted(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "should-not-leak")
    monkeypatch.setenv("MY_CUSTOM_CREDENTIAL", "also-hidden")
    code = (
        "import os; print(os.environ.get('GEMINI_API_KEY'), os.environ.get('MY_CUSTOM_CREDENTIAL'))"
    )
    result = run_command([PY, "-c", code], tmp_path)
    assert result.output.strip() == "None None"
    env = scrubbed_env({"PATH": "/bin", "AWS_SECRET_ACCESS_KEY": "x", "GITHUB_TOKEN": "y"})
    assert "PATH" in env and "AWS_SECRET_ACCESS_KEY" not in env and "GITHUB_TOKEN" not in env


def test_timeout_kills_the_process(tmp_path: Path):
    started = time.monotonic()
    result = run_command([PY, "-c", "import time; time.sleep(30)"], tmp_path, timeout_s=1)
    assert result.timed_out and result.exit_code is None
    assert time.monotonic() - started < 15


def test_cancel_kills_the_whole_process_tree(tmp_path: Path):
    marker = tmp_path / "grandchild-survived.txt"
    grandchild = (
        f"import time, pathlib; time.sleep(3); pathlib.Path({str(marker)!r}).write_text('x')"
    )
    parent = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {grandchild!r}]); time.sleep(30)"
    )
    cancel = CancelToken()
    threading.Timer(0.8, cancel.cancel).start()
    result = run_command([PY, "-c", parent], tmp_path, cancel=cancel)
    assert result.cancelled and result.exit_code is None
    time.sleep(4)
    assert not marker.exists(), "grandchild process outlived the cancel"


def test_output_is_capped(tmp_path: Path):
    result = run_command([PY, "-c", "print('x' * 1_000_000)"], tmp_path, max_output_bytes=10_000)
    assert result.truncated and len(result.output) <= 10_000
    assert "[output truncated]" in result.render()


def test_cwd_is_respected(tmp_path: Path):
    (tmp_path / "sub").mkdir()
    result = run_command([PY, "-c", "import os; print(os.getcwd())"], tmp_path / "sub")
    assert result.output.strip().endswith("sub")
