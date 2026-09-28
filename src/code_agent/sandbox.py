"""Host-side launcher for headless runs: `spawn_sandbox(repo, task)` (ADR 0009).

This is the integration point for an external orchestrator. It uses the docker CLI (no SDK
dependency) to:

1. create an *internal* Docker network (no route to the internet),
2. start the egress proxy on it (also attached to the default bridge), allowlisting only the
   LLM provider's API host,
3. run the agent image on the internal network: non-root, read-only root filesystem, tmpfs for
   scratch space, CPU/memory/pids limits and a hard timeout, with the repo and an output
   directory mounted,
4. return the parsed `report.json` (the patch is in `out_dir/patch.diff`).

The API key is passed as `-e NAME` (value taken from this process's environment), so it never
appears on a command line or in `docker inspect` output of the launcher's arguments.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class SandboxError(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxConfig:
    image: str = "code-agent:latest"
    network: str = "code-agent-internal"
    proxy_name: str = "code-agent-egress"
    proxy_port: int = 3128
    allow: tuple[str, ...] = ("generativelanguage.googleapis.com:443",)
    api_key_env: str = "GEMINI_API_KEY"
    cpus: str = "2"
    memory: str = "4g"
    pids_limit: int = 512
    timeout_s: int = 1800
    extra_env: dict[str, str] = field(default_factory=dict)


def _docker(*args: str, check: bool = True, timeout: float = 120) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout,
                          check=False)  # fmt: skip
    if check and proc.returncode != 0:
        raise SandboxError(f"docker {' '.join(args[:3])} failed: {proc.stderr.strip()[:500]}")
    return proc


def ensure_network_and_proxy(cfg: SandboxConfig) -> None:
    if _docker("network", "inspect", cfg.network, check=False).returncode != 0:
        _docker("network", "create", "--internal", cfg.network)
    state = _docker("inspect", "-f", "{{.State.Running}}", cfg.proxy_name, check=False)
    if state.returncode == 0 and state.stdout.strip() == "true":
        return
    _docker("rm", "-f", cfg.proxy_name, check=False)
    allow_args = [arg for host in cfg.allow for arg in ("--allow", host)]
    _docker(
        "run", "-d", "--name", cfg.proxy_name, "--restart", "unless-stopped",
        "--user", "1000:1000", "--read-only", "--memory", "128m", "--cpus", "0.5",
        "--entrypoint", "python", cfg.image, "-m", "code_agent.egress",
        "--port", str(cfg.proxy_port), *allow_args,
    )  # fmt: skip
    _docker("network", "connect", cfg.network, cfg.proxy_name)


def agent_run_args(
    repo: Path,
    out_dir: Path,
    task: str,
    cfg: SandboxConfig,
    *,
    config_file: Path | None = None,
    image: str | None = None,
    workdir: str = "/work",
    mount_repo: bool = True,
) -> list[str]:
    """The `docker run` argument list (exposed for tests and for SWE-bench's variant)."""
    proxy = f"http://{cfg.proxy_name}:{cfg.proxy_port}"
    args = [
        "run", "--rm", "--network", cfg.network,
        "--user", "1000:1000", "--read-only",
        "--tmpfs", "/tmp:rw,exec,size=2g", "--tmpfs", "/home/agent/.tmp:rw,size=256m",
        "--cpus", cfg.cpus, "--memory", cfg.memory, "--pids-limit", str(cfg.pids_limit),
        "--security-opt", "no-new-privileges",
        "-e", f"HTTPS_PROXY={proxy}", "-e", f"https_proxy={proxy}",
        "-e", "NO_PROXY=localhost,127.0.0.1", "-e", cfg.api_key_env,
        "-v", f"{out_dir.resolve()}:/out",
    ]  # fmt: skip
    if mount_repo:
        args += ["-v", f"{repo.resolve()}:{workdir}"]
    if config_file is not None:
        args += ["-v", f"{config_file.resolve()}:/config/agent.toml:ro",
                 "-e", "CODE_AGENT_CONFIG=/config/agent.toml"]  # fmt: skip
    for key, value in cfg.extra_env.items():
        args += ["-e", f"{key}={value}"]
    args += [image or cfg.image, "run", "--task", task, "--headless", "--auto-approve",
             "-p", workdir, "--out", "/out"]  # fmt: skip
    return args


def spawn_sandbox(
    repo: Path,
    task: str,
    out_dir: Path,
    cfg: SandboxConfig | None = None,
    *,
    config_file: Path | None = None,
) -> dict:
    """Run one headless task against `repo` in an isolated container; return its report."""
    cfg = cfg or SandboxConfig()
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_network_and_proxy(cfg)
    args = agent_run_args(repo, out_dir, task, cfg, config_file=config_file)
    proc = _docker(*args, check=False, timeout=cfg.timeout_s)
    (out_dir / "container.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    report = out_dir / "report.json"
    if not report.exists():
        raise SandboxError(f"no report produced (exit {proc.returncode}); see container.log")
    return json.loads(report.read_text(encoding="utf-8"))
