"""Run the headless agent on SWE-bench instances, one isolated container per task.

    .venv/Scripts/python evals/swebench/run_agent.py --run pilot --subset subset-40-seed42.json \\
        --pilot [--no-validation]

Each task runs in its own SWE-bench image (repo at /testbed, the task's Python environment),
with the agent runtime mounted from `code-agent-runtime` at /opt/agent. The container is on the
internal network: its only route out is the egress proxy, which allows the Gemini API host.

Outputs, per run: runs/<run>/<instance_id>/{report.json, patch.diff, container.log} and
runs/<run>/predictions.jsonl in the format the official harness expects. Resumable: instances
that already have a report are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from code_agent.sandbox import SandboxConfig, ensure_network_and_proxy

HERE = Path(__file__).parent
RUNTIME_IMAGE = "code-agent-runtime:latest"
RUNTIME_CONTAINER = "code-agent-runtime"


def docker(*args: str, timeout: float = 3600, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)  # fmt: skip
    if check and proc.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])}: {proc.stderr.strip()[:400]}")
    return proc


def ensure_runtime_volume() -> None:
    """A stopped container whose /opt/agent volume other containers mount with --volumes-from."""
    if docker("container", "inspect", RUNTIME_CONTAINER, check=False).returncode == 0:
        return
    docker("create", "--name", RUNTIME_CONTAINER, RUNTIME_IMAGE, "true")


def ensure_image(image: str) -> float:
    if docker("image", "inspect", image, check=False).returncode == 0:
        return 0.0
    started = time.monotonic()
    docker("pull", "-q", image, timeout=7200)
    return time.monotonic() - started


def run_instance(inst: dict, out: Path, config: Path, cfg: SandboxConfig, timeout: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    task_dir = out / "task"
    task_dir.mkdir(exist_ok=True)
    (task_dir / "problem.txt").write_text(inst["problem_statement"], encoding="utf-8")
    proxy = f"http://{cfg.proxy_name}:{cfg.proxy_port}"
    args = [
        "run", "--rm", "--network", cfg.network,
        "--volumes-from", RUNTIME_CONTAINER,
        "--cpus", "2", "--memory", "6g", "--pids-limit", "2048",
        "--tmpfs", "/tmp:rw,exec,size=4g",
        "-e", f"HTTPS_PROXY={proxy}", "-e", f"https_proxy={proxy}",
        "-e", "NO_PROXY=localhost,127.0.0.1", *[a for n in cfg.api_key_envs for a in ("-e", n)],
        "-v", f"{out.resolve()}:/out", "-v", f"{task_dir.resolve()}:/task:ro",
        "-v", f"{config.resolve()}:/config/agent.toml:ro",
        "--entrypoint", "/opt/agent/entrypoint.sh", inst["image"],
    ]  # fmt: skip
    started = time.monotonic()
    try:
        proc = docker(*args, timeout=timeout, check=False)
        log = proc.stdout + proc.stderr
        exit_code: int | None = proc.returncode
    except subprocess.TimeoutExpired as exc:
        log = f"container timed out after {timeout}s\n{exc.stdout or ''}{exc.stderr or ''}"
        exit_code = None
    (out / "container.log").write_text(log, encoding="utf-8")
    report_path = out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {
        "status": "NO_REPORT", "applied": False, "message": log[-500:],
    }  # fmt: skip
    report["container_exit_code"] = exit_code
    report["container_seconds"] = round(time.monotonic() - started, 1)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def quota_exhausted(report: dict) -> bool:
    """The task stopped because every model's daily quota was used up, not on its merits."""
    return report.get("status") == "FAILED" and "exceeded your current quota" in (
        report.get("message") or ""
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name, e.g. pilot or full-validation")
    ap.add_argument("--subset", default="subset-40-seed42.json")
    ap.add_argument("--pilot", action="store_true", help="only the subset's first 10 instances")
    ap.add_argument("--only", nargs="*", help="specific instance ids")
    ap.add_argument("--config", default=str(HERE / "agent.toml"))
    ap.add_argument("--no-validation", action="store_true", help="ablation: skip shadow checks")
    ap.add_argument("--timeout", type=int, default=2400)
    args = ap.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("GEMINI_API_KEY is not set")
    subset = json.loads((HERE / args.subset).read_text(encoding="utf-8"))
    ids = args.only or (subset["pilot"] if args.pilot else subset["instance_ids"])
    instances = {}
    for line in (HERE / "instances.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        instances[row["instance_id"]] = row

    run_dir = HERE / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    config = Path(args.config)
    if args.no_validation:
        text = config.read_text(encoding="utf-8").replace("enabled = true", "enabled = false")
        config = run_dir / "agent.toml"
        config.write_text(text, encoding="utf-8")

    cfg = SandboxConfig()
    ensure_network_and_proxy(cfg)
    ensure_runtime_volume()
    predictions = run_dir / "predictions.jsonl"
    for n, iid in enumerate(ids, 1):
        out = run_dir / iid
        if (out / "report.json").exists():
            print(f"[{n}/{len(ids)}] {iid}: done already, skipping")
            continue
        inst = instances[iid]
        pulled = ensure_image(inst["image"])
        report = run_instance(inst, out, config, cfg, args.timeout)
        if quota_exhausted(report):
            # Not a result: park the attempt (kept for inspection) and requeue the task.
            parked = run_dir / "_quota_deferred" / f"{iid}-{int(time.time())}"
            parked.parent.mkdir(exist_ok=True)
            out.rename(parked)
            print(f"[{n}/{len(ids)}] {iid}: daily model quota exhausted; stopping. "
                  "Re-run tomorrow to continue.", flush=True)  # fmt: skip
            return
        patch = (
            (out / "patch.diff").read_text(encoding="utf-8")
            if (out / "patch.diff").exists()
            else ""
        )
        with predictions.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"instance_id": iid, "model_name_or_path": f"code-agent/{args.run}",
                                 "model_patch": patch}) + "\n")  # fmt: skip
        cost = report.get("cost_usd")
        print(f"[{n}/{len(ids)}] {iid}: {report.get('status')} applied={report.get('applied')} "
              f"cost=${cost if cost is not None else '?'} {report.get('container_seconds')}s "
              f"(image pull {pulled:.0f}s)", flush=True)  # fmt: skip


if __name__ == "__main__":
    main()
