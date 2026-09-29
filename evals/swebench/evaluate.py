"""Score a run with the official SWE-bench harness and summarize it.

    .venv-eval/Scripts/python evals/swebench/evaluate.py --run pilot

Runs `swebench.harness.run_evaluation` unchanged (as a subprocess, from the run directory, so
its logs land there), then joins its verdicts with the agent's own reports.

The resolve rate is computed over *every attempted instance*. The harness silently skips
predictions with an empty patch; those count as unresolved here, not as absent.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def harness(run_dir: Path, run: str, ids: list[str], workers: int) -> dict:
    run_id = f"code-agent-{run}"
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", "SWE-bench/SWE-bench_Lite", "--split", "test",
        "--predictions_path", str((run_dir / "predictions.jsonl").resolve()),
        "--run_id", run_id, "--max_workers", str(workers), "--timeout", "1800",
        "--instance_ids", *ids,
    ]  # fmt: skip
    proc = subprocess.run(cmd, cwd=run_dir, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", check=False)  # fmt: skip
    (run_dir / "harness.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    reports = list(run_dir.glob(f"*.{run_id}.json"))
    if not reports:
        raise SystemExit(f"harness produced no report; see {run_dir / 'harness.log'}")
    return json.loads(reports[0].read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-workers", type=int, default=1)
    ap.add_argument("--skip-harness", action="store_true", help="re-summarize existing results")
    args = ap.parse_args()
    run_dir = HERE / "runs" / args.run

    predictions = {}
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        predictions[row["instance_id"]] = row  # last attempt wins
    ids = sorted(predictions)

    if args.skip_harness:
        verdict = json.loads(next(run_dir.glob(f"*.code-agent-{args.run}.json")).read_text("utf-8"))
    else:
        nonempty = [i for i in ids if predictions[i]["model_patch"].strip()]
        verdict = harness(run_dir, args.run, nonempty, args.max_workers) if nonempty else {}
    resolved = set(verdict.get("resolved_ids", []))
    errored = set(verdict.get("error_ids", []))

    rows = []
    for iid in ids:
        report = json.loads((run_dir / iid / "report.json").read_text(encoding="utf-8"))
        validation = report.get("validation") or {}
        rows.append({
            "instance_id": iid,
            "resolved": iid in resolved,
            "harness_error": iid in errored,
            "patch_empty": not predictions[iid]["model_patch"].strip(),
            "status": report.get("status"),
            "message": (report.get("message") or "")[:160],
            "cost_usd": report.get("cost_usd"),
            "seconds": report.get("seconds") or report.get("container_seconds"),
            "llm_calls": report.get("llm_calls"),
            "tool_calls": report.get("tool_calls"),
            "edit_attempts": report.get("edit_attempts"),
            "fix_attempts": report.get("fix_attempts"),
            "rejections": report.get("rejections"),
            "validation": validation.get("summary"),
        })  # fmt: skip

    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if isinstance(r[key], int | float)]
        return round(statistics.mean(values), 4) if values else None

    first_try = [r for r in rows if r["edit_attempts"] is not None and not r["patch_empty"]]
    summary = {
        "run": args.run,
        "instances": len(rows),
        "resolved": len(resolved),
        "resolve_rate": round(len(resolved) / len(rows), 3) if rows else 0.0,
        "empty_patches": sum(r["patch_empty"] for r in rows),
        "harness_errors": len(errored),
        "agent_failed": sum(r["status"] != "SUCCEEDED" for r in rows),
        "avg_cost_usd": mean("cost_usd"),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4),
        "avg_seconds": mean("seconds"),
        "avg_llm_calls": mean("llm_calls"),
        "avg_tool_calls": mean("tool_calls"),
        "avg_edit_attempts": mean("edit_attempts"),
        "avg_fix_attempts": mean("fix_attempts"),
        "apply_first_try_rate": (
            round(sum(r["edit_attempts"] == 0 for r in first_try) / len(first_try), 3)
            if first_try
            else None
        ),
        "rejection_reasons": {
            reason: sum(reason in (r["rejections"] or []) for r in rows)
            for reason in sorted({x for r in rows for x in (r["rejections"] or [])})
        },
        "per_instance": rows,
    }
    results = HERE / "results"
    results.mkdir(exist_ok=True)
    (results / f"{args.run}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_instance"}, indent=2))
    for r in rows:
        mark = "RESOLVED" if r["resolved"] else ("empty" if r["patch_empty"] else "unresolved")
        print(f"  {r['instance_id']:40s} {mark:10s} {r['status']:9s} ${r['cost_usd'] or 0:.3f} "
              f"{r['seconds'] or 0:>6.0f}s  {r['validation'] or ''}")  # fmt: skip


if __name__ == "__main__":
    main()
