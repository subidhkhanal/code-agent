"""Fast Apply latency: plan (hash check + match) + atomic write, by file size and match tier.

Usage:
    python benchmarks/bench_apply.py [--runs 50]

Each run edits one 5-line block at a random position in a synthetic Python file of N lines and
times `Planner.plan()` + `write_all()` (temp file, fsync, rename). Tiers:
  exact       SEARCH copied verbatim
  whitespace  SEARCH dedented by 4 spaces (model re-indented the snippet)
  fuzzy       one SEARCH line has a typo, forcing the fuzzy scan over every window
Target from PLAN.md: < 100 ms for a 500-line file.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from code_agent.edits.apply import Planner
from code_agent.edits.atomic import FileWrite, write_all
from code_agent.edits.parser import EditBlock
from code_agent.hashing import sha256_bytes

HERE = Path(__file__).parent
SIZES = (100, 500, 1000)
BLOCK_LINES = 5


VERBS = ("load", "parse", "check", "build", "merge", "fetch", "render", "store", "scan", "sync")
NOUNS = ("user", "token", "invoice", "order", "config", "report", "session", "record", "event")
STATEMENTS = (
    "    {a} = {b}.get('{n}')",
    "    if {a} is None:",
    "        raise ValueError('missing {n}')",
    "    {a} = [x for x in {b} if x.{n}]",
    "    total = sum(item.{n} for item in {a})",
    "    log.debug('%s %s', {a}, {b})",
    "    {a}.update({n}={b})",
    "    return {a}",
)


def synthetic_module(n_lines: int, rng: random.Random) -> list[str]:
    """Varied, realistic-looking functions (distinct names, bodies and identifiers)."""
    lines = ["import logging", "", "log = logging.getLogger(__name__)", "", ""]
    i = 0
    while len(lines) < n_lines:
        name = f"{rng.choice(VERBS)}_{rng.choice(NOUNS)}_{i}"
        lines.append(f"def {name}({rng.choice(NOUNS)}, {rng.choice(NOUNS)}s):")
        for _ in range(rng.randint(4, 8)):
            a, b, n = rng.choice(NOUNS), rng.choice(NOUNS) + "s", rng.choice(NOUNS)
            lines.append(rng.choice(STATEMENTS).format(a=a, b=b, n=n))
        lines += ["", ""]
        i += 1
    return lines[:n_lines]


def block_start(lines: list[str], rng: random.Random) -> int:
    starts = [i for i, line in enumerate(lines) if line.startswith("def ")]
    return rng.choice([s for s in starts if s + BLOCK_LINES <= len(lines)])


def make_search(lines: list[str], start: int, tier: str) -> str:
    chunk = lines[start : start + BLOCK_LINES]
    if tier == "whitespace":
        chunk = ["    " + line if line.strip() else line for line in chunk]  # re-indented
    elif tier == "fuzzy":
        chunk = chunk.copy()
        chunk[1] = chunk[1][:-3] + chunk[1][-2:]  # drop one character: a typo'd line
    return "\n".join(chunk) + "\n"


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        planner = Planner(root)
        for size in SIZES:
            for tier in ("exact", "whitespace", "fuzzy"):
                samples, kinds = [], set()
                for _ in range(args.runs):
                    lines = synthetic_module(size, rng)
                    data = ("\n".join(lines) + "\n").encode()
                    path = root / "module.py"
                    path.write_bytes(data)
                    start = block_start(lines, rng)
                    search = make_search(lines, start, tier)
                    edit = EditBlock("module.py", search, "    pass\n", 0)
                    base = {"module.py": sha256_bytes(data)}

                    t = time.perf_counter()
                    plan = planner.plan([edit], base)
                    change = plan.changes["module.py"]
                    write_all([FileWrite(path, change.before_hash, change.after, change.mode)])
                    samples.append((time.perf_counter() - t) * 1000)
                    kinds.add(str(plan.results[0].kind))
                results.append(
                    {
                        "lines": size, "tier": tier, "runs": args.runs,
                        "p50_ms": round(statistics.median(samples), 2),
                        "p95_ms": round(pct(samples, 0.95), 2),
                        "matched_as": sorted(kinds),
                    }
                )  # fmt: skip

    report = {
        "machine": f"{platform.system()} {platform.machine()}, {platform.processor()}",
        "python": sys.version.split()[0],
        "results": results,
    }
    out = HERE / "results" / "apply-latency.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'lines':>6} {'tier':<11} {'p50 ms':>8} {'p95 ms':>8}  matched as")
    for r in results:
        print(
            f"{r['lines']:>6} {r['tier']:<11} {r['p50_ms']:>8} {r['p95_ms']:>8}  {r['matched_as']}"
        )


if __name__ == "__main__":
    main()
