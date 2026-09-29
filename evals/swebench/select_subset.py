"""Pick a fixed, random SWE-bench Lite subset.

    python evals/swebench/select_subset.py --size 40 --seed 42

The pilot uses the first 10 ids; the full run uses all of them. Sampling is uniform over the
300 Lite test instances (no filtering by repo or difficulty), so the subset's repo mix follows
the benchmark's (django and sympy dominate). The seed and the resulting ids are committed so
the numbers in the README are reproducible.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench_Lite")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="test")
    ids = sorted(ds["instance_id"])
    chosen = random.Random(args.seed).sample(ids, args.size)
    repos = Counter(i.rsplit("-", 1)[0] for i in chosen)
    out = HERE / f"subset-{args.size}-seed{args.seed}.json"
    record = {"dataset": args.dataset, "seed": args.seed, "instance_ids": chosen,
              "pilot": chosen[:10]}  # fmt: skip
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("pilot:", chosen[:10])
    print("repos:", dict(repos.most_common()))


if __name__ == "__main__":
    main()
