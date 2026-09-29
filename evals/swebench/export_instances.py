"""Export the subset's task inputs (problem statement + image name) for the agent runner.

    .venv-eval/Scripts/python evals/swebench/export_instances.py --subset subset-40-seed42.json

Only what the agent is allowed to see is exported: the problem statement. The gold patch,
test patch and hints stay in the dataset and are used only by the official harness.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="subset-40-seed42.json")
    args = ap.parse_args()
    subset = json.loads((HERE / args.subset).read_text(encoding="utf-8"))
    wanted = set(subset["instance_ids"])
    ds = load_dataset(subset["dataset"], split="test")
    out = HERE / "instances.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for row in ds:
            if row["instance_id"] in wanted:
                fh.write(json.dumps({
                    "instance_id": row["instance_id"],
                    "repo": row["repo"],
                    "image": row["image"],
                    "problem_statement": row["problem_statement"],
                }) + "\n")  # fmt: skip
    print(f"wrote {len(wanted)} instances to {out}")


if __name__ == "__main__":
    main()
