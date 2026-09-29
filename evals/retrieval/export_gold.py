"""Export what the retrieval eval needs: repo, base commit, problem statement, gold files.

    .venv-eval/Scripts/python evals/retrieval/export_gold.py

Gold files are the files the reference patch modifies. They are used only to *score*
retrieval; the agent never sees them.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset

HERE = Path(__file__).parent
SUBSET = HERE.parent / "swebench" / "subset-40-seed42.json"


def main() -> None:
    subset = json.loads(SUBSET.read_text(encoding="utf-8"))
    order = {iid: n for n, iid in enumerate(subset["instance_ids"])}
    ds = load_dataset(subset["dataset"], split="test")
    rows = []
    for row in ds:
        if row["instance_id"] not in order:
            continue
        gold = sorted(set(re.findall(r"^diff --git a/(\S+) b/", row["patch"], re.MULTILINE)))
        rows.append({
            "instance_id": row["instance_id"], "repo": row["repo"],
            "base_commit": row["base_commit"], "problem_statement": row["problem_statement"],
            "gold_files": gold, "pilot": row["instance_id"] in subset["pilot"],
        })  # fmt: skip
    rows.sort(key=lambda r: order[r["instance_id"]])
    out = HERE / "gold.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
