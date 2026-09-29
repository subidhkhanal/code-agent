"""File-level retrieval recall on SWE-bench tasks, with ablations (PLAN.md section 6.2).

    .venv/Scripts/python evals/retrieval/run_retrieval_eval.py --pilot

For each task: check out the repo at the task's base commit, bring the index up to date
(one checkout per repo, so moving between commits is an incremental update that reuses
vectors of unchanged chunks), query with the problem statement, and measure recall@5/@10 of
the files the gold patch touches, for four conditions:

  bm25        keyword search only
  vector      embedding search only
  hybrid      symbol + BM25 + vector, fused with RRF (what the agent uses)
  hybrid+exp  hybrid top chunks plus symbol expansion (definitions they call)

Ranked chunks are turned into a ranked *file* list by first appearance. The query is the raw
problem statement (the agent adds LLM-rewritten queries on top; this measures the
deterministic part without paying for model calls).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from code_agent.config import AgentConfig
from code_agent.context.assembly import ContextAssembler
from code_agent.index.embeddings import FastEmbedEmbedder
from code_agent.index.indexer import Indexer
from code_agent.index.store import open_index
from code_agent.retrieval.search import Mode, Searcher
from code_agent.workspace import Workspace

HERE = Path(__file__).parent
REPOS = HERE / ".repos"
KS = (5, 10)
CANDIDATES = 50


def git(repo: Path, *args: str, timeout: float = 3600) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=timeout, check=True)  # fmt: skip
    return proc.stdout


def checkout(repo_name: str, commit: str) -> Path:
    path = REPOS / repo_name.replace("/", "__")
    if not path.exists():
        path.mkdir(parents=True)
        git(path, "init", "-q")
        git(path, "remote", "add", "origin", f"https://github.com/{repo_name}.git")
    git(path, "fetch", "-q", "--depth", "1", "origin", commit)
    git(path, "checkout", "-q", "--force", commit)
    git(path, "clean", "-fdq", "-e", ".agent")
    return path


def files_in_order(paths: list[str]) -> list[str]:
    return list(dict.fromkeys(paths))


def recall(ranked_files: list[str], gold: list[str], k: int) -> float:
    return len(set(ranked_files[:k]) & set(gold)) / len(gold) if gold else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in (HERE / "gold.jsonl").read_text("utf-8").splitlines()]
    if args.pilot:
        rows = [r for r in rows if r["pilot"]]
    rows.sort(key=lambda r: (r["repo"], r["instance_id"]))  # group by repo: incremental updates

    cfg = AgentConfig()
    embedder = FastEmbedEmbedder(cfg.index.embedding_model, cfg.model_cache_dir,
                                 cfg.index.embedding_batch_size)  # fmt: skip
    results = []
    for n, row in enumerate(rows, 1):
        repo = checkout(row["repo"], row["base_commit"])
        ws = Workspace.discover(repo)
        conn, _ = open_index(ws)
        started = time.monotonic()
        stats = Indexer(ws, conn, cfg, embedder).sync()
        index_s = time.monotonic() - started
        searcher = Searcher(ws.index_path, ws.repo_id, cfg.retrieval, embedder)
        query = row["problem_statement"]

        ranked: dict[str, list[str]] = {}
        for name, mode in (("bm25", Mode.BM25), ("vector", Mode.VECTOR), ("hybrid", Mode.HYBRID)):
            hits = searcher.search(query, k=CANDIDATES, mode=mode).hits
            ranked[name] = files_in_order([h.file_path for h in hits])
        assembler = ContextAssembler(searcher, conn, ws.repo_id, top_k=CANDIDATES)
        bundle = assembler.assemble([query], budget_tokens=10**9)
        ranked["hybrid+exp"] = files_in_order([i.chunk.file_path for i in bundle.items])
        conn.close()

        entry = {
            "instance_id": row["instance_id"], "repo": row["repo"], "gold": row["gold_files"],
            "index_seconds": round(index_s, 1), "chunks_written": stats.chunks_written,
            "vectors_embedded": stats.vectors_embedded, "vectors_reused": stats.vectors_reused,
            "recall": {c: {f"@{k}": recall(files, row["gold_files"], k) for k in KS}
                       for c, files in ranked.items()},
            "top5": {c: files[:5] for c, files in ranked.items()},
        }  # fmt: skip
        results.append(entry)
        print(f"[{n}/{len(rows)}] {row['instance_id']}: index {index_s:.0f}s "
              f"(+{stats.vectors_embedded} embedded, {stats.vectors_reused} reused) "
              + " ".join(f"{c}@10={entry['recall'][c]['@10']:.2f}" for c in ranked),
              flush=True)  # fmt: skip
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    conditions = list(results[0]["recall"]) if results else []
    summary = {
        c: {f"recall@{k}": round(statistics.mean(r["recall"][c][f"@{k}"] for r in results), 3)
            for k in KS}
        for c in conditions
    }  # fmt: skip
    Path(args.out).write_text(
        json.dumps(
            {"instances": len(results), "summary": summary, "per_instance": results}, indent=2
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
