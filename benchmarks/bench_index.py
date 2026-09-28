"""Index benchmark: initial build time and per-file incremental update time.

Usage:
    python benchmarks/bench_index.py --repo https://github.com/pytest-dev/pytest --ref 8.3.4 \
        --model BAAI/bge-small-en-v1.5

The repo is cloned (shallow) into benchmarks/.repos/ and indexed from a throwaway copy, so the
benchmark never writes to a working tree you care about. Results go to benchmarks/results/.

What is measured:
* model_load_s        loading the ONNX model (weights already downloaded; download excluded)
* scan_chunk_fts_s    discovery + hashing + tree-sitter chunking + SQLite/FTS5 writes
* embed_s             embedding every chunk on CPU
* incremental_*_ms    one file modified in one function, then `update_paths([file])`, i.e.
                      what the file watcher does on save (re-chunk + re-embed changed chunks)
* noop_sync_s         full `sync()` when nothing changed (metadata fast path)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path

from code_agent.config import AgentConfig, IndexConfig
from code_agent.index.embeddings import FastEmbedEmbedder
from code_agent.index.indexer import Indexer
from code_agent.index.store import open_index
from code_agent.workspace import Workspace

HERE = Path(__file__).parent


def run(*cmd: str, cwd: Path | None = None) -> str:
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout


def checkout(url: str, ref: str) -> Path:
    name = url.rstrip("/").rsplit("/", 1)[-1]
    dest = HERE / ".repos" / f"{name}-{ref}"
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "-q", "--depth", "1", "--branch", ref, url, str(dest))
    return dest


def rmtree(path: Path) -> None:
    """shutil.rmtree that also removes read-only files (git pack files on Windows)."""

    def make_writable_and_retry(func, target, _exc) -> None:
        os.chmod(target, stat.S_IWRITE)
        func(target)

    if path.exists():
        shutil.rmtree(path, onexc=make_writable_and_retry)


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="https://github.com/pytest-dev/pytest")
    ap.add_argument("--ref", default="8.3.4")
    ap.add_argument("--model", default=AgentConfig().index.embedding_model)
    ap.add_argument("--edits", type=int, default=20, help="incremental samples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    source = checkout(args.repo, args.ref)
    work = HERE / ".repos" / "_bench_copy"
    rmtree(work)
    shutil.copytree(source, work, ignore=shutil.ignore_patterns(".agent"))

    cfg = AgentConfig(index=IndexConfig(embedding_model=args.model))
    embedder = FastEmbedEmbedder(args.model, cfg.model_cache_dir, cfg.index.embedding_batch_size)
    embedder.embed_query("download + warm up")  # ensure weights are present before timing
    reload = FastEmbedEmbedder(args.model, cfg.model_cache_dir, cfg.index.embedding_batch_size)
    t = time.perf_counter()
    _ = reload.dim
    model_load_s = time.perf_counter() - t

    ws = Workspace.discover(work)
    conn, _ = open_index(ws)
    indexer = Indexer(ws, conn, cfg, reload)

    t = time.perf_counter()
    stats = indexer.sync(embed=False)
    scan_chunk_fts_s = time.perf_counter() - t
    t = time.perf_counter()
    last_report = [time.perf_counter()]

    def report(done: int, total: int) -> None:
        # Progress on stderr: a long gap between lines means the machine slept or throttled,
        # which would make embed_s meaningless (wall clock includes suspend time).
        now = time.perf_counter()
        if now - last_report[0] > 30 or done == total:
            print(
                f"embedded {done}/{total} ({done / (now - t):.1f}/s)", file=sys.stderr, flush=True
            )
            last_report[0] = now

    indexer.embed_missing(stats, progress=report)
    embed_s = time.perf_counter() - t
    counts = indexer.store.counts()
    py_files = sum(1 for f in indexer.selector.scan() if f.language == "python")

    t = time.perf_counter()
    noop = indexer.sync()
    noop_sync_s = time.perf_counter() - t
    assert noop.changed == 0

    # Incremental: change the body of one function per sampled file.
    rng = random.Random(args.seed)
    candidates = sorted(
        p
        for p in work.rglob("*.py")
        if ".git" not in p.parts and "\n    return " in p.read_text("utf-8", "replace")
    )
    samples: list[float] = []
    reembedded: list[int] = []
    for path in rng.sample(candidates, min(args.edits, len(candidates))):
        text = path.read_text(encoding="utf-8")
        path.write_text(text.replace("\n    return ", "\n    _bench = 1\n    return ", 1), "utf-8")
        rel = path.relative_to(work).as_posix()
        t = time.perf_counter()
        s = indexer.update_paths([rel])
        samples.append((time.perf_counter() - t) * 1000)
        reembedded.append(s.vectors_embedded)
    conn.close()

    result = {
        "repo": args.repo,
        "ref": args.ref,
        "model": args.model,
        "machine": f"{platform.system()} {platform.machine()}, {platform.processor()}",
        "python": sys.version.split()[0],
        "files_indexed": counts["files"],
        "python_files": py_files,
        "chunks": counts["chunks"],
        "model_load_s": round(model_load_s, 2),
        "scan_chunk_fts_s": round(scan_chunk_fts_s, 2),
        "embed_s": round(embed_s, 1),
        "embed_chunks_per_s": round(counts["chunks"] / embed_s, 1),
        "initial_total_s": round(scan_chunk_fts_s + embed_s, 1),
        "noop_sync_s": round(noop_sync_s, 3),
        "incremental_samples": len(samples),
        "incremental_p50_ms": round(statistics.median(samples), 1),
        "incremental_p95_ms": round(pct(samples, 0.95), 1),
        "incremental_chunks_reembedded_median": statistics.median(reembedded),
    }
    out = HERE / "results" / f"index-{Path(args.repo).name}-{args.model.replace('/', '_')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    rmtree(work)


if __name__ == "__main__":
    main()
