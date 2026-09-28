from __future__ import annotations

import threading
import time
from pathlib import Path

from code_agent.config import AgentConfig
from code_agent.db import connect
from code_agent.index.embeddings import HashingEmbedder
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.index.store import open_index
from code_agent.index.watcher import watch
from code_agent.workspace import Workspace


def test_watcher_reindexes_changed_files(repo: Path, cfg: AgentConfig):
    ws = Workspace.discover(repo)
    batches: list[IndexStats] = []
    ready = threading.Event()
    stop = threading.Event()

    def run() -> None:
        # SQLite connections are per-thread, so the watcher thread opens its own.
        conn, _ = open_index(ws)
        indexer = Indexer(ws, conn, cfg, HashingEmbedder())
        indexer.sync()
        ready.set()
        try:
            watch(indexer, debounce_s=0.2, on_batch=batches.append, stop=stop.is_set)
        finally:
            conn.close()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(30)
    time.sleep(0.5)  # let the observer start

    (repo / "watched.py").write_text("def watched_function():\n    return 42\n")
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not any(b.added for b in batches):
        time.sleep(0.1)
    stop.set()
    thread.join(10)

    assert any(b.added == 1 for b in batches), batches
    conn = connect(ws.index_path)
    try:
        row = conn.execute(
            "SELECT has_vector FROM file_chunks WHERE symbol = 'watched_function'"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None and row[0] == 1
