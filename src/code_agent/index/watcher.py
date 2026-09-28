"""Watch the workspace and re-index changed files.

watchdog delivers events on its own thread; we only push paths onto a queue there. All SQLite
work happens on the calling thread, which drains the queue after a short quiet period
(debounce), so an editor's save-rename-chmod burst or a `git checkout` becomes one batch.
"""

from __future__ import annotations

import queue
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from code_agent.index.files import AGENTIGNORE
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.workspace import STATE_DIR_NAME

# Changes to these mean the *set* of indexable files may have changed: do a full sync.
RESCAN_TRIGGERS = frozenset({".gitignore", AGENTIGNORE})


class _Collector(FileSystemEventHandler):
    def __init__(self, root: Path, out: queue.Queue[str]) -> None:
        self.root = root
        self.out = out

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.event_type in ("opened", "closed", "closed_no_write"):
            return
        for raw in (event.src_path, getattr(event, "dest_path", "")):
            if not raw:
                continue
            path = raw.decode() if isinstance(raw, bytes) else raw
            try:
                rel = Path(path).resolve().relative_to(self.root).as_posix()
            except ValueError:
                continue
            top = rel.split("/", 1)[0]
            if top in (".git", STATE_DIR_NAME):
                continue
            self.out.put(rel)


def watch(
    indexer: Indexer,
    *,
    debounce_s: float = 0.5,
    max_wait_s: float = 5.0,
    on_batch: Callable[[IndexStats], None] | None = None,
    stop: Callable[[], bool] = lambda: False,
) -> None:
    """Block until `stop()` returns True (or KeyboardInterrupt), indexing changes as they come."""
    root = indexer.workspace.root
    events: queue.Queue[str] = queue.Queue()
    observer = Observer()
    observer.schedule(_Collector(root, events), str(root), recursive=True)
    observer.start()
    try:
        while not stop():
            try:
                first = events.get(timeout=0.2)
            except queue.Empty:
                continue
            batch = {first}
            hard_deadline = time.monotonic() + max_wait_s
            deadline = time.monotonic() + debounce_s
            while (remaining := min(deadline, hard_deadline) - time.monotonic()) > 0:
                try:
                    batch.add(events.get(timeout=remaining))
                    deadline = time.monotonic() + debounce_s  # extend while events keep coming
                except queue.Empty:
                    break
            if any(p.rsplit("/", 1)[-1] in RESCAN_TRIGGERS for p in batch):
                stats = indexer.sync()
            else:
                stats = indexer.update_paths(batch)
            if on_batch and stats.changed:
                on_batch(stats)
    finally:
        observer.stop()
        observer.join()
