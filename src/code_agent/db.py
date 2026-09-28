"""SQLite connection and schema.

One database file per workspace (`<repo>/.agent/index.db`) holds the index *and* the agent's task,
approval, audit and undo tables. Keeping them in one file means one transaction can cover e.g.
"apply change set + write audit row", and there is no second store to keep consistent.

The vector table (`chunks_vec`, sqlite-vec) is created lazily because its dimension depends on the
configured embedding model; see `index.store.IndexStore.ensure_vector_table`.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repositories (
    repo_id        TEXT PRIMARY KEY,
    workspace_root TEXT NOT NULL UNIQUE,
    branch         TEXT,
    revision       TEXT,
    indexed_at     TEXT
);

-- One row per indexed file: the unit of incremental change detection.
CREATE TABLE IF NOT EXISTS indexed_files (
    repo_id      TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path    TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    size         INTEGER NOT NULL,
    mtime_ns     INTEGER NOT NULL,
    hashed_at_ns INTEGER NOT NULL,  -- wall clock when content was hashed (racy-mtime guard)
    language     TEXT,
    indexed_at   TEXT NOT NULL,
    PRIMARY KEY (repo_id, file_path)
);

CREATE TABLE IF NOT EXISTS file_chunks (
    id            INTEGER PRIMARY KEY,
    repo_id       TEXT NOT NULL REFERENCES repositories(repo_id) ON DELETE CASCADE,
    file_path     TEXT NOT NULL,
    chunk_index   INTEGER NOT NULL,
    symbol        TEXT,             -- qualified within the file, e.g. "TokenStore.verify"
    name          TEXT,             -- last component, e.g. "verify" (exact-symbol lookup)
    kind          TEXT NOT NULL,    -- module | function | class | method | text
    start_line    INTEGER NOT NULL, -- 1-based, inclusive
    end_line      INTEGER NOT NULL,
    content       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    embed_hash    TEXT NOT NULL,    -- hash of (model id, text that was embedded): vector reuse key
    language      TEXT,
    last_modified INTEGER NOT NULL, -- file mtime_ns when chunked
    has_vector    INTEGER NOT NULL DEFAULT 0,
    UNIQUE (repo_id, file_path, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file   ON file_chunks(repo_id, file_path);
CREATE INDEX IF NOT EXISTS idx_chunks_name   ON file_chunks(repo_id, name);
CREATE INDEX IF NOT EXISTS idx_chunks_symbol ON file_chunks(repo_id, symbol);
CREATE INDEX IF NOT EXISTS idx_chunks_vector ON file_chunks(has_vector);

-- BM25 index. rowid = file_chunks.id. `body` holds the chunk text plus camelCase splits.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    symbol, path, body,
    tokenize = 'porter unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS agent_tasks (
    request_id      TEXT PRIMARY KEY,
    conversation_id TEXT,
    repo_id         TEXT NOT NULL REFERENCES repositories(repo_id),
    base_revision   TEXT,
    task            TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN (
                        'QUEUED', 'RUNNING', 'WAITING_FOR_APPROVAL',
                        'SUCCEEDED', 'FAILED', 'CANCELLED')),
    budgets_json    TEXT NOT NULL,
    usage_json      TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT UNIQUE,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id    TEXT PRIMARY KEY,
    request_id     TEXT NOT NULL REFERENCES agent_tasks(request_id),
    command        TEXT NOT NULL,
    classification TEXT NOT NULL,
    scope          TEXT NOT NULL CHECK (scope IN ('once', 'session')),
    decision       TEXT NOT NULL CHECK (decision IN ('approved', 'denied')),
    decided_at     TEXT NOT NULL,
    consumed_at    TEXT,            -- set when a 'once' approval is used
    revoked_at     TEXT
);

CREATE TABLE IF NOT EXISTS tool_audit_log (
    tool_call_id      TEXT PRIMARY KEY,
    request_id        TEXT REFERENCES agent_tasks(request_id),
    approval_id       TEXT REFERENCES approvals(approval_id),
    tool_name         TEXT NOT NULL,
    actor             TEXT NOT NULL,  -- model | user | system
    redacted_args     TEXT NOT NULL,
    output_hash       TEXT,
    redaction_applied INTEGER NOT NULL DEFAULT 0,
    idempotency_key   TEXT UNIQUE,
    status            TEXT,
    executed_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_sets (
    id         TEXT PRIMARY KEY,
    request_id TEXT REFERENCES agent_tasks(request_id),
    status     TEXT NOT NULL CHECK (status IN (
                   'PROPOSED', 'VALIDATED', 'APPLIED', 'REJECTED', 'UNDONE')),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS change_set_files (
    change_set_id  TEXT NOT NULL REFERENCES change_sets(id) ON DELETE CASCADE,
    file_path      TEXT NOT NULL,
    before_hash    TEXT,            -- NULL when the change creates the file
    after_hash     TEXT,            -- NULL when the change deletes the file
    before_content BLOB,
    after_content  BLOB,
    before_mode    INTEGER,
    PRIMARY KEY (change_set_id, file_path)
);
"""


class SchemaMismatchError(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: Path | str, *, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with sqlite-vec loaded. Read-only connections are for parallel search."""
    if read_only:
        conn = sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    try:
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        if not read_only:
            # WAL lets search connections read while the indexer/watcher writes.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
    except BaseException:
        conn.close()  # e.g. "file is not a database": don't leak the handle (Windows file locks)
        raise
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create tables if missing; refuse to run against a DB from a different schema version."""
    conn.executescript(SCHEMA)
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),)
        )
        conn.commit()
    elif int(row["value"]) != SCHEMA_VERSION:
        raise SchemaMismatchError(
            f"index schema v{row['value']} != expected v{SCHEMA_VERSION}; rebuild needed"
        )


def get_meta(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return None if row is None else row["value"]


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
