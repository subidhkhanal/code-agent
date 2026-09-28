from __future__ import annotations

import os
import stat
from collections.abc import Iterator
from pathlib import Path

import pytest

from code_agent.db import connect, init_schema
from code_agent.edits import atomic
from code_agent.edits.apply import ApplyErrorCode, ApplyPlan, Planner
from code_agent.edits.atomic import WriteConflictError
from code_agent.edits.changesets import ChangeSetStore, UndoConflictError
from code_agent.edits.parser import EditBlock
from code_agent.hashing import sha256_bytes

TOKENS = b"""import time


def verify(token):
    if token is None:
        return False
    return token.expires_at > 0
"""


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    root = tmp_path / "ws"
    (root / "auth").mkdir(parents=True)
    (root / "auth" / "tokens.py").write_bytes(TOKENS)
    return root


@pytest.fixture
def store(ws: Path) -> Iterator[ChangeSetStore]:
    conn = connect(ws.parent / "state.db")
    init_schema(conn)
    yield ChangeSetStore(conn, ws)
    conn.close()


def block(path: str, search: str, replace: str, index: int = 0) -> EditBlock:
    return EditBlock(path, search, replace, index)


def read_hashes(root: Path, *paths: str) -> dict[str, str]:
    return {p: sha256_bytes((root / p).read_bytes()) for p in paths}


def plan(root: Path, blocks: list[EditBlock], hashes: dict[str, str] | None = None) -> ApplyPlan:
    if hashes is None:
        hashes = {
            b.path: sha256_bytes((root / b.path).read_bytes())
            for b in blocks
            if (root / b.path).exists()
        }
    return Planner(root).plan(blocks, hashes)


FIX = block("auth/tokens.py", "    return token.expires_at > 0\n",
            "    return token.expires_at > time.time()\n")  # fmt: skip


# -- planning ------------------------------------------------------------------------------


def test_exact_edit(ws: Path):
    p = plan(ws, [FIX])
    assert p.ok and p.results[0].kind == "exact" and p.results[0].start_line == 7
    assert p.changes["auth/tokens.py"].after == TOKENS.replace(b"> 0", b"> time.time()")
    assert (ws / "auth/tokens.py").read_bytes() == TOKENS  # planning never writes


def test_crlf_file_stays_crlf(ws: Path):
    path = ws / "auth/tokens.py"
    path.write_bytes(TOKENS.replace(b"\n", b"\r\n"))
    p = plan(ws, [FIX])
    after = p.changes["auth/tokens.py"].after
    assert after == TOKENS.replace(b"> 0", b"> time.time()").replace(b"\n", b"\r\n")


def test_multiline_replacement_in_crlf_file_uses_crlf(ws: Path):
    (ws / "auth/tokens.py").write_bytes(TOKENS.replace(b"\n", b"\r\n"))
    p = plan(ws, [block("auth/tokens.py", "import time\n", "import os\nimport time\n")])
    assert p.changes["auth/tokens.py"].after.startswith(b"import os\r\nimport time\r\n")
    assert b"\n" not in p.changes["auth/tokens.py"].after.replace(b"\r\n", b"")


def test_mixed_line_endings_untouched_lines_keep_theirs(ws: Path):
    (ws / "m.py").write_bytes(b"a = 1\r\nb = 2\nc = 3\r\n")
    p = plan(ws, [block("m.py", "b = 2\n", "b = 20\n")])
    assert p.changes["m.py"].after == b"a = 1\r\nb = 20\nc = 3\r\n"


def test_missing_final_newline_is_preserved(ws: Path):
    (ws / "n.py").write_bytes(b"x = 1\ny = 2")
    p = plan(ws, [block("n.py", "y = 2\n", "y = 3\n")])
    assert p.changes["n.py"].after == b"x = 1\ny = 3"


def test_deleting_unterminated_last_line(ws: Path):
    (ws / "n.py").write_bytes(b"x = 1\ny = 2")
    p = plan(ws, [block("n.py", "y = 2\n", "")])
    assert p.changes["n.py"].after == b"x = 1"


def test_utf8_bom_is_preserved(ws: Path):
    (ws / "b.py").write_bytes(b"\xef\xbb\xbfname = 'caf\xc3\xa9'\n")
    p = plan(ws, [block("b.py", "name = 'café'\n", "name = 'cafe'\n")])
    assert p.changes["b.py"].after == b"\xef\xbb\xbfname = 'cafe'\n"


def test_non_utf8_file_is_rejected(ws: Path):
    (ws / "l.py").write_bytes("x = 'café'\n".encode("latin-1"))
    p = plan(ws, [block("l.py", "x\n", "y\n")])
    assert p.results[0].error is ApplyErrorCode.ENCODING


def test_stale_file_is_rejected_never_patched(ws: Path):
    hashes = read_hashes(ws, "auth/tokens.py")
    (ws / "auth/tokens.py").write_bytes(TOKENS + b"# edited by the user\n")
    p = plan(ws, [FIX], hashes)
    assert not p.ok and p.results[0].error is ApplyErrorCode.STALE_FILE
    assert p.changes == {} and "re-read" in p.feedback()


def test_editing_a_file_that_was_never_read(ws: Path):
    p = plan(ws, [FIX], hashes={})
    assert p.results[0].error is ApplyErrorCode.NOT_READ


@pytest.mark.parametrize(
    "path", ["../outside.py", "auth/../../outside.py", ".env", ".git/config", ".agent/index.db",
             "secrets/key.py"],
)  # fmt: skip
def test_denied_paths(ws: Path, path: str):
    p = Planner(ws).plan([block(path, "", "x = 1\n")], {})
    assert p.results[0].error is ApplyErrorCode.PATH_DENIED
    assert not (ws.parent / "outside.py").exists()


def test_ambiguous_match_lists_candidates(ws: Path):
    (ws / "d.py").write_bytes(b"def a():\n    return 1\n\ndef b():\n    return 1\n")
    p = plan(ws, [block("d.py", "    return 1\n", "    return 2\n")])
    assert p.results[0].error is ApplyErrorCode.AMBIGUOUS_MATCH
    assert "lines 2, 5" in p.results[0].message


def test_no_match_feedback_shows_closest_lines(ws: Path):
    wrong = "def verify(tok):\n    if tok is None:\n        return None\n"
    p = plan(ws, [block("auth/tokens.py", wrong, "pass\n")])
    result = p.results[0]
    assert result.error is ApplyErrorCode.NO_MATCH
    assert "Closest lines" in result.message and "    4 | def verify(token):" in result.message


def test_fuzzy_edit_applies(ws: Path):
    typo = (
        "def verify(token):\n    if token is None:\n        return Flase\n"
        "    return token.expires_at > 0\n"
    )
    p = plan(ws, [block("auth/tokens.py", typo, "def verify(token):\n    return True\n")])
    assert p.ok and p.results[0].kind == "fuzzy"


def test_multiple_blocks_same_file_apply_in_order(ws: Path):
    p = plan(ws, [FIX, block("auth/tokens.py", "import time\n", "import os\nimport time\n", 1)])
    after = p.changes["auth/tokens.py"].after
    assert after.startswith(b"import os\nimport time\n") and b"> time.time()" in after


def test_all_or_nothing_across_files(ws: Path):
    (ws / "other.py").write_bytes(b"a = 1\n")
    p = plan(ws, [FIX, block("other.py", "does not exist\n", "b\n", 1)])
    assert not p.ok and p.changes == {}
    assert [r.ok for r in p.results] == [True, False]


def test_create_new_file(ws: Path):
    p = plan(ws, [block("pkg/new.py", "", "def f():\n    return 1\n")])
    assert p.ok and p.changes["pkg/new.py"].before is None
    assert p.changes["pkg/new.py"].after == b"def f():\n    return 1\n"


def test_empty_search_on_existing_file(ws: Path):
    p = plan(ws, [block("auth/tokens.py", "", "x = 1\n")])
    assert p.results[0].error is ApplyErrorCode.EMPTY_SEARCH


def test_non_empty_search_on_missing_file(ws: Path):
    p = plan(ws, [block("missing.py", "a\n", "b\n")])
    assert p.results[0].error is ApplyErrorCode.FILE_MISSING


# -- writing and undo ------------------------------------------------------------------------


def test_apply_and_undo_roundtrip(ws: Path, store: ChangeSetStore):
    p = plan(ws, [FIX, block("pkg/new.py", "", "x = 1\n", 1)])
    cs = store.apply(p)
    assert b"time.time()" in (ws / "auth/tokens.py").read_bytes()
    assert (ws / "pkg/new.py").exists()
    result = store.undo()
    assert result.change_set_id == cs and set(result.restored) == {"auth/tokens.py", "pkg/new.py"}
    assert (ws / "auth/tokens.py").read_bytes() == TOKENS
    assert not (ws / "pkg/new.py").exists()
    with pytest.raises(LookupError):
        store.undo()


def test_no_temp_files_left_behind(ws: Path, store: ChangeSetStore):
    store.apply(plan(ws, [FIX]))
    assert not list(ws.rglob("*.agent-tmp"))


def test_write_refuses_if_file_changed_after_planning(ws: Path, store: ChangeSetStore):
    p = plan(ws, [FIX])
    (ws / "auth/tokens.py").write_bytes(TOKENS + b"# user saved while reviewing\n")
    with pytest.raises(WriteConflictError):
        store.apply(p)
    assert (ws / "auth/tokens.py").read_bytes().endswith(b"# user saved while reviewing\n")


def test_failure_mid_write_rolls_back_earlier_files(ws: Path, store: ChangeSetStore, monkeypatch):
    (ws / "b.py").write_bytes(b"b = 1\n")
    p = plan(ws, [FIX, block("b.py", "b = 1\n", "b = 2\n", 1)])
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("disk full")
        return real_replace(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky_replace)
    with pytest.raises(OSError, match="disk full"):
        store.apply(p)
    monkeypatch.setattr(atomic.os, "replace", real_replace)
    assert (ws / "auth/tokens.py").read_bytes() == TOKENS
    assert (ws / "b.py").read_bytes() == b"b = 1\n"
    assert not list(ws.rglob("*.agent-tmp"))
    status = store.conn.execute("SELECT status FROM change_sets").fetchone()[0]
    assert status == "REJECTED"


def test_undo_refuses_to_discard_later_user_edits(ws: Path, store: ChangeSetStore):
    store.apply(plan(ws, [FIX]))
    (ws / "auth/tokens.py").write_bytes(b"# user rewrote the file\n")
    with pytest.raises(UndoConflictError) as exc:
        store.undo()
    assert exc.value.paths == ["auth/tokens.py"]
    assert (ws / "auth/tokens.py").read_bytes() == b"# user rewrote the file\n"


def test_undo_after_crash_restores_only_files_that_were_written(ws: Path, store: ChangeSetStore):
    (ws / "b.py").write_bytes(b"b = 1\n")
    p = plan(ws, [FIX, block("b.py", "b = 1\n", "b = 2\n", 1)])
    cs = store.apply(p)
    # Simulate a crash mid-write: record still PROPOSED, only tokens.py was replaced.
    store.conn.execute("UPDATE change_sets SET status = 'PROPOSED' WHERE id = ?", (cs,))
    store.conn.commit()
    (ws / "b.py").write_bytes(b"b = 1\n")
    result = store.undo()
    assert result.restored == ["auth/tokens.py"] and result.already_original == ["b.py"]
    assert (ws / "auth/tokens.py").read_bytes() == TOKENS


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_permissions_are_preserved(ws: Path, store: ChangeSetStore):
    script = ws / "run.py"
    script.write_bytes(b"print('hi')\n")
    script.chmod(0o755)
    store.apply(plan(ws, [block("run.py", "print('hi')\n", "print('hello')\n")]))
    assert stat.S_IMODE(script.stat().st_mode) == 0o755


def test_copied_line_number_gutter_is_stripped(ws: Path):
    search = "    7 |     return token.expires_at > 0\n"
    replace = "    7 |     return token.expires_at > time.time()\n"
    p = plan(ws, [block("auth/tokens.py", search, replace)])
    assert p.ok, p.feedback()
    assert b"    return token.expires_at > time.time()\n" in p.changes["auth/tokens.py"].after


def test_partial_gutter_is_not_stripped(ws: Path):
    search = "    7 |     return token.expires_at > 0\n    return x\n"
    p = plan(ws, [block("auth/tokens.py", search, "pass\n")])
    assert not p.ok


@pytest.mark.parametrize("where", ["search", "replace"])
def test_blocks_touching_redacted_secrets_are_refused(ws: Path, where: str):
    (ws / "settings.py").write_bytes(b"DEBUG = True\nAPI_KEY = 'real-secret-value'\n")
    search = "API_KEY = '[REDACTED:assigned_secret]'\n" if where == "search" else "DEBUG = True\n"
    replace = "DEBUG = False\n" + (
        "API_KEY = '[REDACTED:assigned_secret]'\n" if where == "replace" else ""
    )
    p = plan(ws, [block("settings.py", search, replace)])
    assert p.results[0].error is ApplyErrorCode.REDACTED_CONTENT
    assert (ws / "settings.py").read_bytes().endswith(b"'real-secret-value'\n")
