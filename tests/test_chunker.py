from pathlib import Path

import pytest

from code_agent.index.chunker import chunk_python, chunk_text

from .conftest import FIXTURES

SAMPLE = '''"""Module doc."""
import os
from x import y

TIMEOUT = 30

# helper comment
# second line
@cache
def top(a, b):
    return a + b


# detached comment

class Store(Base):
    """A store."""
    limit = 5

    @property
    def size(self) -> int:
        return 1

    async def verify(
        self, token: str,
    ) -> bool:
        """Check."""
        return True

    class Inner:
        def deep(self): return 2

def one(): return 1

if __name__ == "__main__":
    top(1, 2)
'''


def by_symbol(chunks):
    return {c.symbol: c for c in chunks if c.symbol}


def test_definitions_become_chunks_with_qualified_symbols():
    chunks = chunk_python(SAMPLE)
    kinds = [(c.kind, c.symbol) for c in chunks]
    assert kinds == [
        ("module", None),
        ("function", "top"),
        ("module", None),  # the detached comment
        ("class", "Store"),
        ("method", "Store.size"),
        ("method", "Store.verify"),
        ("class", "Store.Inner"),
        ("method", "Store.Inner.deep"),
        ("function", "one"),
        ("module", None),
    ]
    assert by_symbol(chunks)["Store.verify"].name == "verify"


def test_line_numbers_are_one_based_inclusive():
    chunks = by_symbol(chunk_python(SAMPLE))
    assert (chunks["top"].start_line, chunks["top"].end_line) == (7, 11)
    assert (chunks["Store.verify"].start_line, chunks["Store.verify"].end_line) == (24, 28)
    assert (chunks["one"].start_line, chunks["one"].end_line) == (33, 33)


def test_decorators_and_adjacent_comments_attach_to_definition():
    top = by_symbol(chunk_python(SAMPLE))["top"]
    assert top.content.startswith("# helper comment\n# second line\n@cache\ndef top")
    size = by_symbol(chunk_python(SAMPLE))["Store.size"]
    assert size.content.lstrip().startswith("@property")


def test_comment_separated_by_blank_line_is_not_attached():
    chunks = chunk_python(SAMPLE)
    assert chunks[2].kind == "module" and chunks[2].content == "# detached comment"
    assert "detached" not in by_symbol(chunks)["Store"].content


def test_class_header_has_signatures_not_bodies():
    header = by_symbol(chunk_python(SAMPLE))["Store"].content
    assert '"""A store."""' in header and "limit = 5" in header
    assert "def size(self) -> int:" in header
    assert "async def verify(" in header and ") -> bool:" in header
    assert "class Inner:" in header
    assert "return True" not in header and "return 1" not in header
    assert header.count("...") == 3


def test_module_runs_are_contiguous_and_separate():
    chunks = [c for c in chunk_python(SAMPLE) if c.kind == "module"]
    assert chunks[0].content.startswith('"""Module doc."""') and "TIMEOUT = 30" in chunks[0].content
    assert chunks[-1].content.startswith('if __name__ == "__main__":')


def fixture_python_files() -> list[Path]:
    return sorted((FIXTURES / "sample_repo").rglob("*.py")) + sorted(
        (Path(__file__).parents[1] / "src").rglob("*.py")
    )


@pytest.mark.parametrize("path", fixture_python_files(), ids=lambda p: p.name)
def test_non_class_chunks_are_verbatim_slices_of_the_file(path: Path):
    """Invariant the edit engine relies on: chunk text can be quoted back and found in the file."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    for chunk in chunk_python(text):
        if chunk.kind == "class":
            continue
        assert chunk.content == "\n".join(lines[chunk.start_line - 1 : chunk.end_line]), chunk


def test_crlf_gives_same_chunks_as_lf():
    lf = chunk_python(SAMPLE)
    crlf = chunk_python(SAMPLE.replace("\n", "\r\n"))
    assert [(c.symbol, c.start_line, c.end_line, c.content) for c in lf] == [
        (c.symbol, c.start_line, c.end_line, c.content) for c in crlf
    ]


def test_syntax_errors_do_not_crash_and_good_code_is_still_chunked():
    broken = "def ok():\n    return 1\n\ndef broken(:\n    pass\n\nclass Fine:\n    x = 1\n"
    chunks = chunk_python(broken)
    assert "ok" in by_symbol(chunks)
    assert "Fine" in by_symbol(chunks)


def test_empty_and_whitespace_files_have_no_chunks():
    assert chunk_python("") == []
    assert chunk_python("\n\n   \n") == []


def test_one_line_class():
    (chunk,) = chunk_python("class E(Exception): pass\n")
    assert (chunk.kind, chunk.symbol, chunk.content) == ("class", "E", "class E(Exception): pass")


def test_long_module_run_is_windowed_with_overlap():
    text = "\n".join(f"X{i} = {i}" for i in range(300)) + "\n"
    chunks = chunk_python(text, max_module_lines=100, window=60, overlap=10)
    assert all(c.kind == "module" for c in chunks)
    assert chunks[0].start_line == 1 and chunks[0].end_line == 60
    assert chunks[1].start_line == 51  # 10 lines of overlap
    assert chunks[-1].end_line == 300


def test_text_windows():
    text = "\n".join(f"line {i}" for i in range(1, 131)) + "\n"
    chunks = chunk_text(text, window=60, overlap=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert all(c.kind == "text" and c.symbol is None for c in chunks)


def test_text_windows_skip_blank_regions():
    text = "a\n" + "\n" * 100 + "b\n"
    contents = [c.content.strip() for c in chunk_text(text, window=20, overlap=0)]
    assert contents == ["a", "b"]


def test_content_hash_changes_with_content():
    a = chunk_python("def f():\n    return 1\n")[0]
    b = chunk_python("def f():\n    return 2\n")[0]
    assert a.content_hash != b.content_hash
