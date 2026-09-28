import pytest

from code_agent.edits.parser import (
    BlockStarted,
    EditBlock,
    ParseError,
    StreamingEditParser,
    Text,
    parse_all,
    parse_edit_blocks,
)

RESPONSE = """I'll fix the expiry check.

auth/tokens.py
```python
<<<<<<< SEARCH
        return token.expires_at > 0
=======
        return token.expires_at > time.time()
>>>>>>> REPLACE
```

And add the import:

```python
<<<<<<< SEARCH
import secrets
=======
import secrets
import time
>>>>>>> REPLACE
```
Done.
"""


def test_parses_blocks_with_paths_and_reuses_previous_path():
    blocks, errors = parse_all(RESPONSE)
    assert errors == []
    assert blocks == [
        EditBlock(
            "auth/tokens.py",
            "        return token.expires_at > 0\n",
            "        return token.expires_at > time.time()\n",
            0,
        ),
        EditBlock("auth/tokens.py", "import secrets\n", "import secrets\nimport time\n", 1),
    ]


def events(chunks):
    return list(parse_edit_blocks(chunks))


def test_every_split_point_gives_identical_events():
    """Streaming must not depend on how the provider fragments tokens."""
    expected = events([RESPONSE])
    for cut in range(len(RESPONSE) + 1):
        assert events([RESPONSE[:cut], RESPONSE[cut:]]) == expected, cut


@pytest.mark.parametrize("size", [1, 2, 3, 7, 16])
def test_fixed_size_fragments(size: int):
    chunks = [RESPONSE[i : i + size] for i in range(0, len(RESPONSE), size)]
    assert events(chunks) == events([RESPONSE])


def test_block_is_emitted_as_soon_as_it_closes():
    parser = StreamingEditParser()
    first_block_end = RESPONSE.index(">>>>>>> REPLACE") + len(">>>>>>> REPLACE\n")
    seen = list(parser.feed(RESPONSE[:first_block_end]))
    assert any(isinstance(e, BlockStarted) for e in seen)
    assert [e.path for e in seen if isinstance(e, EditBlock)] == ["auth/tokens.py"]


def test_prose_passes_through_as_text():
    text = "".join(e.text for e in events([RESPONSE]) if isinstance(e, Text))
    assert "I'll fix the expiry check." in text and "Done." in text
    assert "<<<<<<<" not in text


@pytest.mark.parametrize(
    "path_line",
    ["auth/tokens.py", "`auth/tokens.py`", "**auth/tokens.py**", "File: auth/tokens.py",
     "# auth/tokens.py", "auth/tokens.py:"],
)  # fmt: skip
def test_decorated_path_lines(path_line: str):
    blocks, _ = parse_all(f"{path_line}\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n")
    assert blocks[0].path == "auth/tokens.py"


def test_crlf_model_output_is_normalized():
    blocks, _ = parse_all("x.py\r\n<<<<<<< SEARCH\r\na\r\n=======\r\nb\r\n>>>>>>> REPLACE\r\n")
    assert blocks == [EditBlock("x.py", "a\n", "b\n", 0)]


def test_empty_search_for_new_file():
    blocks, _ = parse_all("new.py\n<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE\n")
    assert blocks == [EditBlock("new.py", "", "x = 1\n", 0)]


def test_missing_trailing_newline_at_end_of_stream():
    blocks, errors = parse_all("x.py\n<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE")
    assert errors == [] and blocks[0].replace == "b\n"


def test_unterminated_block_is_reported():
    _, errors = parse_all("x.py\n<<<<<<< SEARCH\na\n=======\nb\n")
    assert len(errors) == 1 and "REPLACE" in errors[0].message


def test_block_without_path_is_reported_and_dropped():
    blocks, errors = parse_all("<<<<<<< SEARCH\na\n=======\nb\n>>>>>>> REPLACE\n")
    assert blocks == []
    assert isinstance(errors[0], ParseError) and "path" in errors[0].message


def test_unexpected_marker_is_reported():
    _, errors = parse_all("x.py\n<<<<<<< SEARCH\na\n<<<<<<< SEARCH\n=======\nb\n>>>>>>> REPLACE\n")
    assert errors and "unexpected marker" in errors[0].message


def test_path_line_must_be_adjacent_prose_not_a_sentence():
    # A sentence (contains spaces) is not a path; the path from the earlier block is reused.
    text = (
        "a.py\n<<<<<<< SEARCH\nx\n=======\ny\n>>>>>>> REPLACE\n"
        "Now the second change:\n<<<<<<< SEARCH\np\n=======\nq\n>>>>>>> REPLACE\n"
    )
    blocks, _ = parse_all(text)
    assert [b.path for b in blocks] == ["a.py", "a.py"]
