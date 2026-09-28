from code_agent.edits.matching import Ambiguous, Match, MatchKind, NoMatch, find_block

FILE = """class TokenStore:
    def issue(self, subject):
        token_id = new_id()
        self._tokens[token_id] = Token(subject, time.time() + self.ttl)
        return token_id

    def verify_token(self, token_id):
        token = self._tokens.get(token_id)
        if token is None:
            return False
        return token.expires_at > 0

    def revoke(self, token_id):
        self._tokens.pop(token_id, None)
        return None
""".splitlines()


def lines(text: str) -> list[str]:
    return text.splitlines()


def test_exact_unique():
    m = find_block(FILE, ["        return token.expires_at > 0"])
    assert isinstance(m, Match) and m.kind is MatchKind.EXACT and (m.start, m.end) == (10, 11)


def test_exact_ambiguous_is_rejected_not_first_picked():
    out = find_block(FILE + FILE, ["        return token.expires_at > 0"])
    assert isinstance(out, Ambiguous) and out.kind is MatchKind.EXACT and len(out.starts) == 2


def test_trailing_whitespace_tolerated():
    out = find_block(FILE, ["        if token is None:   ", "            return False\t"])
    assert isinstance(out, Match) and out.kind is MatchKind.WHITESPACE
    assert (out.indent_add, out.indent_remove) == ("", "")


def test_consistently_dedented_search_matches_and_replacement_is_reindented():
    needle = lines("token = self._tokens.get(token_id)\nif token is None:\n    return False")
    out = find_block(FILE, needle)
    assert isinstance(out, Match) and out.kind is MatchKind.WHITESPACE
    assert out.indent_add == "        "
    assert out.reindent(["if token:", "    pass", ""]) == [
        "        if token:",
        "            pass",
        "",
    ]


def test_over_indented_search_matches_with_removal():
    needle = ["    " + line for line in FILE[7:10]]
    out = find_block(FILE, needle)
    assert isinstance(out, Match) and out.indent_remove == "    "
    assert out.reindent(["            x = 1"]) == ["        x = 1"]


def test_inconsistent_indentation_is_not_a_whitespace_match():
    needle = ["token = self._tokens.get(token_id)", "        if token is None:"]
    out = find_block(FILE, needle)
    assert not (isinstance(out, Match) and out.kind is MatchKind.WHITESPACE)


def test_fuzzy_match_tolerates_a_small_typo():
    needle = FILE[6:11].copy()
    needle[1] = "        token = self._tokens.get(tokenid)"  # model mis-remembered one line
    out = find_block(FILE, needle)
    assert isinstance(out, Match) and out.kind is MatchKind.FUZZY
    assert (out.start, out.end) == (6, 11) and 0.9 <= out.similarity < 1.0


def test_fuzzy_is_disabled_for_short_blocks():
    out = find_block(FILE, ["        return token.expires_at > 1"])  # 1 line, 1 char off
    assert isinstance(out, NoMatch)
    assert out.closest_start == 10 and out.closest_similarity > 0.9  # still reported as a hint


def test_fuzzy_ambiguity_between_near_identical_methods():
    doubled = FILE + [line.replace("verify_token", "verify_token2") for line in FILE]
    needle = FILE[6:11].copy()
    needle[0] = "    def verify_tokenx(self, token_id):"
    out = find_block(doubled, needle)
    assert isinstance(out, Ambiguous) and out.kind is MatchKind.FUZZY


def test_below_threshold_is_no_match_with_closest_hint():
    needle = ["def something_else():", "    compute()", "    return 42"]
    out = find_block(FILE, needle)
    assert isinstance(out, NoMatch)


def test_needle_longer_than_file():
    assert isinstance(find_block(["a"], ["a", "b"]), NoMatch)
