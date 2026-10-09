from textkit.words import reading_time, word_count


def test_single_spaces():
    assert word_count("one two three") == 3


def test_any_whitespace_separates_words():
    assert word_count("one  two\nthree\tfour") == 4


def test_empty_text_has_no_words():
    assert word_count("") == 0


def test_reading_time_is_at_least_one_minute():
    assert reading_time("short") == 1
