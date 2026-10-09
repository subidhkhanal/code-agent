"""Word counting and reading-time estimates."""

import math

WORDS_PER_MINUTE = 200


def word_count(text: str) -> int:
    return len(text.split(" "))


def reading_time(text: str) -> int:
    return max(1, math.ceil(word_count(text) / WORDS_PER_MINUTE))
