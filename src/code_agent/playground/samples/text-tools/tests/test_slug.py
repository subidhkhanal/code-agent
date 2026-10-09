from textkit.slug import slugify


def test_simple_title():
    assert slugify("Hello World") == "hello-world"


def test_punctuation_collapses_to_one_dash():
    assert slugify("Hello, World!") == "hello-world"


def test_accents_are_removed():
    assert slugify("Café Crème") == "cafe-creme"
