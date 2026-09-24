"""Tests for count_words. One of them fails against the shipped code."""
from wordcount import count_words


def test_empty_text_has_no_words() -> None:
    assert count_words('') == 0


def test_single_line() -> None:
    assert count_words('the quick brown fox') == 4


def test_repeated_spaces_do_not_count() -> None:
    assert count_words('a  b   c') == 3


def test_words_across_newlines() -> None:
    assert count_words('one two\nthree\nfour five\n') == 5
