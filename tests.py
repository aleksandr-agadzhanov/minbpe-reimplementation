from utils import (
    get_byte_pair_counts,
    get_most_frequent_byte_pair,
    merge_most_frequent_byte_pairs,
)

import pytest


def test_empty_tokens():
    assert get_byte_pair_counts([]) == {}


def test_single_token():
    assert get_byte_pair_counts([1]) == {}


def test_two_tokens():
    assert get_byte_pair_counts([1, 2]) == {(1, 2): 1}


def test_no_repeated_pairs():
    assert get_byte_pair_counts([1, 2, 3, 4]) == {
        (1, 2): 1,
        (2, 3): 1,
        (3, 4): 1,
    }


def test_repeated_pairs_are_counted():
    assert get_byte_pair_counts([1, 2, 1, 2]) == {
        (1, 2): 2,
        (2, 1): 1,
    }


def test_all_same_token():
    assert get_byte_pair_counts([5, 5, 5, 5]) == {(5, 5): 3}


def test_does_not_mutate_input():
    tokens = [1, 2, 3]
    get_byte_pair_counts(tokens)
    assert tokens == [1, 2, 3]


def test_get_most_frequent_byte_pair_single_pair():
    assert get_most_frequent_byte_pair({(1, 2): 1}) == (1, 2)


def test_get_most_frequent_byte_pair_clear_winner():
    assert get_most_frequent_byte_pair(
        {(1, 2): 1, (2, 3): 5, (3, 4): 2}
    ) == (2, 3)


def test_get_most_frequent_byte_pair_breaks_ties_by_first_occurrence():
    assert get_most_frequent_byte_pair(
        {(1, 2): 3, (2, 3): 3}
    ) == (1, 2)


def test_merge_replaces_single_occurrence():
    assert merge_most_frequent_byte_pairs([1, 2, 3], 99) == [99, 3]


def test_merge_replaces_multiple_occurrences():
    assert merge_most_frequent_byte_pairs([5, 1, 2, 1, 2, 5], 99) == [
        5,
        99,
        99,
        5,
    ]


def test_merge_leaves_unmatched_tokens_untouched():
    assert merge_most_frequent_byte_pairs([1, 2, 9, 1, 2], 99) == [99, 9, 99]


def test_merge_raises_on_empty_tokens():
    with pytest.raises(ValueError):
        merge_most_frequent_byte_pairs([], 99)


def test_merge_raises_on_single_token():
    with pytest.raises(ValueError):
        merge_most_frequent_byte_pairs([1], 99)


def test_merge_handles_pair_start_as_last_token():
    # The last token equals `byte_pair[0]` but has no following token to
    # pair with, so it should be copied through unchanged rather than
    # raising an IndexError.
    assert merge_most_frequent_byte_pairs([3, 4, 3], 99) == [99, 3]
