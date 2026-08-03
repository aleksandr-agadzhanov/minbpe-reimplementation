import pickle
from pathlib import Path

import pytest

from tokenizers.basic_tokenizer import BasicTokenizer

# BasicTokenizer resolves vocabularies/inputs relative to its own module file,
# so tests that need real files on disk must use these same directories.
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "training_datasets"
VOCABULARIES_DIR = REPO_ROOT / "vocabularies"


def make_tokenizer(encode_vocabulary, decode_vocabulary, special_tokens=None):
    # Bypasses __init__ (which reads a pickle from disk) so encode/decode can
    # be tested against a known vocabulary without touching the filesystem.
    tokenizer = BasicTokenizer.__new__(BasicTokenizer)
    tokenizer.encode_vocabulary = encode_vocabulary
    tokenizer.decode_vocabulary = decode_vocabulary
    tokenizer.special_tokens = special_tokens or {}
    return tokenizer


@pytest.fixture
def temp_vocabulary_file():
    file_name = "test_vocabulary.pkl"
    path = VOCABULARIES_DIR / file_name
    vocabulary = {
        "encode": {(97, 98): 256},
        "decode": {256: [97, 98]},
        "special_tokens": {},
    }
    with open(path, "wb") as file:
        pickle.dump(vocabulary, file)
    yield file_name, vocabulary
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_input_file():
    file_name = "test_train_input.txt"
    path = INPUTS_DIR / file_name
    path.write_text("abababab abababab abababab", encoding="utf-8")
    yield file_name
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_vocabulary_output():
    file_name = "test_train_output.pkl"
    path = VOCABULARIES_DIR / file_name
    # Ensure no leftover file from a previous failed run interferes with the test.
    path.unlink(missing_ok=True)
    yield file_name
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_loads_existing_vocabulary(temp_vocabulary_file):
    file_name, vocabulary = temp_vocabulary_file

    tokenizer = BasicTokenizer(file_name)

    assert tokenizer.encode_vocabulary == vocabulary["encode"]
    assert tokenizer.decode_vocabulary == vocabulary["decode"]
    assert tokenizer.special_tokens == vocabulary["special_tokens"]


def test_init_raises_filenotfounderror_for_missing_vocabulary():
    with pytest.raises(FileNotFoundError):
        BasicTokenizer("this_vocabulary_does_not_exist.pkl")


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def test_encode_applies_merges_in_learned_order():
    # Learn (a,b) -> 256 first, then (256,c) -> 257, so "abc" should fully collapse.
    encode_vocabulary = {(97, 98): 256, (256, 99): 257}
    tokenizer = make_tokenizer(encode_vocabulary, {})

    assert tokenizer.encode("abc") == [257]


def test_encode_leaves_unknown_pairs_unmerged():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.encode("ab") == [97, 98]


def test_encode_empty_string_returns_empty_list():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.encode("") == []


def test_encode_single_character_returns_single_token():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.encode("h") == [104]


def test_encode_maps_special_token_text_directly_to_reserved_id():
    tokenizer = make_tokenizer({}, {}, {"<|endoftext|>": 1000})

    assert tokenizer.encode("<|endoftext|>") == [1000]


def test_encode_preserves_special_tokens_embedded_in_normal_text():
    tokenizer = make_tokenizer({}, {}, {"<|endoftext|>": 1000})

    assert tokenizer.encode("a<|endoftext|>b") == [97, 1000, 98]


def test_encode_prefers_longer_special_token_when_tokens_overlap():
    tokenizer = make_tokenizer(
        {},
        {},
        {"<|end|>": 1000, "<|endoftext|>": 1001},
    )

    assert tokenizer.encode("<|endoftext|>") == [1001]


# ---------------------------------------------------------------------------
# decode
# ---------------------------------------------------------------------------


def test_decode_expands_merged_tokens():
    tokenizer = make_tokenizer({}, {256: [97, 98]})

    assert tokenizer.decode([256, 99]) == "abc"


def test_decode_raw_bytes_only():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.decode([104, 105]) == "hi"


def test_decode_raises_keyerror_for_unknown_token_id():
    tokenizer = make_tokenizer({}, {})

    with pytest.raises(KeyError):
        tokenizer.decode([999])


def test_decode_replaces_invalid_utf8_instead_of_raising():
    tokenizer = make_tokenizer({}, {})

    # A lone 0x80 byte is not valid UTF-8 on its own.
    assert tokenizer.decode([0x80]) == "\ufffd"


def test_encode_decode_roundtrip():
    encode_vocabulary = {(97, 98): 256}
    decode_vocabulary = {256: [97, 98]}
    tokenizer = make_tokenizer(encode_vocabulary, decode_vocabulary)
    text = "abab hello abab"

    assert tokenizer.decode(tokenizer.encode(text)) == text


# ---------------------------------------------------------------------------
# decode_as_list
# ---------------------------------------------------------------------------


def test_decode_as_list_expands_merged_tokens():
    tokenizer = make_tokenizer({}, {256: [97, 98]})

    assert tokenizer.decode_as_list([256, 99]) == ["ab", "c"]


def test_decode_as_list_raw_bytes_only():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.decode_as_list([104, 105]) == ["h", "i"]


def test_decode_as_list_raises_keyerror_for_unknown_token_id():
    tokenizer = make_tokenizer({}, {})

    with pytest.raises(KeyError):
        tokenizer.decode_as_list([999])


def test_decode_as_list_replaces_invalid_utf8_instead_of_raising():
    tokenizer = make_tokenizer({}, {})

    # A lone 0x80 byte is not valid UTF-8 on its own.
    assert tokenizer.decode_as_list([0x80]) == ["\ufffd"]


def test_decode_as_list_empty_input_returns_empty_list():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.decode_as_list([]) == []


# ---------------------------------------------------------------------------
# _expand_token
# ---------------------------------------------------------------------------


def test_expand_token_returns_single_byte_for_raw_token():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer._expand_token(104) == [104]


def test_expand_token_returns_raw_bytes_for_merged_token():
    tokenizer = make_tokenizer({}, {256: [97, 98]})

    assert tokenizer._expand_token(256) == [97, 98]


def test_expand_token_raises_keyerror_for_unknown_token_id():
    tokenizer = make_tokenizer({}, {})

    with pytest.raises(KeyError):
        tokenizer._expand_token(999)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_raises_valueerror_for_vocabulary_size_too_small():
    with pytest.raises(ValueError):
        BasicTokenizer.train("irrelevant.txt", 256, "irrelevant.pkl")


def test_train_raises_filenotfounderror_for_missing_input(temp_vocabulary_output):
    with pytest.raises(FileNotFoundError):
        BasicTokenizer.train(
            "this_input_does_not_exist.txt", 257, temp_vocabulary_output
        )


def test_train_raises_fileexistserror_if_vocabulary_already_exists(
    temp_input_file, temp_vocabulary_output
):
    (VOCABULARIES_DIR / temp_vocabulary_output).write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        BasicTokenizer.train(temp_input_file, 257, temp_vocabulary_output)


def test_train_raises_valueerror_when_running_out_of_pairs_to_merge(
    temp_vocabulary_output,
):
    file_name = "test_train_short_input.txt"
    path = INPUTS_DIR / file_name
    path.write_text("ab", encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            BasicTokenizer.train(file_name, 300, temp_vocabulary_output)
    finally:
        path.unlink(missing_ok=True)


def test_train_creates_a_usable_vocabulary(temp_input_file, temp_vocabulary_output):
    BasicTokenizer.train(temp_input_file, 258, temp_vocabulary_output)

    vocabulary_path = VOCABULARIES_DIR / temp_vocabulary_output
    assert vocabulary_path.exists()

    with open(vocabulary_path, "rb") as file:
        vocabulary = pickle.load(file)

    assert len(vocabulary["encode"]) == 2  # 258 - 256
    assert set(vocabulary["encode"].values()) == {256, 257}
    assert set(vocabulary["decode"].keys()) == {256, 257}

    tokenizer = BasicTokenizer(temp_vocabulary_output)
    text = "abababab"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_train_raises_valueerror_for_vocabulary_size_too_small_with_special_tokens():
    with pytest.raises(ValueError):
        BasicTokenizer.train(
            "irrelevant.txt", 257, "irrelevant.pkl", special_tokens={"<|x|>": 300}
        )


def test_train_raises_valueerror_for_duplicate_special_token_ids():
    with pytest.raises(ValueError):
        BasicTokenizer.train(
            "irrelevant.txt",
            1000,
            "irrelevant.pkl",
            special_tokens={"<|a|>": 300, "<|b|>": 300},
        )


def test_train_raises_valueerror_for_special_token_id_below_base_vocabulary_size():
    with pytest.raises(ValueError):
        BasicTokenizer.train(
            "irrelevant.txt", 1000, "irrelevant.pkl", special_tokens={"<|x|>": 100}
        )


def test_train_raises_valueerror_for_special_token_id_colliding_with_merge_range():
    # vocabulary_size=258 with one special token leaves exactly one merge, which
    # will be assigned token id 256 - the same id claimed by the special token.
    with pytest.raises(ValueError):
        BasicTokenizer.train(
            "irrelevant.txt", 258, "irrelevant.pkl", special_tokens={"<|x|>": 256}
        )


def test_train_adds_special_tokens_to_decode_vocabulary_after_merges(
    temp_input_file, temp_vocabulary_output
):
    BasicTokenizer.train(
        temp_input_file,
        258,
        temp_vocabulary_output,
        special_tokens={"<|endoftext|>": 1000},
    )

    with open(VOCABULARIES_DIR / temp_vocabulary_output, "rb") as file:
        vocabulary = pickle.load(file)

    # A merge still fills id 256; the special token keeps its caller-chosen id.
    assert set(vocabulary["encode"].values()) == {256}
    assert 1000 not in vocabulary["encode"].values()
    assert vocabulary["decode"][1000] == list(b"<|endoftext|>")
    assert vocabulary["special_tokens"] == {"<|endoftext|>": 1000}

    tokenizer = BasicTokenizer(temp_vocabulary_output)
    assert tokenizer.decode([1000]) == "<|endoftext|>"


# ---------------------------------------------------------------------------
# merge_token_pairs
# ---------------------------------------------------------------------------


def test_merge_token_pairs_merges_non_overlapping_occurrences():
    assert BasicTokenizer._merge_token_pairs([1, 2, 1, 2, 3], (1, 2), 99) == [99, 99, 3]


def test_merge_token_pairs_handles_adjacent_occurrences_of_same_pair():
    # "AAAA" merged left-to-right, non-overlapping, leaves 2 merged tokens.
    assert BasicTokenizer._merge_token_pairs([1, 1, 1, 1], (1, 1), 99) == [99, 99]


def test_merge_token_pairs_leaves_unmatched_tokens_unchanged():
    assert BasicTokenizer._merge_token_pairs([1, 2, 3], (4, 5), 99) == [1, 2, 3]


def test_merge_token_pairs_empty_input_returns_empty_list():
    assert BasicTokenizer._merge_token_pairs([], (1, 2), 99) == []


def test_merge_token_pairs_and_update_counts_matches_two_pass_behavior():
    tokens = [1, 2, 1, 2, 3]

    merged_tokens, updated_counts = BasicTokenizer._merge_token_pairs_and_update_counts(
        tokens, (1, 2), 99
    )

    expected_merged_tokens = BasicTokenizer._merge_token_pairs(tokens, (1, 2), 99)
    expected_counts = BasicTokenizer._get_token_pair_counts(expected_merged_tokens)

    assert merged_tokens == expected_merged_tokens
    assert updated_counts == expected_counts


def test_merge_token_pairs_and_update_counts_no_match_keeps_tokens_and_recounts():
    tokens = [1, 2, 3]

    merged_tokens, updated_counts = BasicTokenizer._merge_token_pairs_and_update_counts(
        tokens, (4, 5), 99
    )

    assert merged_tokens == tokens
    assert updated_counts == BasicTokenizer._get_token_pair_counts(tokens)


def test_merge_token_pairs_and_update_counts_adjacent_matches_are_counted_correctly():
    tokens = [1, 1, 1, 1]

    merged_tokens, updated_counts = BasicTokenizer._merge_token_pairs_and_update_counts(
        tokens, (1, 1), 99
    )

    assert merged_tokens == [99, 99]
    assert updated_counts == {(99, 99): 1}


def test_merge_token_pairs_and_update_counts_empty_input_returns_empty_results():
    merged_tokens, updated_counts = BasicTokenizer._merge_token_pairs_and_update_counts(
        [], (1, 2), 99
    )

    assert merged_tokens == []
    assert updated_counts == {}


# ---------------------------------------------------------------------------
# get_token_pair_counts
# ---------------------------------------------------------------------------


def test_get_token_pair_counts_counts_overlapping_pairs():
    assert BasicTokenizer._get_token_pair_counts([1, 2, 1, 2, 3]) == {
        (1, 2): 2,
        (2, 1): 1,
        (2, 3): 1,
    }


def test_get_token_pair_counts_empty_for_fewer_than_two_tokens():
    assert BasicTokenizer._get_token_pair_counts([]) == {}
    assert BasicTokenizer._get_token_pair_counts([1]) == {}


# ---------------------------------------------------------------------------
# get_most_frequent_token_pair
# ---------------------------------------------------------------------------


def test_get_most_frequent_token_pair_returns_highest_count():
    token_pair_counts = {(1, 2): 3, (2, 3): 5, (3, 4): 1}

    assert BasicTokenizer._get_most_frequent_token_pair(token_pair_counts) == (2, 3)


def test_get_most_frequent_token_pair_raises_valueerror_for_empty_counts():
    with pytest.raises(ValueError):
        BasicTokenizer._get_most_frequent_token_pair({})
