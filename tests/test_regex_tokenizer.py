import pickle
from pathlib import Path

import pytest

from tokenizers.regex_tokenizer import RegexTokenizer

# GPT4Tokenizer resolves vocabularies/inputs relative to its own module file,
# so tests that need real files on disk must use these same directories.
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "training_datasets"
VOCABULARIES_DIR = REPO_ROOT / "vocabularies"


def make_tokenizer(encode_vocabulary, decode_vocabulary):
    # Bypasses __init__ (which reads a pickle from disk) so encode/decode can
    # be tested against a known vocabulary without touching the filesystem.
    tokenizer = RegexTokenizer.__new__(RegexTokenizer)
    tokenizer.encode_vocabulary = encode_vocabulary
    tokenizer.decode_vocabulary = decode_vocabulary
    # super().encode() (still BasicTokenizer's, unchanged in this file) now needs this attribute.
    tokenizer.special_tokens = {}
    return tokenizer


@pytest.fixture
def temp_input_file():
    file_name = "gpt4_test_train_input.txt"
    path = INPUTS_DIR / file_name
    # "a b a b a b a b" splits into chunks ["a", " b", " a", " b", ...] under
    # SPLIT_PATTERN, so it can also be used to test chunk-boundary behavior.
    path.write_text("a b a b a b a b", encoding="utf-8")
    yield file_name
    path.unlink(missing_ok=True)


@pytest.fixture
def temp_vocabulary_output():
    file_name = "gpt4_test_train_output.pkl"
    path = VOCABULARIES_DIR / file_name
    # Ensure no leftover file from a previous failed run interferes with the test.
    path.unlink(missing_ok=True)
    yield file_name
    path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# encode
# ---------------------------------------------------------------------------


def test_encode_merges_within_a_single_chunk():
    # "a b" splits into chunks ["a", " b"]; (32, 98) is the byte pair for " b".
    tokenizer = make_tokenizer({(32, 98): 256}, {})

    assert tokenizer.encode("a b") == [97, 256]


def test_encode_does_not_merge_a_pair_that_only_occurs_across_a_chunk_boundary():
    # "a b" splits into chunks ["a", " b"], so (97, 32) - the last byte of the
    # first chunk and the first byte of the second - never occurs within a
    # single chunk and must never be merged, even though it's in the vocabulary.
    tokenizer = make_tokenizer({(97, 32): 999}, {})

    assert tokenizer.encode("a b") == [97, 32, 98]


def test_encode_leaves_unknown_pairs_unmerged():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.encode("ab") == [97, 98]


def test_encode_empty_string_returns_empty_list():
    tokenizer = make_tokenizer({}, {})

    assert tokenizer.encode("") == []


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


def test_train_raises_valueerror_for_vocabulary_size_too_small():
    with pytest.raises(ValueError):
        RegexTokenizer.train("irrelevant.txt", 256, "irrelevant.pkl")


def test_train_raises_filenotfounderror_for_missing_input(temp_vocabulary_output):
    with pytest.raises(FileNotFoundError):
        RegexTokenizer.train(
            "this_input_does_not_exist.txt", 257, temp_vocabulary_output
        )


def test_train_raises_fileexistserror_if_vocabulary_already_exists(
    temp_input_file, temp_vocabulary_output
):
    (VOCABULARIES_DIR / temp_vocabulary_output).write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        RegexTokenizer.train(temp_input_file, 257, temp_vocabulary_output)


def test_train_raises_valueerror_when_running_out_of_pairs_to_merge(
    temp_vocabulary_output,
):
    file_name = "gpt4_test_train_short_input.txt"
    path = INPUTS_DIR / file_name
    path.write_text("ab", encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            RegexTokenizer.train(file_name, 300, temp_vocabulary_output)
    finally:
        path.unlink(missing_ok=True)


def test_train_never_merges_a_pair_that_only_occurs_across_a_chunk_boundary(
    temp_input_file, temp_vocabulary_output
):
    # (97, 32) - 'a' followed by a space - is the most frequent pair if the
    # text were treated as one flat sequence, but it only ever occurs across a
    # chunk boundary, so the actual merge chosen must be (32, 98) instead.
    RegexTokenizer.train(temp_input_file, 257, temp_vocabulary_output)

    with open(VOCABULARIES_DIR / temp_vocabulary_output, "rb") as file:
        vocabulary = pickle.load(file)

    assert vocabulary["encode"] == {(32, 98): 256}


def test_train_creates_a_usable_vocabulary(temp_input_file, temp_vocabulary_output):
    RegexTokenizer.train(temp_input_file, 257, temp_vocabulary_output)

    vocabulary_path = VOCABULARIES_DIR / temp_vocabulary_output
    assert vocabulary_path.exists()

    with open(vocabulary_path, "rb") as file:
        vocabulary = pickle.load(file)

    assert len(vocabulary["encode"]) == 1  # 257 - 256
    assert set(vocabulary["decode"].keys()) == {256}

    tokenizer = RegexTokenizer(temp_vocabulary_output)
    text = "a b a b a b a b"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_train_raises_valueerror_for_vocabulary_size_too_small_with_special_tokens():
    with pytest.raises(ValueError):
        RegexTokenizer.train(
            "irrelevant.txt", 257, "irrelevant.pkl", special_tokens={"<|x|>": 300}
        )


def test_train_raises_valueerror_for_duplicate_special_token_ids():
    with pytest.raises(ValueError):
        RegexTokenizer.train(
            "irrelevant.txt",
            1000,
            "irrelevant.pkl",
            special_tokens={"<|a|>": 300, "<|b|>": 300},
        )


def test_train_raises_valueerror_for_special_token_id_below_base_vocabulary_size():
    with pytest.raises(ValueError):
        RegexTokenizer.train(
            "irrelevant.txt", 1000, "irrelevant.pkl", special_tokens={"<|x|>": 100}
        )


def test_train_raises_valueerror_for_special_token_id_colliding_with_merge_range():
    # vocabulary_size=258 with one special token leaves exactly one merge, which
    # will be assigned token id 256 - the same id claimed by the special token.
    with pytest.raises(ValueError):
        RegexTokenizer.train(
            "irrelevant.txt", 258, "irrelevant.pkl", special_tokens={"<|x|>": 256}
        )


def test_train_adds_special_tokens_to_decode_vocabulary_after_merges(
    temp_input_file, temp_vocabulary_output
):
    RegexTokenizer.train(
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

    tokenizer = RegexTokenizer(temp_vocabulary_output)
    assert tokenizer.decode([1000]) == "<|endoftext|>"


# ---------------------------------------------------------------------------
# get_token_pair_counts_for_chunks
# ---------------------------------------------------------------------------


def test_get_token_pair_counts_for_chunks_counts_within_each_chunk_separately():
    token_chunks = [[1, 2, 3], [4, 5]]

    # (3, 4) would only occur if the chunk boundary were ignored - it must not be counted.
    assert RegexTokenizer._get_token_pair_counts_for_chunks(token_chunks) == {
        (1, 2): 1,
        (2, 3): 1,
        (4, 5): 1,
    }


def test_get_token_pair_counts_for_chunks_sums_counts_across_chunks():
    token_chunks = [[1, 2], [1, 2], [1, 2]]

    assert RegexTokenizer._get_token_pair_counts_for_chunks(token_chunks) == {(1, 2): 3}


def test_get_token_pair_counts_for_chunks_ignores_chunks_with_fewer_than_two_tokens():
    token_chunks = [[1], [2, 3], []]

    assert RegexTokenizer._get_token_pair_counts_for_chunks(token_chunks) == {(2, 3): 1}


def test_get_token_pair_counts_for_chunks_empty_list_returns_empty_dict():
    assert RegexTokenizer._get_token_pair_counts_for_chunks([]) == {}


# ---------------------------------------------------------------------------
# merge_token_pairs_for_chunks
# ---------------------------------------------------------------------------


def test_merge_token_pairs_for_chunks_merges_within_each_chunk_independently():
    token_chunks = [[1, 2, 3], [1, 2]]

    assert RegexTokenizer._merge_token_pairs_for_chunks(token_chunks, (1, 2), 99) == [
        [99, 3],
        [99],
    ]


def test_merge_token_pairs_for_chunks_does_not_merge_across_chunk_boundaries():
    # (2, 3) only occurs across the boundary between the two chunks, so it must be left alone.
    token_chunks = [[1, 2], [3, 4]]

    assert RegexTokenizer._merge_token_pairs_for_chunks(token_chunks, (2, 3), 99) == [
        [1, 2],
        [3, 4],
    ]


def test_merge_token_pairs_for_chunks_empty_list_returns_empty_list():
    assert RegexTokenizer._merge_token_pairs_for_chunks([], (1, 2), 99) == []
