import pickle
from pathlib import Path

import pytest

from tokenizers.gpt4_tokenizer import GPT4Tokenizer

# GPT4Tokenizer resolves vocabularies/inputs relative to its own module file,
# so tests that need real files on disk must use these same directories.
REPO_ROOT = Path(__file__).resolve().parent.parent
INPUTS_DIR = REPO_ROOT / "training_datasets"
VOCABULARIES_DIR = REPO_ROOT / "vocabularies"


def make_tokenizer(encode_vocabulary, decode_vocabulary):
    # Bypasses __init__ (which reads a pickle from disk) so encode/decode can
    # be tested against a known vocabulary without touching the filesystem.
    tokenizer = GPT4Tokenizer.__new__(GPT4Tokenizer)
    tokenizer.encode_vocabulary = encode_vocabulary
    tokenizer.decode_vocabulary = decode_vocabulary
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
        GPT4Tokenizer.train("irrelevant.txt", 256, "irrelevant.pkl")


def test_train_raises_filenotfounderror_for_missing_input(temp_vocabulary_output):
    with pytest.raises(FileNotFoundError):
        GPT4Tokenizer.train(
            "this_input_does_not_exist.txt", 257, temp_vocabulary_output
        )


def test_train_raises_fileexistserror_if_vocabulary_already_exists(
    temp_input_file, temp_vocabulary_output
):
    (VOCABULARIES_DIR / temp_vocabulary_output).write_bytes(b"placeholder")

    with pytest.raises(FileExistsError):
        GPT4Tokenizer.train(temp_input_file, 257, temp_vocabulary_output)


def test_train_raises_valueerror_when_running_out_of_pairs_to_merge(
    temp_vocabulary_output,
):
    file_name = "gpt4_test_train_short_input.txt"
    path = INPUTS_DIR / file_name
    path.write_text("ab", encoding="utf-8")
    try:
        with pytest.raises(ValueError):
            GPT4Tokenizer.train(file_name, 300, temp_vocabulary_output)
    finally:
        path.unlink(missing_ok=True)


def test_train_never_merges_a_pair_that_only_occurs_across_a_chunk_boundary(
    temp_input_file, temp_vocabulary_output
):
    # (97, 32) - 'a' followed by a space - is the most frequent pair if the
    # text were treated as one flat sequence, but it only ever occurs across a
    # chunk boundary, so the actual merge chosen must be (32, 98) instead.
    GPT4Tokenizer.train(temp_input_file, 257, temp_vocabulary_output)

    with open(VOCABULARIES_DIR / temp_vocabulary_output, "rb") as file:
        vocabulary = pickle.load(file)

    assert vocabulary["encode"] == {(32, 98): 256}


def test_train_creates_a_usable_vocabulary(temp_input_file, temp_vocabulary_output):
    GPT4Tokenizer.train(temp_input_file, 257, temp_vocabulary_output)

    vocabulary_path = VOCABULARIES_DIR / temp_vocabulary_output
    assert vocabulary_path.exists()

    with open(vocabulary_path, "rb") as file:
        vocabulary = pickle.load(file)

    assert len(vocabulary["encode"]) == 1  # 257 - 256
    assert set(vocabulary["decode"].keys()) == {256}

    tokenizer = GPT4Tokenizer(temp_vocabulary_output)
    text = "a b a b a b a b"
    assert tokenizer.decode(tokenizer.encode(text)) == text


# ---------------------------------------------------------------------------
# get_token_pair_counts_for_chunks
# ---------------------------------------------------------------------------


def test_get_token_pair_counts_for_chunks_counts_within_each_chunk_separately():
    token_chunks = [[1, 2, 3], [4, 5]]

    # (3, 4) would only occur if the chunk boundary were ignored - it must not be counted.
    assert GPT4Tokenizer.get_token_pair_counts_for_chunks(token_chunks) == {
        (1, 2): 1,
        (2, 3): 1,
        (4, 5): 1,
    }


def test_get_token_pair_counts_for_chunks_sums_counts_across_chunks():
    token_chunks = [[1, 2], [1, 2], [1, 2]]

    assert GPT4Tokenizer.get_token_pair_counts_for_chunks(token_chunks) == {(1, 2): 3}


def test_get_token_pair_counts_for_chunks_ignores_chunks_with_fewer_than_two_tokens():
    token_chunks = [[1], [2, 3], []]

    assert GPT4Tokenizer.get_token_pair_counts_for_chunks(token_chunks) == {(2, 3): 1}


def test_get_token_pair_counts_for_chunks_empty_list_returns_empty_dict():
    assert GPT4Tokenizer.get_token_pair_counts_for_chunks([]) == {}


# ---------------------------------------------------------------------------
# merge_token_pairs_for_chunks
# ---------------------------------------------------------------------------


def test_merge_token_pairs_for_chunks_merges_within_each_chunk_independently():
    token_chunks = [[1, 2, 3], [1, 2]]

    assert GPT4Tokenizer.merge_token_pairs_for_chunks(token_chunks, (1, 2), 99) == [
        [99, 3],
        [99],
    ]


def test_merge_token_pairs_for_chunks_does_not_merge_across_chunk_boundaries():
    # (2, 3) only occurs across the boundary between the two chunks, so it must be left alone.
    token_chunks = [[1, 2], [3, 4]]

    assert GPT4Tokenizer.merge_token_pairs_for_chunks(token_chunks, (2, 3), 99) == [
        [1, 2],
        [3, 4],
    ]


def test_merge_token_pairs_for_chunks_empty_list_returns_empty_list():
    assert GPT4Tokenizer.merge_token_pairs_for_chunks([], (1, 2), 99) == []
