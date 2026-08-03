import pickle
from pathlib import Path

import pytest

from tokenizers.basic_tokenizer import BasicTokenizer
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
# get_token_pair_counts_and_locations_for_chunks
# ---------------------------------------------------------------------------


def test_get_token_pair_counts_and_locations_for_chunks_counts_within_each_chunk_separately():
    token_chunks = [[1, 2, 3], [4, 5]]

    # (3, 4) would only occur if the chunk boundary were ignored - it must not be counted.
    token_pair_counts, _ = RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(
        token_chunks
    )

    assert token_pair_counts == {
        (1, 2): 1,
        (2, 3): 1,
        (4, 5): 1,
    }


def test_get_token_pair_counts_and_locations_for_chunks_sums_counts_across_chunks():
    token_chunks = [[1, 2], [1, 2], [1, 2]]

    token_pair_counts, _ = RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(
        token_chunks
    )

    assert token_pair_counts == {(1, 2): 3}


def test_get_token_pair_counts_and_locations_for_chunks_ignores_chunks_with_fewer_than_two_tokens():
    token_chunks = [[1], [2, 3], []]

    token_pair_counts, _ = RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(
        token_chunks
    )

    assert token_pair_counts == {(2, 3): 1}


def test_get_token_pair_counts_and_locations_for_chunks_empty_list_returns_empty_dict():
    token_pair_counts, _ = RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(
        []
    )

    assert token_pair_counts == {}


def test_merge_token_pairs_for_chunks_and_update_counts_matches_two_pass_behavior():
    original_token_chunks = [[1, 2, 3], [1, 2]]
    token_chunks = [chunk.copy() for chunk in original_token_chunks]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    expected_chunks = [
        BasicTokenizer._merge_token_pairs(chunk, (1, 2), 99)
        for chunk in original_token_chunks
    ]
    expected_counts = {}
    expected_indices = {}
    for chunk_index, chunk in enumerate(expected_chunks):
        chunk_pair_counts = BasicTokenizer._get_token_pair_counts(chunk)
        for token_pair, count in chunk_pair_counts.items():
            expected_counts[token_pair] = expected_counts.get(token_pair, 0) + count
            expected_indices.setdefault(token_pair, set()).add(chunk_index)

    assert token_chunks == expected_chunks
    assert token_pair_counts == expected_counts
    assert pair_chunk_indices == expected_indices


def test_merge_token_pairs_for_chunks_and_update_counts_removes_a_pair_left_with_no_occurrences():
    # (2, 3) only occurs in chunk 0 and disappears entirely once (1, 2) is merged there.
    token_chunks = [[1, 2, 3]]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == [[99, 3]]
    assert token_pair_counts == {(99, 3): 1}
    assert pair_chunk_indices == {(99, 3): {0}}
    assert (2, 3) not in token_pair_counts
    assert (2, 3) not in pair_chunk_indices


def test_merge_token_pairs_for_chunks_and_update_counts_handles_overlapping_pairs_in_one_chunk():
    # (1, 2) occurs twice back-to-back, so merging must also produce the new (99, 99) pair.
    token_chunks = [[1, 2, 1, 2, 3]]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == [[99, 99, 3]]
    assert token_pair_counts == {(99, 99): 1, (99, 3): 1}
    assert pair_chunk_indices == {(99, 99): {0}, (99, 3): {0}}


def test_merge_token_pairs_for_chunks_and_update_counts_combines_new_pair_count_across_chunks():
    # Chunk 1 already contains (99, 3); merging chunk 0's (1, 2) into 99 must add to,
    # not overwrite, that existing global count and chunk-index entry.
    token_chunks = [[1, 2, 3], [99, 3]]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == [[99, 3], [99, 3]]
    assert token_pair_counts == {(99, 3): 2}
    assert pair_chunk_indices == {(99, 3): {0, 1}}


def test_merge_token_pairs_for_chunks_and_update_counts_skips_chunks_without_the_pair():
    # Only chunk 0 contains (1, 2) - chunk 1 must be left completely untouched.
    token_chunks = [[1, 2], [3, 4]]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == [[99], [3, 4]]
    assert token_pair_counts == {(3, 4): 1}
    assert pair_chunk_indices == {(3, 4): {1}}


def test_merge_token_pairs_for_chunks_and_update_counts_respects_chunk_boundaries():
    # (2, 3) only occurs across the boundary between the two chunks, so it was
    # never indexed/counted and merging it must be a no-op.
    token_chunks = [[1, 2], [3, 4]]
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (2, 3), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == [[1, 2], [3, 4]]
    assert token_pair_counts == {(1, 2): 1, (3, 4): 1}


def test_merge_token_pairs_for_chunks_and_update_counts_empty_list_is_a_no_op():
    token_chunks = []
    token_pair_counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
        token_chunks, (1, 2), 99, token_pair_counts, pair_chunk_indices
    )

    assert token_chunks == []
    assert token_pair_counts == {}


def test_get_token_pair_counts_and_locations_for_chunks_records_chunk_indices_per_pair():
    token_chunks = [[1, 2, 3], [4, 5], [1, 2]]

    _, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
    )

    assert pair_chunk_indices == {
        (1, 2): {0, 2},
        (2, 3): {0},
        (4, 5): {1},
    }


def test_get_token_pair_counts_and_locations_for_chunks_empty_list_returns_empty_results():
    counts, pair_chunk_indices = (
        RegexTokenizer._get_token_pair_counts_and_locations_for_chunks([])
    )

    assert counts == {}
    assert pair_chunk_indices == {}
