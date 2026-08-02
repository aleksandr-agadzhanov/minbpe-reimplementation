from __future__ import annotations

import pickle
import time
from pathlib import Path

import regex as re

from tokenizers.basic_tokenizer import BasicTokenizer


class GPT4Tokenizer(BasicTokenizer):
    SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def _encode_non_special(self, text: str) -> list[int]:
        """Encode text containing no special tokens into a list of token ids.

        The text is first split into chunks using `SPLIT_PATTERN`, and each
        chunk is encoded independently, so a merge is never applied across a
        chunk boundary - matching how the vocabulary was trained.

        Args:
            text: The text to encode. Must not contain special token text -
                use `encode` for text that might.

        Returns:
            A list of token ids representing the encoded text.
        """
        # Splitting before encoding is what stops a merge from being applied across a chunk boundary.
        text_chunks = re.findall(GPT4Tokenizer.SPLIT_PATTERN, text)

        tokens = []
        for text_chunk in text_chunks:
            # super()._encode_non_special() applies BasicTokenizer's merge loop to a single chunk at a time.
            tokens.extend(super()._encode_non_special(text_chunk))

        return tokens

    @staticmethod
    def train(
        input_file_name: str,
        vocabulary_size: int,
        vocabulary_file_name: str,
        special_tokens: dict[str, int] | None = None,
        verbose: bool = False,
    ):
        """Train a BPE vocabulary from a text file and save it to `vocabularies/{vocabulary_file_name}`.

        The input text is first split into chunks using `SPLIT_PATTERN`, and
        every merge is counted and applied within each chunk separately - a
        token pair is never merged if it straddles two chunks, keeping
        training consistent with how `encode` processes chunks.

        Args:
            input_file_name: Name of the input text file, relative to the `training_datasets/` directory.
            vocabulary_size: Desired size of the final vocabulary - 256 base
                token values, plus `len(special_tokens)`, plus however many
                merges are needed to reach this size.
            vocabulary_file_name: Name to save the resulting vocabulary under,
                relative to the `vocabularies/` directory.
            special_tokens: A mapping of special token text (e.g. "<|endoftext|>")
                to the token id the caller wants it saved under. Ids are
                caller-controlled and must each be unique, >= 256 (below that
                is reserved for raw byte values), and outside the range of ids
                the merges will occupy.
            verbose: If True, print each merge as it's added to the vocabulary.

        Raises:
            ValueError: If `vocabulary_size` leaves room for fewer than one
                merge, if `special_tokens` contains a duplicate or
                out-of-range id, or if the input text runs out of token pairs
                to merge before reaching `vocabulary_size`.
            FileNotFoundError: If no input file exists at the resolved path, or
                if the `vocabularies/` directory doesn't exist.
            FileExistsError: If a vocabulary file already exists at the resolved
                save path.
        """
        special_tokens = special_tokens or {}
        num_special_tokens = len(special_tokens)

        if (
            vocabulary_size
            <= BasicTokenizer.BASE_VOCABULARY_SIZE + num_special_tokens
        ):
            # 0-255 are reserved for raw token values and the rest for special
            # tokens, so at least one merge is required to reach vocabulary_size.
            raise ValueError(
                f"vocabulary_size must be greater than "
                f"{BasicTokenizer.BASE_VOCABULARY_SIZE} + len(special_tokens) "
                f"({BasicTokenizer.BASE_VOCABULARY_SIZE + num_special_tokens}), got {vocabulary_size}"
            )

        num_merges = (
            vocabulary_size
            - BasicTokenizer.BASE_VOCABULARY_SIZE
            - num_special_tokens
        )
        # The ids merges will use, so a caller-chosen special token id can be checked against it.
        merge_id_range = range(
            BasicTokenizer.BASE_VOCABULARY_SIZE,
            BasicTokenizer.BASE_VOCABULARY_SIZE + num_merges,
        )
        if len(set(special_tokens.values())) != num_special_tokens:
            raise ValueError(
                f"special_tokens ids must be unique, got: {special_tokens}"
            )
        for token, token_id in special_tokens.items():
            if token_id < BasicTokenizer.BASE_VOCABULARY_SIZE:
                raise ValueError(
                    f"special_tokens id for {token!r} must be >= "
                    f"{BasicTokenizer.BASE_VOCABULARY_SIZE}, got {token_id}"
                )
            if token_id in merge_id_range:
                raise ValueError(
                    f"special_tokens id for {token!r} ({token_id}) collides with a merge "
                    f"token id (merges will use ids {merge_id_range.start}-{merge_id_range.stop - 1})"
                )

        start_time = time.time()

        # Checked upfront so training never runs only to fail on the save step at the very end.
        vocabulary_path = (
            Path(__file__).resolve().parent.parent
            / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
            / vocabulary_file_name
        )
        if vocabulary_path.exists():
            raise FileExistsError(f"Vocabulary file already exists: {vocabulary_path}")

        input_path = (
            Path(__file__).resolve().parent.parent
            / BasicTokenizer.INPUTS_DIRECTORY_NAME
            / input_file_name
        )
        try:
            with open(
                input_path, "r", encoding=BasicTokenizer.TEXT_ENCODING
            ) as input_file:
                text = input_file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}") from None

        # Splitting before encoding is what stops a merge from ever spanning a chunk boundary.
        text_chunks = re.findall(GPT4Tokenizer.SPLIT_PATTERN, text)

        # Keep the pre-merge token chunks around so we can report the compression ratio later.
        original_token_chunks = [
            list(chunk.encode(BasicTokenizer.TEXT_ENCODING)) for chunk in text_chunks
        ]

        # Work on a copy so `original_tokens` still reflects the starting length.
        token_chunks = original_token_chunks.copy()
        encode_vocabulary = {}
        decode_vocabulary = {}
        new_token_id = BasicTokenizer.BASE_VOCABULARY_SIZE

        for _ in range(num_merges):
            # Counts are accumulated per chunk internally, so cross-chunk pairs are never counted.
            token_pair_counts = GPT4Tokenizer.get_token_pair_counts_for_chunks(
                token_chunks
            )
            try:
                token_pair = BasicTokenizer.get_most_frequent_token_pair(
                    token_pair_counts
                )
            except ValueError:
                # get_most_frequent_token_pair raises when token_pair_counts is empty,
                # i.e. fewer than 2 tokens remain to form a pair.
                raise ValueError(
                    f"Cannot reach vocabulary_size={vocabulary_size}: ran out of token "
                    f"pairs to merge after {new_token_id - BasicTokenizer.BASE_VOCABULARY_SIZE} merge(s)"
                ) from None
            # Merging each chunk independently keeps every chunk's boundaries intact for the next iteration.
            token_chunks = GPT4Tokenizer.merge_token_pairs_for_chunks(
                token_chunks, token_pair, new_token_id
            )

            encode_vocabulary[token_pair] = new_token_id
            decode_vocabulary[new_token_id] = []

            # A component >= BASE_VOCABULARY_SIZE is itself an earlier merge, already fully
            # expanded to raw bytes in decode_vocabulary - splice that in so every entry stays flat.
            for token_id in token_pair:
                if token_id >= BasicTokenizer.BASE_VOCABULARY_SIZE:
                    decode_vocabulary[new_token_id].extend(decode_vocabulary[token_id])
                else:
                    decode_vocabulary[new_token_id].append(token_id)

            if verbose:
                merge_hours, merge_remainder_seconds = divmod(
                    int(time.time() - start_time), 3600
                )
                merge_minutes, merge_seconds = divmod(merge_remainder_seconds, 60)
                print(
                    f"Added new token - {token_pair} - to the vocabulary with token ID - {new_token_id} "
                    f"({merge_hours}h {merge_minutes}m {merge_seconds}s elapsed)"
                )

            new_token_id = new_token_id + 1

        # Special tokens are never produced by merging byte pairs, so they only need a
        # decode_vocabulary entry - _expand_token() already handles any id >= 256 generically.
        for token, token_id in special_tokens.items():
            decode_vocabulary[token_id] = list(token.encode(BasicTokenizer.TEXT_ENCODING))

        # token_chunks is a list of chunks, so the total token count is the sum of each chunk's length.
        num_original_tokens = sum([len(chunk) for chunk in original_token_chunks])
        num_merged_tokens = sum([len(chunk) for chunk in token_chunks])

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Initial number of tokens - {num_original_tokens}")
        print(f"Final number of tokens   - {num_merged_tokens}")
        # How many raw tokens, on average, each remaining token now represents.
        print(f"Compression ratio        - {num_original_tokens / num_merged_tokens}x")

        # Saved together since encode() and decode() each need a differently shaped vocabulary.
        vocabulary = {
            BasicTokenizer.ENCODE_VOCABULARY_KEY: encode_vocabulary,
            BasicTokenizer.DECODE_VOCABULARY_KEY: decode_vocabulary,
            BasicTokenizer.SPECIAL_TOKENS_KEY: special_tokens,
        }
        try:
            with open(vocabulary_path, "wb") as output_file:
                pickle.dump(vocabulary, output_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabularies directory not found: {vocabulary_path.parent}"
            ) from None

        elapsed_hours, remainder_seconds = divmod(int(time.time() - start_time), 3600)
        elapsed_minutes, elapsed_seconds = divmod(remainder_seconds, 60)

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Saved the vocabulary to the path - {vocabulary_path}")
        print(f"Time elapsed              - {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")

    @staticmethod
    def get_token_pair_counts_for_chunks(
        token_chunks: list[list[int]],
    ) -> dict[tuple[int, int], int]:
        """Count occurrences of each consecutive pair of tokens within each chunk.

        Pairs are counted separately per chunk before being summed, so a pair
        that straddles two chunks (e.g. the last token of one chunk and the
        first token of the next) is never counted - this keeps merges from
        ever crossing a chunk boundary.

        Args:
            token_chunks: A list of token id chunks, each produced by
                splitting the original text with `SPLIT_PATTERN`.

        Returns:
            A dictionary mapping each adjacent (token, next_token) pair to the
            total number of times it occurs consecutively within any chunk.
        """
        token_pair_counts = {}

        for chunk in token_chunks:
            # Counting within a single chunk at a time is what prevents pairs from spanning chunks.
            chunk_token_pair_counts = BasicTokenizer.get_token_pair_counts(chunk)
            for token_pair, count in chunk_token_pair_counts.items():
                # Accumulate into the running total across all chunks seen so far.
                token_pair_counts[token_pair] = (
                    token_pair_counts.get(token_pair, 0) + count
                )

        return token_pair_counts

    @staticmethod
    def merge_token_pairs_for_chunks(
        token_chunks: list[list[int]], token_pair: tuple[int, int], merged_token_id: int
    ) -> list[list[int]]:
        """Replace every occurrence of `token_pair` with `merged_token_id`, within each chunk.

        Args:
            token_chunks: A list of token id chunks, each produced by
                splitting the original text with `SPLIT_PATTERN`.
            token_pair: The adjacent pair of token ids to replace wherever it occurs.
            merged_token_id: The token id to substitute for `token_pair`.

        Returns:
            A new list of chunks, each with every occurrence of `token_pair`
            collapsed into a single `merged_token_id`.
        """
        # Merging each chunk separately keeps a pair from ever being merged across a chunk boundary.
        return [
            BasicTokenizer.merge_token_pairs(chunk, token_pair, merged_token_id)
            for chunk in token_chunks
        ]
