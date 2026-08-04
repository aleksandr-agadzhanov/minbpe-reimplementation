from __future__ import annotations

import time

import regex as re

from tokenizers.basic_tokenizer import BasicTokenizer


class RegexTokenizer(BasicTokenizer):
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
        text_chunks = re.findall(RegexTokenizer.SPLIT_PATTERN, text)

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
        num_merges = BasicTokenizer._validate_special_tokens_and_get_num_merges(
            vocabulary_size, special_tokens
        )

        start_time = time.time()

        # Checked upfront so training never runs only to fail on the save step at the very end.
        vocabulary_path = BasicTokenizer._prepare_vocabulary_save_path(
            vocabulary_file_name
        )

        text = BasicTokenizer._read_training_text(input_file_name)

        # Splitting before encoding is what stops a merge from ever spanning a chunk boundary.
        text_chunks = re.findall(RegexTokenizer.SPLIT_PATTERN, text)

        # Keep the pre-merge token chunks around so we can report the compression ratio later.
        original_token_chunks = [
            list(chunk.encode(BasicTokenizer.TEXT_ENCODING)) for chunk in text_chunks
        ]

        # Work on a copy so `original_token_chunks` still reflects the starting length.
        token_chunks = original_token_chunks.copy()
        # Count once across all chunks, and track which chunks contain each pair so
        # later merges only rescan the chunks that could actually contain them.
        token_pair_counts, pair_chunk_indices = (
            RegexTokenizer._get_token_pair_counts_and_locations_for_chunks(token_chunks)
        )
        encode_vocabulary = {}
        decode_vocabulary = {}
        new_token_id = BasicTokenizer.BASE_VOCABULARY_SIZE

        for _ in range(num_merges):
            try:
                token_pair = BasicTokenizer._get_most_frequent_token_pair(
                    token_pair_counts
                )
            except ValueError:
                # get_most_frequent_token_pair raises when token_pair_counts is empty,
                # i.e. fewer than 2 tokens remain to form a pair.
                raise ValueError(
                    f"Cannot reach vocabulary_size={vocabulary_size}: ran out of token "
                    f"pairs to merge after {new_token_id - BasicTokenizer.BASE_VOCABULARY_SIZE} merge(s)"
                ) from None
            # Only chunks known to contain token_pair are rescanned - every other
            # chunk is left untouched, avoiding a full-corpus rescan per merge.
            RegexTokenizer._merge_token_pairs_for_chunks_and_update_counts(
                token_chunks,
                token_pair,
                new_token_id,
                token_pair_counts,
                pair_chunk_indices,
            )

            BasicTokenizer._record_merge(
                encode_vocabulary, decode_vocabulary, token_pair, new_token_id
            )

            if verbose:
                BasicTokenizer._print_merge_progress(
                    token_pair, new_token_id, start_time
                )

            new_token_id = new_token_id + 1

        BasicTokenizer._add_special_tokens_to_decode_vocabulary(
            decode_vocabulary, special_tokens
        )

        # token_chunks is a list of chunks, so the total token count is the sum of each chunk's length.
        num_original_tokens = sum([len(chunk) for chunk in original_token_chunks])
        num_merged_tokens = sum([len(chunk) for chunk in token_chunks])

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Initial number of tokens - {num_original_tokens}")
        print(f"Final number of tokens   - {num_merged_tokens}")
        # How many raw tokens, on average, each remaining token now represents.
        print(f"Compression ratio        - {num_original_tokens / num_merged_tokens}x")

        BasicTokenizer._save_vocabulary(
            vocabulary_path, encode_vocabulary, decode_vocabulary, special_tokens
        )

        BasicTokenizer._print_run_summary(start_time, vocabulary_path)

    @staticmethod
    def _get_token_pair_counts_and_locations_for_chunks(
        token_chunks: list[list[int]],
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], set[int]]]:
        """Count consecutive token pairs per chunk and record which chunks contain each pair.

        The chunk-index lookup lets a later merge skip every chunk that can't
        contain the pair being merged, instead of rescanning the whole corpus.

        Args:
            token_chunks: A list of token id chunks, each produced by
                splitting the original text with `SPLIT_PATTERN`.

        Returns:
            A tuple of (token_pair_counts, pair_chunk_indices): the summed
            per-chunk pair counts, and a mapping from each pair to the set of
            chunk indices it occurs in.
        """
        token_pair_counts = {}
        pair_chunk_indices = {}

        for chunk_index, chunk in enumerate(token_chunks):
            # Counting within a single chunk at a time is what prevents pairs from spanning chunks.
            chunk_token_pair_counts = BasicTokenizer._get_token_pair_counts(chunk)
            for token_pair, count in chunk_token_pair_counts.items():
                token_pair_counts[token_pair] = (
                    token_pair_counts.get(token_pair, 0) + count
                )
                pair_chunk_indices.setdefault(token_pair, set()).add(chunk_index)

        return token_pair_counts, pair_chunk_indices

    @staticmethod
    def _merge_token_pairs_for_chunks_and_update_counts(
        token_chunks: list[list[int]],
        token_pair: tuple[int, int],
        merged_token_id: int,
        token_pair_counts: dict[tuple[int, int], int],
        pair_chunk_indices: dict[tuple[int, int], set[int]],
    ) -> None:
        """Merge `token_pair`, but only in the chunks that actually contain it.

        `pair_chunk_indices` guarantees every chunk containing `token_pair` is
        listed under it, so every other chunk is guaranteed not to contain it
        and can be skipped entirely - this avoids rescanning the full corpus
        on every merge.

        Args:
            token_chunks: A list of token id chunks; entries are replaced
                in place for every chunk that contained `token_pair`.
            token_pair: The adjacent pair of token ids to replace wherever it occurs.
            merged_token_id: The token id to substitute for `token_pair`.
            token_pair_counts: Global pair counts, updated in place to reflect the merge.
            pair_chunk_indices: Pair -> chunk-index lookup, updated in place to reflect the merge.
        """
        # token_pair can't reappear after merging, so drop its entry instead of updating it.
        affected_chunk_indices = pair_chunk_indices.pop(token_pair, set())

        for chunk_index in affected_chunk_indices:
            # Snapshot this chunk before mutation so we can diff its old contribution.
            old_chunk = token_chunks[chunk_index]
            old_chunk_pair_counts = BasicTokenizer._get_token_pair_counts(old_chunk)

            # Rebuild this chunk once: merge target pair and get its new local pair counts.
            new_chunk, new_chunk_pair_counts = (
                BasicTokenizer._merge_token_pairs_and_update_counts(
                    old_chunk, token_pair, merged_token_id
                )
            )
            token_chunks[chunk_index] = new_chunk

            # Both dicts sum/union contributions across chunks, so update by delta rather than overwrite.
            changed_pairs = old_chunk_pair_counts.keys() | new_chunk_pair_counts.keys()
            for pair in changed_pairs:
                count_delta = new_chunk_pair_counts.get(
                    pair, 0
                ) - old_chunk_pair_counts.get(pair, 0)
                if count_delta == 0:
                    continue

                updated_count = token_pair_counts.get(pair, 0) + count_delta
                if updated_count > 0:
                    token_pair_counts[pair] = updated_count
                else:
                    # Remove dead entries to keep _get_most_frequent_token_pair input compact.
                    token_pair_counts.pop(pair, None)

                if pair in new_chunk_pair_counts:
                    # This chunk now contains pair (possibly for the first time) - index it.
                    pair_chunk_indices.setdefault(pair, set()).add(chunk_index)
                else:
                    # This chunk no longer contains pair - drop it from the index.
                    chunk_indices = pair_chunk_indices.get(pair)
                    if chunk_indices is not None:
                        chunk_indices.discard(chunk_index)
                        if not chunk_indices:
                            # Drop empty index sets so pair_chunk_indices only contains live pairs.
                            del pair_chunk_indices[pair]
