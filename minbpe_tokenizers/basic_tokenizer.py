from __future__ import annotations

import pickle
import re
import time
from pathlib import Path


class BasicTokenizer:
    """A byte-level BPE tokenizer trained by iteratively merging the most frequent adjacent token pair.

    Merges are learned directly over the raw byte stream of the input text,
    with no pre-tokenization step. A trained vocabulary is loaded from, and
    saved to, the `vocabularies/` directory via `__init__` and `train`.
    """

    # Ids 0-255 are reserved for raw byte values; learned merges start at this id.
    BASE_VOCABULARY_SIZE = 256
    VOCABULARIES_DIRECTORY_NAME = "vocabularies"
    INPUTS_DIRECTORY_NAME = "training_datasets"
    TEXT_ENCODING = "utf-8"
    ENCODE_VOCABULARY_KEY = "encode"
    DECODE_VOCABULARY_KEY = "decode"
    SPECIAL_TOKENS_KEY = "special_tokens"
    PROGRESS_SEPARATOR = "--------------------------------------------------"

    def __init__(self, vocabulary_file_name: str):
        """Load a previously trained vocabulary from `vocabularies/{vocabulary_file_name}`.

        Args:
            vocabulary_file_name: Name of the pickled vocabulary file, relative
                to the `vocabularies/` directory.

        Raises:
            FileNotFoundError: If no vocabulary file exists at the resolved path.
        """
        vocabulary_path = (
            Path(__file__).resolve().parent.parent
            / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
            / vocabulary_file_name
        )
        try:
            with open(vocabulary_path, "rb") as file:
                vocabulary = pickle.load(file)
                self.encode_vocabulary = vocabulary[
                    BasicTokenizer.ENCODE_VOCABULARY_KEY
                ]
                self.decode_vocabulary = vocabulary[
                    BasicTokenizer.DECODE_VOCABULARY_KEY
                ]
                self.special_tokens = vocabulary[BasicTokenizer.SPECIAL_TOKENS_KEY]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocabulary_path}"
            ) from None

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids using this tokenizer's trained vocabulary.

        Special token text (e.g. "<|endoftext|>") is matched as a whole
        segment and mapped directly to its reserved id; everything else is
        encoded with `_encode_non_special`.

        Args:
            text: The text to encode.

        Returns:
            A list of token ids representing the encoded text.
        """
        if not self.special_tokens:
            return self.encode_non_special(text)

        # Longest tokens first, so a shorter special token can't shadow a longer one it's a prefix of.
        special_token_pattern = "|".join(
            re.escape(token)
            for token in sorted(self.special_tokens, key=len, reverse=True)
        )
        # The capturing group keeps the special token matches in the split result.
        segments = re.split(f"({special_token_pattern})", text)

        tokens = []
        for segment in segments:
            if segment in self.special_tokens:
                tokens.append(self.special_tokens[segment])
            elif segment:
                tokens.extend(self.encode_non_special(segment))

        return tokens

    def encode_non_special(self, text: str) -> list[int]:
        """Encode text containing no special tokens into a list of token ids.

        Repeatedly applies the highest-priority (earliest learned) merge found
        among the currently adjacent token pairs, until no known pair remains.

        Args:
            text: The text to encode. Must not contain special token text -
                use `encode` for text that might.

        Returns:
            A list of token ids representing the encoded text.
        """
        tokens = list(text.encode(BasicTokenizer.TEXT_ENCODING))

        while True:
            # Fewer than 2 tokens means there's no pair left to possibly merge.
            if len(tokens) < 2:
                break

            # Track the best (lowest-id) candidate found so far in a single pass.
            token_pair_with_lowest_id = None
            lowest_token_id = float("inf")

            for i in range(len(tokens) - 1):
                token_pair = (tokens[i], tokens[i + 1])
                # Pairs with no learned merge default to infinity so they can
                # never outrank an actual entry in encode_vocabulary.
                token_pair_id = self.encode_vocabulary.get(token_pair, float("inf"))
                # Merges must be applied in the order they were learned - lower ids
                # were learned earlier, so keep whichever pair has the lowest id so far.
                if token_pair_id < lowest_token_id:
                    token_pair_with_lowest_id = token_pair
                    lowest_token_id = token_pair_id

            if lowest_token_id == float("inf"):
                # None of the pairs present are known merges - nothing left to do.
                break

            # Replace every occurrence of this pair before searching for the next merge.
            tokens = BasicTokenizer._merge_token_pairs(
                tokens, token_pair_with_lowest_id, lowest_token_id
            )

        return tokens

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token ids produced by `encode` back into text.

        Args:
            tokens: A sequence of token ids, using this tokenizer's vocabulary.

        Returns:
            The decoded text, with any invalid UTF-8 token sequences replaced.

        Raises:
            KeyError: If `tokens` contains an id not present in this tokenizer's vocabulary.
        """
        decoded_tokens = []
        for token_id in tokens:
            decoded_tokens.extend(self._expand_token(token_id))
        # Invalid UTF-8 (e.g. from a partial token sequence) is replaced with U+FFFD instead of raising.
        text = bytes(decoded_tokens).decode(
            BasicTokenizer.TEXT_ENCODING, errors="replace"
        )
        return text

    def decode_as_list(self, tokens: list[int]) -> list[str]:
        """Decode each token id individually into its own textual representation.

        Unlike `decode`, which concatenates all raw bytes before decoding,
        each token is decoded on its own - useful for inspecting what
        individual tokens in a sequence represent.

        Args:
            tokens: A sequence of token ids, using this tokenizer's vocabulary.

        Returns:
            A list of strings, one per input token id, with any invalid UTF-8
            byte sequences replaced.

        Raises:
            KeyError: If `tokens` contains an id not present in this tokenizer's vocabulary.
        """
        return [
            bytes(self._expand_token(token_id)).decode(
                BasicTokenizer.TEXT_ENCODING, errors="replace"
            )
            for token_id in tokens
        ]

    def _expand_token(self, token_id: int) -> list[int]:
        """Expand a single token id into the raw byte values it represents.

        Args:
            token_id: A single token id, using this tokenizer's vocabulary.

        Returns:
            The list of raw byte values (0-255) that `token_id` expands to.

        Raises:
            KeyError: If `token_id` is not present in this tokenizer's vocabulary.
        """
        if token_id < BasicTokenizer.BASE_VOCABULARY_SIZE:
            # Raw token value - nothing to expand.
            return [token_id]
        try:
            # Maps a merged token id to the full raw-byte sequence it expands to.
            return self.decode_vocabulary[token_id]
        except KeyError:
            raise KeyError(f"Unknown token id: {token_id}") from None

    @staticmethod
    def train(
        input_file_name: str,
        vocabulary_size: int,
        vocabulary_file_name: str,
        special_tokens: dict[str, int] | None = None,
        verbose: bool = False,
    ):
        """Train a BPE vocabulary from a text file and save it to `vocabularies/{vocabulary_file_name}`.

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

        # Keep the pre-merge tokens around so we can report the compression ratio later.
        original_tokens = list(text.encode(BasicTokenizer.TEXT_ENCODING))

        # Work on a copy so `original_tokens` still reflects the starting length.
        tokens = original_tokens.copy()
        # Count once on the full dataset, then update after each merge.
        token_pair_counts = BasicTokenizer._get_token_pair_counts(tokens)
        encode_vocabulary = {}
        decode_vocabulary = {}
        new_token_id = BasicTokenizer.BASE_VOCABULARY_SIZE

        for _ in range(num_merges):
            try:
                token_pair = BasicTokenizer._get_most_frequent_token_pair(
                    token_pair_counts
                )
            except ValueError:
                # Raised when token_pair_counts is empty, i.e. fewer than 2 tokens remain.
                raise ValueError(
                    f"Cannot reach vocabulary_size={vocabulary_size}: ran out of token "
                    f"pairs to merge after {new_token_id - BasicTokenizer.BASE_VOCABULARY_SIZE} merge(s)"
                ) from None
            tokens, token_pair_counts = (
                BasicTokenizer._merge_token_pairs_and_update_counts(
                    tokens, token_pair, new_token_id
                )
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

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Initial number of tokens - {len(original_tokens)}")
        print(f"Final number of tokens   - {len(tokens)}")
        # How many raw tokens, on average, each remaining token now represents.
        print(f"Compression ratio        - {len(original_tokens) / len(tokens)}x")

        BasicTokenizer._save_vocabulary(
            vocabulary_path, encode_vocabulary, decode_vocabulary, special_tokens
        )

        BasicTokenizer._print_run_summary(start_time, vocabulary_path)

    @staticmethod
    def _validate_special_tokens_and_get_num_merges(
        vocabulary_size: int, special_tokens: dict[str, int]
    ) -> int:
        """Validate `special_tokens` and compute how many merges `train` should perform.

        Args:
            vocabulary_size: Desired size of the final vocabulary.
            special_tokens: A mapping of special token text to caller-chosen ids.

        Returns:
            The number of merges needed to reach `vocabulary_size`.

        Raises:
            ValueError: If `vocabulary_size` leaves room for fewer than one
                merge, or if `special_tokens` contains a duplicate or
                out-of-range id.
        """
        num_special_tokens = len(special_tokens)

        if vocabulary_size <= BasicTokenizer.BASE_VOCABULARY_SIZE + num_special_tokens:
            # At least one merge is required since ids 0-255 are reserved and special tokens claim the rest.
            raise ValueError(
                f"vocabulary_size must be greater than "
                f"{BasicTokenizer.BASE_VOCABULARY_SIZE} + len(special_tokens) "
                f"({BasicTokenizer.BASE_VOCABULARY_SIZE + num_special_tokens}), got {vocabulary_size}"
            )

        num_merges = (
            vocabulary_size - BasicTokenizer.BASE_VOCABULARY_SIZE - num_special_tokens
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

        return num_merges

    @staticmethod
    def _prepare_vocabulary_save_path(vocabulary_file_name: str) -> Path:
        """Resolve where a trained vocabulary should be saved, before training runs.

        Args:
            vocabulary_file_name: Name to save the vocabulary under, relative
                to the `vocabularies/` directory.

        Returns:
            The resolved path to save the vocabulary to.

        Raises:
            FileExistsError: If a vocabulary file already exists at the resolved path.
        """
        vocabulary_path = (
            Path(__file__).resolve().parent.parent
            / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
            / vocabulary_file_name
        )
        if vocabulary_path.exists():
            raise FileExistsError(f"Vocabulary file already exists: {vocabulary_path}")
        return vocabulary_path

    @staticmethod
    def _read_training_text(input_file_name: str) -> str:
        """Read the input text to train a vocabulary from.

        Args:
            input_file_name: Name of the input text file, relative to the
                `training_datasets/` directory.

        Returns:
            The full contents of the input file.

        Raises:
            FileNotFoundError: If no input file exists at the resolved path.
        """
        input_path = (
            Path(__file__).resolve().parent.parent
            / BasicTokenizer.INPUTS_DIRECTORY_NAME
            / input_file_name
        )
        try:
            with open(
                input_path, "r", encoding=BasicTokenizer.TEXT_ENCODING
            ) as input_file:
                return input_file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}") from None

    @staticmethod
    def _record_merge(
        encode_vocabulary: dict[tuple[int, int], int],
        decode_vocabulary: dict[int, list[int]],
        token_pair: tuple[int, int],
        new_token_id: int,
    ) -> None:
        """Record a newly learned merge in both vocabularies, in place.

        Args:
            encode_vocabulary: Maps a byte-pair to the merge id that replaces it.
            decode_vocabulary: Maps a token id to the raw bytes it expands to.
            token_pair: The adjacent pair of token ids being merged.
            new_token_id: The token id assigned to `token_pair`.
        """
        encode_vocabulary[token_pair] = new_token_id
        decode_vocabulary[new_token_id] = []

        # A component >= BASE_VOCABULARY_SIZE is an earlier merge - splice its expansion in to keep entries flat.
        for token_id in token_pair:
            if token_id >= BasicTokenizer.BASE_VOCABULARY_SIZE:
                decode_vocabulary[new_token_id].extend(decode_vocabulary[token_id])
            else:
                decode_vocabulary[new_token_id].append(token_id)

    @staticmethod
    def _print_merge_progress(
        token_pair: tuple[int, int], new_token_id: int, start_time: float
    ) -> None:
        """Print a single line reporting a merge that was just added to the vocabulary.

        Args:
            token_pair: The adjacent pair of token ids that was merged.
            new_token_id: The token id assigned to `token_pair`.
            start_time: The `time.time()` value training started at.
        """
        merge_hours, merge_remainder_seconds = divmod(
            int(time.time() - start_time), 3600
        )
        merge_minutes, merge_seconds = divmod(merge_remainder_seconds, 60)
        print(
            f"Added new token - {token_pair} - to the vocabulary with token ID - {new_token_id} "
            f"({merge_hours}h {merge_minutes}m {merge_seconds}s elapsed)"
        )

    @staticmethod
    def _add_special_tokens_to_decode_vocabulary(
        decode_vocabulary: dict[int, list[int]], special_tokens: dict[str, int]
    ) -> None:
        """Add each special token's raw-byte expansion to `decode_vocabulary`, in place.

        Special tokens are never produced by merging byte pairs, so they only
        need a decode_vocabulary entry - `_expand_token` already handles any
        id >= 256 generically.

        Args:
            decode_vocabulary: Maps a token id to the raw bytes it expands to.
            special_tokens: A mapping of special token text to caller-chosen ids.
        """
        for token, token_id in special_tokens.items():
            decode_vocabulary[token_id] = list(
                token.encode(BasicTokenizer.TEXT_ENCODING)
            )

    @staticmethod
    def _save_vocabulary(
        vocabulary_path: Path,
        encode_vocabulary: dict[tuple[int, int], int],
        decode_vocabulary: dict[int, list[int]],
        special_tokens: dict[str, int],
    ) -> None:
        """Pickle the trained vocabulary to `vocabulary_path`.

        Args:
            vocabulary_path: The resolved path to save the vocabulary to.
            encode_vocabulary: Maps a byte-pair to the merge id that replaces it.
            decode_vocabulary: Maps a token id to the raw bytes it expands to.
            special_tokens: A mapping of special token text to caller-chosen ids.

        Raises:
            FileNotFoundError: If the `vocabularies/` directory doesn't exist.
        """
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

    @staticmethod
    def _print_run_summary(start_time: float, vocabulary_path: Path) -> None:
        """Print where the vocabulary was saved and how long training took.

        Args:
            start_time: The `time.time()` value training started at.
            vocabulary_path: The resolved path the vocabulary was saved to.
        """
        elapsed_hours, remainder_seconds = divmod(int(time.time() - start_time), 3600)
        elapsed_minutes, elapsed_seconds = divmod(remainder_seconds, 60)

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Saved the vocabulary to the path - {vocabulary_path}")
        print(
            f"Time elapsed              - {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s"
        )

    @staticmethod
    def _get_token_pair_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
        """Count occurrences of each consecutive pair of tokens.

        Args:
            tokens: A sequence of integer token ids.

        Returns:
            A dictionary mapping each adjacent (token, next_token) pair to the
            number of times it occurs consecutively in `tokens`.
        """
        token_pair_counts = {}

        # Stop one early so `i + 1` never goes out of bounds.
        for i in range(len(tokens) - 1):
            token_pair = (tokens[i], tokens[i + 1])
            # Increment the count for this pair, defaulting to 0 if unseen.
            token_pair_counts[token_pair] = token_pair_counts.get(token_pair, 0) + 1

        return token_pair_counts

    @staticmethod
    def _get_most_frequent_token_pair(
        token_pair_counts: dict[tuple[int, int], int],
    ) -> tuple[int, int]:
        """Return the most frequent token pair.

        Args:
            token_pair_counts: A dictionary mapping token pairs to their counts.

        Returns:
            The (token, next_token) pair with the highest count.

        Raises:
            ValueError: If `token_pair_counts` is empty.
        """
        if not token_pair_counts:
            raise ValueError("token_pair_counts must not be empty")
        # Find the pair with the maximum count, then extract the pair tuple (skip the count value).
        return max(token_pair_counts.items(), key=lambda item: item[1])[0]

    @staticmethod
    def _merge_token_pairs(
        tokens: list[int], token_pair: tuple[int, int], merged_token_id: int
    ) -> list[int]:
        """Replace every occurrence of `token_pair` with `merged_token_id`.

        Args:
            tokens: A sequence of integer token ids.
            token_pair: The adjacent pair of token ids to replace wherever it occurs.
            merged_token_id: The token id to substitute for `token_pair`.

        Returns:
            A new list of tokens with each occurrence of `token_pair` collapsed
            into a single `merged_token_id`.
        """
        output_tokens = []

        # Replace every occurrence of the target pair with `merged_token_id`, copying everything else through.
        i = 0
        while i < len(tokens):
            # Guards against reading `tokens[i + 1]` out of bounds when `i` is the last index.
            if (
                i < len(tokens) - 1
                and tokens[i] == token_pair[0]
                and tokens[i + 1] == token_pair[1]
            ):
                # Found the target pair - emit the merged token and skip both.
                output_tokens.append(merged_token_id)
                i = i + 2
            else:
                # No match at this position - keep the token and advance by one.
                output_tokens.append(tokens[i])
                i = i + 1

        return output_tokens

    @staticmethod
    def _merge_token_pairs_and_update_counts(
        tokens: list[int], token_pair: tuple[int, int], merged_token_id: int
    ) -> tuple[list[int], dict[tuple[int, int], int]]:
        """Merge token_pair and rebuild pair counts in the same pass.

        This avoids a separate full-sequence recount after every merge.

        Args:
            tokens: A sequence of integer token ids.
            token_pair: The adjacent pair of token ids to replace wherever it occurs.
            merged_token_id: The token id to substitute for `token_pair`.

        Returns:
            A tuple containing the merged token sequence and the updated
            consecutive-pair counts for that merged sequence.
        """
        output_tokens = []
        updated_token_pair_counts = {}

        i = 0
        previous_output_token = None
        while i < len(tokens):
            # Emit exactly one output token per step: either a merged token or the current raw token.
            if (
                i < len(tokens) - 1
                and tokens[i] == token_pair[0]
                and tokens[i + 1] == token_pair[1]
            ):
                output_token = merged_token_id
                i = i + 2
            else:
                output_token = tokens[i]
                i = i + 1

            output_tokens.append(output_token)
            # Count adjacent pairs directly on the merged output stream to avoid a second full pass.
            if previous_output_token is not None:
                output_pair = (previous_output_token, output_token)
                updated_token_pair_counts[output_pair] = (
                    updated_token_pair_counts.get(output_pair, 0) + 1
                )
            previous_output_token = output_token

        return output_tokens, updated_token_pair_counts
