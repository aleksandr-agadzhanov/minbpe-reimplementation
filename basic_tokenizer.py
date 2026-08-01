import pickle
from pathlib import Path


class BasicTokenizer:
    # Ids 0-255 are reserved for raw byte values; learned merges start at this id.
    BASE_VOCABULARY_SIZE = 256
    VOCABULARIES_DIRECTORY_NAME = "vocabularies"
    INPUTS_DIRECTORY_NAME = "training_datasets"
    TEXT_ENCODING = "utf-8"
    ENCODE_VOCABULARY_KEY = "encode"
    DECODE_VOCABULARY_KEY = "decode"
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
            Path(__file__).resolve().parent
            / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
            / vocabulary_file_name
        )
        try:
            with open(vocabulary_path, "rb") as file:
                vocabulary = pickle.load(file)
                self.encode_vocabulary = vocabulary[BasicTokenizer.ENCODE_VOCABULARY_KEY]
                self.decode_vocabulary = vocabulary[BasicTokenizer.DECODE_VOCABULARY_KEY]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocabulary_path}"
            ) from None

    def encode(self, text: str) -> list[int]:
        """Encode text into a list of token ids using this tokenizer's trained vocabulary.

        Repeatedly applies the highest-priority (earliest learned) merge found
        among the currently adjacent token pairs, until no known pair remains.

        Args:
            text: The text to encode.

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
            tokens = BasicTokenizer.merge_token_pairs(
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
            if token_id < BasicTokenizer.BASE_VOCABULARY_SIZE:
                # Raw token value - nothing to expand.
                decoded_tokens.append(token_id)
            else:
                try:
                    # Maps a merged token id to the full raw-byte sequence it expands to.
                    decoded_tokens.extend(self.decode_vocabulary[token_id])
                except KeyError:
                    raise KeyError(f"Unknown token id: {token_id}") from None
        # A subset of tokens (e.g. from a partial/invalid sequence) may not align to
        # valid UTF-8 - substitute the U+FFFD replacement character instead of raising.
        text = bytes(decoded_tokens).decode(BasicTokenizer.TEXT_ENCODING, errors="replace")
        return text

    @staticmethod
    def train(
        input_file_name: str,
        vocabulary_size: int,
        vocabulary_file_name: str,
        verbose: bool = False,
    ):
        """Train a BPE vocabulary from a text file and save it to `vocabularies/{vocabulary_file_name}`.

        Args:
            input_file_name: Name of the input text file, relative to the `inputs/` directory.
            vocabulary_size: Desired size of the final vocabulary (256 base
                token values plus however many merges are needed to reach this size).
            vocabulary_file_name: Name to save the resulting vocabulary under,
                relative to the `vocabularies/` directory.
            verbose: If True, print each merge as it's added to the vocabulary.

        Raises:
            ValueError: If `vocabulary_size` is not greater than 256, or if the
                input text runs out of token pairs to merge before reaching it.
            FileNotFoundError: If no input file exists at the resolved path, or
                if the `vocabularies/` directory doesn't exist.
            FileExistsError: If a vocabulary file already exists at the resolved
                save path.
        """
        if vocabulary_size <= BasicTokenizer.BASE_VOCABULARY_SIZE:
            # 0-255 are already reserved for raw token values, so at least one merge is required.
            raise ValueError(
                f"vocabulary_size must be greater than {BasicTokenizer.BASE_VOCABULARY_SIZE}, got {vocabulary_size}"
            )

        # Checked upfront so training never runs only to fail on the save step at the very end.
        vocabulary_path = (
            Path(__file__).resolve().parent
            / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
            / vocabulary_file_name
        )
        if vocabulary_path.exists():
            raise FileExistsError(f"Vocabulary file already exists: {vocabulary_path}")

        input_path = (
            Path(__file__).resolve().parent
            / BasicTokenizer.INPUTS_DIRECTORY_NAME
            / input_file_name
        )
        try:
            with open(input_path, "r", encoding=BasicTokenizer.TEXT_ENCODING) as input_file:
                text = input_file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}") from None

        # Keep the pre-merge tokens around so we can report the compression ratio later.
        original_tokens = list(text.encode(BasicTokenizer.TEXT_ENCODING))

        # Work on a copy so `original_tokens` still reflects the starting length.
        tokens = list(original_tokens)
        encode_vocabulary = {}
        decode_vocabulary = {}
        new_token_id = BasicTokenizer.BASE_VOCABULARY_SIZE
        num_merges = vocabulary_size - new_token_id

        for _ in range(num_merges):
            token_pair_counts = BasicTokenizer.get_token_pair_counts(tokens)
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
            tokens = BasicTokenizer.merge_token_pairs(tokens, token_pair, new_token_id)

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
                print(
                    f"Added new token - {token_pair} - to the vocabulary with token ID - {new_token_id}"
                )

            new_token_id = new_token_id + 1

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Initial number of tokens - {len(original_tokens)}")
        print(f"Final number of tokens   - {len(tokens)}")
        # How many raw tokens, on average, each remaining token now represents.
        print(f"Compression ratio        - {len(original_tokens) / len(tokens)}x")

        # Saved together since encode() and decode() each need a differently shaped vocabulary.
        vocabulary = {
            BasicTokenizer.ENCODE_VOCABULARY_KEY: encode_vocabulary,
            BasicTokenizer.DECODE_VOCABULARY_KEY: decode_vocabulary,
        }
        try:
            with open(vocabulary_path, "wb") as output_file:
                pickle.dump(vocabulary, output_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabularies directory not found: {vocabulary_path.parent}"
            ) from None

        print(BasicTokenizer.PROGRESS_SEPARATOR)
        print(f"Saved the vocabulary to the path - {vocabulary_path}")

    @staticmethod
    def merge_token_pairs(
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

        # Walk the tokens, replacing every occurrence of the target pair with
        # `merged_token_id` and copying everything else through unchanged.
        i = 0
        while i < len(tokens):
            # `i < len(tokens) - 1` guards against reading `tokens[i + 1]` when
            # `i` is the last index, which would otherwise raise an IndexError
            # if that final token happens to equal `token_pair[0]`.
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
    def get_token_pair_counts(tokens: list[int]) -> dict[tuple[int, int], int]:
        """Count occurrences of each consecutive pair of tokens.

        Args:
            tokens: A sequence of integer token ids.

        Returns:
            A dictionary mapping each adjacent (token, next_token) pair to the
            number of times it occurs consecutively in `tokens`.
        """
        token_pair_counts = {}

        # Slide a window of size 2 over the tokens, stopping one early so
        # `i + 1` never goes out of bounds.
        for i in range(len(tokens) - 1):
            token_pair = (tokens[i], tokens[i + 1])
            # Increment the count for this pair, defaulting to 0 if unseen.
            token_pair_counts[token_pair] = token_pair_counts.get(token_pair, 0) + 1

        return token_pair_counts

    @staticmethod
    def get_most_frequent_token_pair(
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
