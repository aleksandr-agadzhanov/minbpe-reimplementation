import pickle
from pathlib import Path


class BasicTokenizer:

    def __init__(self, vocabulary_file_name: str):
        """Load a previously trained vocabulary from `vocabularies/{vocabulary_file_name}`.

        Args:
            vocabulary_file_name: Name of the pickled vocabulary file, relative
                to the `vocabularies/` directory.

        Raises:
            FileNotFoundError: If no vocabulary file exists at the resolved path.
        """
        # Resolved relative to this module's directory, not the current working directory.
        vocabulary_path = Path(__file__).resolve().parent / "vocabularies" / vocabulary_file_name
        try:
            with open(vocabulary_path, 'rb') as file:
                vocabulary = pickle.load(file)
                self.encode_vocabulary = vocabulary["encode"]
                self.decode_vocabulary = vocabulary["decode"]
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found: {vocabulary_path}") from None


    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        new_token_id = 256
        num_merges = len(self.encode_vocabulary) - new_token_id

        for _ in range(num_merges):
            tokens, token_pair = BasicTokenizer.merge_most_frequent_token_pairs(tokens, new_token_id)
            self.encode_vocabulary[token_pair] = new_token_id
            new_token_id = new_token_id + 1

        return tokens


    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token ids produced by `encode` back into text.

        Args:
            tokens: A sequence of token ids, using this tokenizer's vocabulary.

        Returns:
            The decoded text, with any invalid UTF-8 token sequences replaced.
        """
        # Maps a merged token id back to the token pair it was created from.
        decoded_tokens = []
        for i in range(len(tokens)):
            if tokens[i] < 256:
                # Raw token value - nothing to expand.
                decoded_tokens.append(tokens[i])
            else:
                decoded_tokens.extend(self.decode_vocabulary[tokens[i]])
        text = bytes(decoded_tokens).decode("utf-8", errors="replace")
        return text


    @staticmethod
    def train(input_file_name: str, vocabulary_size: int, vocabulary_file_name: str, verbose: bool = False):
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
            FileNotFoundError: If no input file exists at the resolved path.
        """
        if vocabulary_size <= 256:
            # 0-255 are already reserved for raw token values, so at least one merge is required.
            raise ValueError(f"vocabulary_size must be greater than 256, got {vocabulary_size}")

        input_path = Path(__file__).resolve().parent / "inputs" / input_file_name
        try:
            with open(input_path, 'r', encoding="utf-8") as input_file:
                text = input_file.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {input_path}") from None

        # Keep the pre-merge tokens around so we can report the compression ratio later.
        original_tokens = list(text.encode("utf-8"))

        # Work on a copy so `original_tokens` still reflects the starting length.
        tokens = list(original_tokens)
        encode_vocabulary = {}
        decode_vocabulary = {}
        new_token_id = 256
        num_merges = vocabulary_size - new_token_id

        for _ in range(num_merges):
            try:
                tokens, token_pair = BasicTokenizer.merge_most_frequent_token_pairs(tokens, new_token_id)
            except ValueError:
                # merge_most_frequent_token_pairs raises when no pair is left to merge.
                raise ValueError(
                    f"Cannot reach vocabulary_size={vocabulary_size}: ran out of token "
                    f"pairs to merge after {new_token_id - 256} merge(s)"
                ) from None
            encode_vocabulary[token_pair] = new_token_id
            decode_vocabulary[new_token_id] = []
            for token_id in token_pair:
                if token_id >= 256:
                    decode_vocabulary[new_token_id].extend(decode_vocabulary[token_id])
                else:
                    decode_vocabulary[new_token_id].append(token_id)

            if verbose:
                print(f"Added new token - {token_pair} - to the vocabulary with token ID - {new_token_id}")

            new_token_id = new_token_id + 1

        print("--------------------------------------------------")
        print(f"Initial number of tokens - {len(original_tokens)}")
        print(f"Final number of tokens   - {len(tokens)}")
        # How many raw tokens, on average, each remaining token now represents.
        print(f"Compression ratio        - {len(original_tokens) / len(tokens)}x")

        vocabulary = {
            "encode": encode_vocabulary,
            "decode": decode_vocabulary
        }
        vocabulary_path = Path(__file__).resolve().parent / "vocabularies" / vocabulary_file_name
        with open(vocabulary_path, 'wb') as output_file:
            pickle.dump(vocabulary, output_file)

        print("--------------------------------------------------")
        print(f"Saved the vocabulary to the path - {vocabulary_path}")


    @staticmethod
    def merge_most_frequent_token_pairs(tokens: list[int], merged_token_id: int) -> list[int]:
        """Replace every occurrence of the most frequent token pair with a new token.

        Args:
            tokens: A sequence of integer token ids.
            merged_token_id: The token id to substitute for the most frequent pair.

        Returns:
            A new list of tokens with each occurrence of the most frequent token
            pair collapsed into a single `merged_token_id`.

        Raises:
            ValueError: If `tokens` has fewer than 2 elements, since no token
                pair can be formed to merge.
        """
        output_tokens = []
        token_pair_counts = BasicTokenizer.get_token_pair_counts(tokens)
        most_frequent_token_pair = BasicTokenizer.get_most_frequent_token_pair(token_pair_counts)

        # Walk the tokens, replacing every occurrence of the target pair with
        # `merged_token_id` and copying everything else through unchanged.
        i = 0
        while i < len(tokens):
            # `i < len(tokens) - 1` guards against reading `tokens[i + 1]` when
            # `i` is the last index, which would otherwise raise an IndexError
            # if that final token happens to equal `token_pair[0]`.
            if i < len(tokens) - 1 and tokens[i] == most_frequent_token_pair[0] and tokens[i + 1] == most_frequent_token_pair[1]:
                # Found the target pair - emit the merged token and skip both.
                output_tokens.append(merged_token_id)
                i = i + 2
            else:
                # No match at this position - keep the token and advance by one.
                output_tokens.append(tokens[i])
                i = i + 1

        return output_tokens, most_frequent_token_pair


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
    def get_most_frequent_token_pair(token_pair_counts: dict[tuple[int, int], int]) -> tuple[int, int]:
        """Return the most frequent token pair.

        Args:
            token_pair_counts: A dictionary mapping token pairs to their counts.

        Returns:
            The (token, next_token) pair with the highest count.
        """
        if not token_pair_counts:
            raise ValueError("token_pair_counts must not be empty")
        return max(token_pair_counts.items(), key=lambda item: item[1])[0]
