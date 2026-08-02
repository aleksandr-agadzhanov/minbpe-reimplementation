import pickle
from pathlib import Path

from tokenizers.basic_tokenizer import BasicTokenizer


def print_vocabulary(vocabulary_file_name: str, num_tokens: int) -> None:
    """Print the longest tokens in a trained vocabulary, longest first.

    Args:
        vocabulary_file_name: Name of the pickled vocabulary file, relative
            to the `vocabularies/` directory.
        num_tokens: How many of the longest tokens to print.
    """
    vocabulary_path = (
        Path(__file__).resolve().parent
        / BasicTokenizer.VOCABULARIES_DIRECTORY_NAME
        / vocabulary_file_name
    )
    with open(vocabulary_path, "rb") as file:
        vocabulary = pickle.load(file)
    decode_vocabulary = vocabulary[BasicTokenizer.DECODE_VOCABULARY_KEY]

    sorted_tokens = sorted(decode_vocabulary.values(), key=len, reverse=True)
    for rank, tokens in enumerate(sorted_tokens[:num_tokens], start=1):
        # A merged token can be a partial UTF-8 sequence, so decoding may fail - replace instead of crashing.
        characters = bytes(tokens).decode(BasicTokenizer.TEXT_ENCODING, errors="replace")
        print(f"{rank}. {characters}")
