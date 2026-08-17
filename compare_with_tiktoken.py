from collections import Counter

import tiktoken

from minbpe_tokenizers.basic_tokenizer import BasicTokenizer
from minbpe_tokenizers.regex_tokenizer import RegexTokenizer

VOCABULARY_FILE_NAME = "fineweb_edu_100mb_16389.pkl"
TIKTOKEN_ENCODING_NAME = "cl100k_base"

# Ten short sentences, each exercising a different tokenization scenario.
TEST_PHRASES = [
    "I can't believe it's already Friday, can you?!",
    "The meeting is scheduled for 3:45 PM on July 21st, 2024.",
    "Line one.\nLine two.\n\nLine three after a blank line.",
    "The so-called 'quick fix' didn't actually solve anything at all.",
    "Dr. Smith arrived at 9 a.m. sharp, ready to start the important meeting.",
    "STOP RIGHT THERE AND LISTEN CAREFULLY TO ME!!!",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "The total cost came to $19.99 after a 15% discount was applied.",
    "Visit https://example.com/page?id=123&ref=test for more info.",
    "Great job team! Let's keep pushing forward together, well done!",
]


def _tiktoken_decode_as_list(
    encoding: tiktoken.Encoding, tokens: list[int]
) -> list[str]:
    """Decode each tiktoken token id individually, mirroring `decode_as_list`."""
    return [
        token_bytes.decode(BasicTokenizer.TEXT_ENCODING, errors="replace")
        for token_bytes in encoding.decode_tokens_bytes(tokens)
    ]


def _token_list_similarity(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compare two decoded token lists.

    Returns 1.0 if the lists match exactly. Otherwise, returns the multiset
    overlap between the two lists (each shared token counted once per
    occurrence in both) divided by their combined total token count - this
    still evaluates to 1.0 for an exact match and to 0.0 when nothing overlaps.
    """
    if not tokens_a and not tokens_b:
        return 1.0
    common_count = sum((Counter(tokens_a) & Counter(tokens_b)).values())
    return 2 * common_count / (len(tokens_a) + len(tokens_b))


regex_tokenizer = RegexTokenizer(VOCABULARY_FILE_NAME)
tiktoken_encoding = tiktoken.get_encoding(TIKTOKEN_ENCODING_NAME)

similarities = []
for phrase_index, phrase in enumerate(TEST_PHRASES, start=1):
    regex_tokens = regex_tokenizer.decode_as_list(regex_tokenizer.encode(phrase))
    tiktoken_tokens = _tiktoken_decode_as_list(
        tiktoken_encoding, tiktoken_encoding.encode(phrase)
    )

    similarity = _token_list_similarity(regex_tokens, tiktoken_tokens)
    similarities.append(similarity)

    print(f"{phrase_index}. {phrase!r}")
    print(f"   RegexTokenizer - {regex_tokens}")
    print(f"   tiktoken       - {tiktoken_tokens}")
    print(f"   Similarity     - {similarity:.1%}")
    print()

average_similarity = sum(similarities) / len(similarities)
print(
    f"Average similarity across {len(TEST_PHRASES)} phrases: {average_similarity:.1%}"
)
