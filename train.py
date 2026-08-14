from tokenizers.regex_tokenizer import RegexTokenizer

INPUT_FILE_NAME = "fineweb_edu_100mb.txt"
NUM_BASE_TOKENS = 256
NUM_MERGED_TOKENS = 16383 - 256
SPECIAL_TOKENS = {
    "<|endoftext|>": 16383
}
VOCABULARY_FILE_NAME = "fineweb_edu_100mb_16384.pkl"

vocabulary_size = NUM_BASE_TOKENS + NUM_MERGED_TOKENS + len(SPECIAL_TOKENS)

RegexTokenizer.train(
    INPUT_FILE_NAME,
    vocabulary_size,
    VOCABULARY_FILE_NAME,
    verbose=True,
    special_tokens=SPECIAL_TOKENS,
)
