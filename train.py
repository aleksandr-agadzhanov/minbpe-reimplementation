from tokenizers.regex_tokenizer import RegexTokenizer

INPUT_FILE_NAME = "fineweb_edu_100mb.txt"
NUM_BASE_TOKENS = 256
NUM_MERGED_TOKENS = 16384 - 256
SPECIAL_TOKENS = {
    "<|endoftext|>": 16385,
    "<|fim_prefix|>": 16386,
    "<|fim_middle|>": 16387,
    "<|fim_suffix|>": 16388,
    "<|endofprompt|>": 16389,
}
VOCABULARY_FILE_NAME = "fineweb_edu_100mb_16389.pkl"

vocabulary_size = NUM_BASE_TOKENS + NUM_MERGED_TOKENS + len(SPECIAL_TOKENS)

RegexTokenizer.train(
    INPUT_FILE_NAME,
    vocabulary_size,
    VOCABULARY_FILE_NAME,
    verbose=True,
    special_tokens=SPECIAL_TOKENS,
)
