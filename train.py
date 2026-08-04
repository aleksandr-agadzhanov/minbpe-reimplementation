from tokenizers.regex_tokenizer import RegexTokenizer

input_file_name = "fineweb_edu_100mb.txt"
num_base_tokens = 256
num_merged_tokens = 16384 - 256
special_tokens = {
    "<|endoftext|>": 16385,
    "<|fim_prefix|>": 16386,
    "<|fim_middle|>": 16387,
    "<|fim_suffix|>": 16388,
    "<|endofprompt|>": 16389,
}

vocabulary_size = num_base_tokens + num_merged_tokens + len(special_tokens)
vocabulary_file_name = "fineweb_edu_100mb_16389.pkl"

RegexTokenizer.train(
    input_file_name,
    vocabulary_size,
    vocabulary_file_name,
    verbose=True,
    special_tokens=special_tokens,
)
