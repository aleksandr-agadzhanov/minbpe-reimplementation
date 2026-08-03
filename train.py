from tokenizers.regex_tokenizer import RegexTokenizer

input_file_name = "fineweb_edu_1mb.txt"
vocabulary_size = 256 + 1000
vocabulary_file_name = "stub.pkl"
# special_tokens = {
#     "<|endoftext|>": 357,
#     "<|fim_prefix|>": 358,
#     "<|fim_middle|>": 359,
#     "<|fim_suffix|>": 360,
#     "<|endofprompt|>": 361
# }

RegexTokenizer.train(
    input_file_name,
    vocabulary_size,
    vocabulary_file_name,
    verbose=True,
    # special_tokens=special_tokens
)
