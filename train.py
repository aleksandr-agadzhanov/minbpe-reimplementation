from tokenizers.gpt4_tokenizer import GPT4Tokenizer
from tokenizers.basic_tokenizer import BasicTokenizer

input_file_name = "tiny_shakespeare.txt"
vocabulary_size = 256 + 500 + 1
vocabulary_file_name = "stub.pkl"
special_tokens = {
    "<|endoftext|>": 757
}

BasicTokenizer.train(input_file_name, vocabulary_size, vocabulary_file_name, verbose=True)
