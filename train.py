from tokenizers.basic_tokenizer import BasicTokenizer
from tokenizers.gpt4_tokenizer import GPT4Tokenizer

input_file_name = "tiny_shakespeare.txt"
vocabulary_size = 1256
vocabulary_file_name = "tokenizer_1000_gpt4.pkl"

# BasicTokenizer.train(input_file_name, vocabulary_size, vocabulary_file_name, verbose=True)
GPT4Tokenizer.train(input_file_name, vocabulary_size, vocabulary_file_name, verbose=True)
