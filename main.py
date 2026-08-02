import pickle

import tiktoken

from tokenizers.basic_tokenizer import BasicTokenizer
from tokenizers.gpt4_tokenizer import GPT4Tokenizer
from utils import print_vocabulary

vocabulary_file_name = "gpt4_tokenizer_1000.pkl"

# tokenizer = BasicTokenizer(vocabulary_file_name)
# tokenizer = GPT4Tokenizer(vocabulary_file_name)

# text = "Hello world how are you doing? It is a new day today!"

# tokens = tokenizer.encode(text)
# decoded_text_as_list = tokenizer.decode_as_list(tokens)
# decoded_text = tokenizer.decode(tokens)

# print(tokens)
# print(decoded_text_as_list)
# print(decoded_text)

# print_vocabulary(vocabulary_file_name, 100)

# tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")
# tiktoken_tokens = tiktoken_tokenizer.encode(text)
# tiktoken_decoded_text = tiktoken_tokenizer.decode(tiktoken_tokens)

# print(tiktoken_tokens)
# print(tiktoken_decoded_text)
