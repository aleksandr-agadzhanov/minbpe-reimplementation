import pickle
import tiktoken

from tokenizers.basic_tokenizer import BasicTokenizer
from tokenizers.gpt4_tokenizer import GPT4Tokenizer
from utils import print_vocabulary

vocabulary_file_name = "tokenizer_1000.pkl"
input_file_name = "text.txt"

# with open(f"vocabularies/{vocabulary_file_name}", 'rb') as file:
#     vocabulary = pickle.load(file)

# for tokens in vocabulary["decode"].values():
#     # A merged token can be a partial UTF-8 sequence, so decoding may fail - replace instead of crashing.
#     characters = bytes(tokens).decode("utf-8", errors="replace")
#     print(characters)

# tokenizer = BasicTokenizer(vocabulary_file_name)
tokenizer = GPT4Tokenizer(vocabulary_file_name)

# with open(f"inputs/{input_file_name}", 'r') as input_file:
#     text = input_file.read()

text = "hello world!!!? (안녕하세요!) lol123 😉"

tokens = tokenizer.encode(text)
decoded_text = tokenizer.decode(tokens)

print(tokens)
print(decoded_text)

# print_vocabulary(vocabulary_file_name, 100)

# tiktoken_tokenizer = tiktoken.get_encoding("cl100k_base")
# tiktoken_tokens = tiktoken_tokenizer.encode(text)
# tiktoken_decoded_text = tiktoken_tokenizer.decode(tiktoken_tokens)

# print(tiktoken_tokens)
# print(tiktoken_decoded_text)
