import pickle

from basic_tokenizer import BasicTokenizer

vocabulary_file_name = "tokenizer_1000.pkl"
input_file_name = "text.txt"

with open(f"vocabularies/{vocabulary_file_name}", 'rb') as file:
    vocabulary = pickle.load(file)

# for tokens in vocabulary["decode"].values():
#     # A merged token can be a partial UTF-8 sequence, so decoding may fail - replace instead of crashing.
#     characters = bytes(tokens).decode("utf-8", errors="replace")
#     print(characters)

tokenizer = BasicTokenizer(vocabulary_file_name)

# with open(f"inputs/{input_file_name}", 'r') as input_file:
#     text = input_file.read()


text = "Hello there I am not using any social media"

tokens = tokenizer.encode(text)
print(tokens)

decoded_text = tokenizer.decode(tokens)
print(decoded_text)
