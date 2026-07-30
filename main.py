from basic_tokenizer import BasicTokenizer

vocabulary_file_name = "tokenizer_2000.pkl"
input_file_name = "text.txt"

tokenizer = BasicTokenizer(vocabulary_file_name)

with open(f"inputs/{input_file_name}", 'r') as input_file:
    text = input_file.read()

tokens = tokenizer.encode(text)
print(tokens[:100])
decoded_text = tokenizer.decode(tokens)

with open("outputs/decoded_text.txt", 'w') as output_file:
    output_file.write(decoded_text)
