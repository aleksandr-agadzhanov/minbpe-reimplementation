from basic_tokenizer import BasicTokenizer

input_file_name = "tiny_shakespeare.txt"
vocabulary_size = 1256
vocabulary_file_name = "tokenizer_2000.pkl"

BasicTokenizer.train(input_file_name, vocabulary_size, vocabulary_file_name, verbose=True)
