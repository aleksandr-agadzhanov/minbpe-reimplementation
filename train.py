from basic_tokenizer import BasicTokenizer

input_file_name = "text.txt"
vocabulary_size = 2256
vocabulary_file_name = "tokenizer_2000.pkl"

BasicTokenizer.train(input_file_name, vocabulary_size, vocabulary_file_name, verbose=True)
