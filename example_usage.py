from pathlib import Path

from minbpe_tokenizers import RegexTokenizer

INPUT_FILE_NAME = "fineweb_edu_1mb.txt"
VOCABULARY_FILE_NAME = "example_usage.pkl"
VOCABULARY_SIZE = 256 + 50  # small, so this example trains in a couple of seconds

vocabulary_path = (
    Path(__file__).resolve().parent
    / RegexTokenizer.VOCABULARIES_DIRECTORY_NAME
    / VOCABULARY_FILE_NAME
)
vocabulary_path.unlink(missing_ok=True)  # train() refuses to overwrite an existing file

RegexTokenizer.train(
    INPUT_FILE_NAME, VOCABULARY_SIZE, VOCABULARY_FILE_NAME, verbose=True
)

tokenizer = RegexTokenizer(VOCABULARY_FILE_NAME)

text = "Byte Pair Encoding turns raw text into a compact sequence of learned tokens."
tokens = tokenizer.encode(text)

print()
print(f"Text:          {text!r}")
print(f"Encoded ids:   {tokens}")
print(f"Decoded parts: {tokenizer.decode_as_list(tokens)}")
print(f"Decoded text:  {tokenizer.decode(tokens)!r}")
