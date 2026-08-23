# minbpe-reimplementation

A from-scratch implementation of Byte Pair Encoding (BPE) tokenization in
Python, following Andrej Karpathy's ["Let's build the GPT Tokenizer"](https://www.youtube.com/watch?v=zduSFxRajkE)
YouTube tutorial and using his reference repository [karpathy/minbpe](https://github.com/karpathy/minbpe)
as a guide. Includes training a tokenizer on raw text, encoding/decoding,
and a full unit test suite for the core BPE logic.

The core BPE algorithm (`BasicTokenizer`, `RegexTokenizer`, and the training
loop) was written by hand, without AI assistance, to build a real
understanding of how it works. AI was used afterward to add docstrings and
comments, split the code into smaller functions, write the unit tests, and
optimize the training loop (e.g. incrementally updating pair counts instead
of rescanning the whole corpus after every merge).

## Tokenizers

- `BasicTokenizer`: the core byte-level BPE tokenizer. It repeatedly merges
  the most frequent adjacent pair of tokens until it reaches a target
  vocabulary size, and supports special tokens (e.g. `<|endoftext|>`).
- `RegexTokenizer`: a `BasicTokenizer` subclass that first splits text into
  chunks using a GPT-4-style regex pattern (separating letters, numbers,
  punctuation, contractions, and whitespace) before training or encoding.
  Merges are only ever applied within a chunk, never across a chunk boundary,
  which keeps tokens more semantically meaningful (e.g. it won't merge a
  trailing space into an unrelated word).

## Training data and results

A `RegexTokenizer` vocabulary was trained on 100 MB of text from the
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
dataset, producing a vocabulary of 16,384 tokens: 256 base byte tokens,
16,127 learned merges, and 1 special token. In this setup, the full
training run completed in under 15 minutes.

`compare_with_tiktoken.py` compares this tokenizer's output against OpenAI's
`tiktoken` (`cl100k_base`) on a set of 10 varied English test sentences,
decoding each token individually and measuring how closely the two token
sequences match. This tokenizer reaches a 72.5% average similarity to
`tiktoken` - a reasonable result given it was trained on far less data and a
smaller vocabulary than `cl100k_base`.

## Running locally

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run the unit tests:

```bash
pytest tests/ -q
```

Try the tokenizer without training your own vocabulary, using the example
script (trains a small demo vocabulary on the bundled 1 MB sample dataset):

```bash
python example_usage.py
```

Train a full-size vocabulary yourself (expects a text file under
`training_datasets/`):

```bash
python train.py
```

Compare a trained vocabulary against `tiktoken`:

```bash
python compare_with_tiktoken.py
```
