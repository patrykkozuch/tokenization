# Tokenization module

This module provides custom tokenizers for the second laboratory of computational linguistics classes.

## Tokenizers

- `WhitespaceTokenizer`: A simple tokenizer that splits text based on whitespace and punctuation.
- `BocianTokenizer`: A BPE-based tokenizer.

Both tokenizers were trained on a Polish poetry corpus (`plwikisource` dataset from the SpeakLeash library).

## Training the Tokenizers

To train the tokenizers, you can use the following code snippets:

### Whitespace Tokenizer

```bash
uv run train_whitespace.py --corpus-path path/to/corpus.jsonl.zst
```

### Bocian Tokenizer

```bash
uv run train_bocian.py --corpus-path path/to/corpus.jsonl.zst