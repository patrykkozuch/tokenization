# Tokenization module

This module provides custom tokenizers for tokenization benchmark for the second laboratory of computational linguistics classes.

Report: [Weights and biases](https://wandb.ai/patryk-kozuch-personal/BocianTokenizer/reports/Tokenization-Efficiency-Benchmark--VmlldzoxNTE0MzEwMQ)

## Tokenizers

- `Whitespace`: A simple tokenizer that splits text based on whitespace and punctuation.
- `BocianBPE`: A BPE-based tokenizer.
- `BocianSentencePiece`: A SentencePiece (+BPE) tokenizer.

Both tokenizers were trained on a Polish poetry corpus (`plwikisource` dataset from the SpeakLeash library).

## Training the Tokenizers

To train the tokenizers, you can use the following code snippets:

### Whitespace Tokenizer

```bash
uv run train_whitespace.py --corpus-path path/to/corpus.jsonl.zst
```

### Bocian Tokenizer (BPE)

```bash
uv run train_bocian.py --corpus-path path/to/corpus.jsonl.zst
```

### Bocian Tokenizer (SentencePiece)

```bash
uv run train_bocian_sp.py --corpus-path path/to/corpus.jsonl.zst
```
