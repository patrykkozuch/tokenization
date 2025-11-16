from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer,
    Regex
)

tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>", vocab={}))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),  # Sentence-piece uses NFKC normalization
    normalizers.Replace(Regex(" {2,}"), " "),
])

tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)

WHITESPACE_TOKENIZER = tokenizer