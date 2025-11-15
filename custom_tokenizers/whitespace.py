from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer
)

tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>", vocab={}))

tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)

WHITESPACE_TOKENIZER = tokenizer