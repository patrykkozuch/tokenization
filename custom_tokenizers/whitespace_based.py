from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer, Regex,
)

tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>", vocab={}))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

WHITESPACE_TOKENIZER = tokenizer