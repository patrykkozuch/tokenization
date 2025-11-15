from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer, Regex,
)

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.NFKC(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

BPE_TOKENIZER = tokenizer
