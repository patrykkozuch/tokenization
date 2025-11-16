from tokenizers import (
    models,
    normalizers,
    Tokenizer,
    pre_tokenizers,
)
from tokenizers.tokenizers import Regex

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", dropout=0.1, fuse_unk=True, byte_fallback=True))

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFKC(),  # Sentence-piece uses NFKC normalization
    normalizers.Replace(Regex(" {2,}"), " "),
])

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()


BPE_TOKENIZER = tokenizer
