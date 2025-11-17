from tokenizers import (
    models,
    normalizers,
    Tokenizer,
    pre_tokenizers,
    decoders
)

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", dropout=0.1, fuse_unk=True))

tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.decoder = decoders.Metaspace()

BPE_TOKENIZER = tokenizer
