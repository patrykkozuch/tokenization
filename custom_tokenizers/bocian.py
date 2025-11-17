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
tokenizer.decoder = decoders.Sequence([decoders.Metaspace(), decoders.Strip(" ", 1, 0)])

BPE_TOKENIZER = tokenizer
