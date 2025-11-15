from tokenizers import (
    models,
    normalizers,
    Tokenizer, pre_tokenizers, decoders
)

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", dropout=0.1))

tokenizer.normalizer = normalizers.NFKC() # Sentence-piece uses NFKC normalization
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.decoder = decoders.Metaspace()

BPE_TOKENIZER = tokenizer