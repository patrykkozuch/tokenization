from tokenizers import (
    models,
    normalizers,
    Tokenizer,
    pre_tokenizers,
    decoders,
    processors
)

tokenizer = Tokenizer(models.BPE(unk_token="<UNK>", dropout=0.1))

tokenizer.normalizer = normalizers.NFKC() # Sentence-piece uses NFKC normalization
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.post_processor = processors.TemplateProcessing(
    single="<BOS>:0 $A:0",
    pair="<BOS>:0 $A:0 <BOS>:1 $B:1",
    special_tokens=[
        ("<BOS>", 1),
    ],
)
tokenizer.decoder = decoders.Sequence([decoders.Metaspace(), decoders.ByteFallback()])

BPE_TOKENIZER = tokenizer