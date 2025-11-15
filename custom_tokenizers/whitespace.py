from tokenizers import (
    models,
    normalizers,
    pre_tokenizers,
    Tokenizer,
    processors
)

tokenizer = Tokenizer(models.WordLevel(unk_token="<UNK>", vocab={}))

tokenizer.normalizer = normalizers.NFKC()
tokenizer.post_processor = processors.TemplateProcessing(
    single="<BOS>:0 $A:0",
    pair="<BOS>:0 $A:0 <BOS>:1 $B:1",
    special_tokens=[
        ("<BOS>", 1),
    ],
)
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Punctuation(),
    ]
)

WHITESPACE_TOKENIZER = tokenizer