import argparse
from tokenizers import processors, Tokenizer, normalizers, pre_tokenizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast

from corpus import get_training_corpus


def train(corpus_path: str):
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.normalizer = normalizers.NFC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=32768,
        min_frequency=100,
        special_tokens=["<UNK>", "<BOS>", "<PAD>", "<EOS>"]
    )
    tokenizer.train_from_iterator(get_training_corpus(corpus_path), trainer=trainer)

    bos_token_id = tokenizer.token_to_id("<BOS>")
    eos_token_id = tokenizer.token_to_id("<EOS>")

    tokenizer.post_processor = processors.Sequence([
        processors.ByteLevel(trim_offsets=True),
        processors.TemplateProcessing(
            single="<BOS>:0 $A:0 <EOS>:0",
            pair="<BOS>:0 $A:0 <EOS>:0 <BOS>:1 $B:1 <EOS>:1",
            special_tokens=[("<BOS>", bos_token_id), ("<EOS>", eos_token_id)]
        )
    ])

    tokenizer.decoder = decoders.ByteLevel()

    pretrained_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<BOS>",
        eos_token="<EOS>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        padding_side="right"
    )
    # Let's follow the polish model names - there was "Bielik", "Sójka", let's have "Bocian" now
    pretrained_tokenizer.save_pretrained("pretrained/bocian_tokenizer")

    print(pretrained_tokenizer("Zażół gęślą jaźń"))
    print(tokenizer.encode("Zażółć gęśłą jaźń").tokens)
    print(pretrained_tokenizer.encode("Zażółć gęślą jaźń.", add_special_tokens=True))
    print(pretrained_tokenizer.decode(pretrained_tokenizer.encode("Zażółć gęślą jaźń.", add_special_tokens=True),
                                      skip_special_tokens=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--corpus-path", type=str, help="Path to the training corpus", required=True)
    args = parser.parse_args()
    train(args.corpus_path)
