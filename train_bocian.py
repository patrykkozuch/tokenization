import argparse
import string
from tokenizers import trainers, processors, decoders
from transformers import PreTrainedTokenizerFast

from custom_tokenizers.bocian import BPE_TOKENIZER
from custom_tokenizers.corpus import get_training_corpus

tokenizer = BPE_TOKENIZER

initial_alphabet = string.printable + "ąęćłńóśźżĄĘĆŁŃÓŚŹŻ"

def train(corpus_path: str):
    trainer = trainers.BpeTrainer(
        vocab_size=32768,
        min_frequency=2,
        limit_alphabet=1000,
        special_tokens=["<UNK>", "<BOS>", "<PAD>", "<EOS>"],
        initial_alphabet=list(initial_alphabet)
    )
    tokenizer.train_from_iterator(get_training_corpus(corpus_path), trainer=trainer)

    bos_token_id = tokenizer.token_to_id("<BOS>")
    eos_token_id = tokenizer.token_to_id("<EOS>")

    tokenizer.post_processor = processors.TemplateProcessing(
        single="<BOS>:0 $A:0 <EOS>:0",
        pair="<BOS>:0 $A:0 <EOS>:0 <BOS>:1 $B:1 <EOS>:1",
        special_tokens=[("<BOS>", bos_token_id), ("<EOS>", eos_token_id)],
    )

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

    print(pretrained_tokenizer.encode("Zażółć gęślą jaźń.", add_special_tokens=True))
    print(pretrained_tokenizer.decode(pretrained_tokenizer.encode("Zażółć gęślą jaźń.", add_special_tokens=True), skip_special_tokens=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--corpus-path", type=str, help="Path to the training corpus", required=True)
    args = parser.parse_args()
    train(args.corpus_path)
