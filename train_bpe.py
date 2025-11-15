import argparse
from string import printable
from tokenizers import trainers
from transformers import PreTrainedTokenizerFast

from custom_tokenizers.bpe import BPE_TOKENIZER
from custom_tokenizers.corpus import get_training_corpus

tokenizer = BPE_TOKENIZER

initial_alphabet = printable + "ąęćłńóśźżĄĘĆŁŃÓŚŹŻ"

def train(corpus_path: str):
    trainer = trainers.BpeTrainer(
        vocab_size=32000,
        special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        initial_alphabet=list(initial_alphabet)
    )
    tokenizer.train_from_iterator(get_training_corpus(corpus_path), trainer=trainer)
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    # Let's follow the polish model names - there was "Bielik", "Sójka", let's have "Bocian" now
    pretrained_tokenizer.save_pretrained("pretrained/bocian_tokenizer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--corpus_path", type=str, help="Path to the training corpus", required=True)
    args = parser.parse_args()
    train(args.corpus_path)
