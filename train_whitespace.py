import argparse

from tokenizers import trainers
from transformers import PreTrainedTokenizerFast

from custom_tokenizers.corpus import get_training_corpus
from custom_tokenizers.whitespace import WHITESPACE_TOKENIZER

tokenizer = WHITESPACE_TOKENIZER

def train(corpus_path: str):
    trainer = trainers.WordLevelTrainer(vocab_size=32000, special_tokens=["<PAD>", "<UNK>", "<BOS>", "<EOS>"])
    tokenizer.train_from_iterator(get_training_corpus(corpus_path), trainer=trainer)
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    pretrained_tokenizer.save_pretrained("pretrained/whitespace_tokenizer")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument("--corpus-path", type=str, help="Path to the training corpus", required=True)
    args = parser.parse_args()
    train(args.corpus_path)
