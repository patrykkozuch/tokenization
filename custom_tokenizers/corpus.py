import glob

from datasets import load_dataset, Features, Value, List
from sympy.integrals.meijerint_doc import category
from tokenizers.implementations import SentencePieceBPETokenizer


def get_training_corpus(corpus_path: str):
    """
    Clean corpus to remove non-Polish text before training.
    This prevents Greek, Cyrillic, and other non-Latin characters from entering vocab.

    Args:
        text_iterator: Iterator yielding batches of text

    Yields:
        Cleaned batches of text
    """
    import re

    # Define what characters to keep
    # Latin alphabet + Polish diacritics + numbers + basic punctuation
    allowed_pattern = re.compile(
        r'[a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s.,!?;:\-\'\"()\[\]{}/@#$%&*+=<>_~`|\\]+'
    )

    data_files = list(glob.glob(corpus_path))
    dataset = (
        load_dataset('json', data_files=data_files, split="train", num_proc=20)
        .filter(lambda x: x.get("meta", {}).get("quality", "") == "HIGH", num_proc=20)
    )

    for i in range(0, len(dataset), 1000):
        batch = dataset[i : i + 1000]["text"]

        cleaned_batch = []
        for text in batch:
            if not text:
                continue
            # Keep only allowed characters
            cleaned_text = ' '.join(allowed_pattern.findall(text))
            if cleaned_text.strip():
                cleaned_batch.append(cleaned_text)

        if cleaned_batch:
            yield cleaned_batch
