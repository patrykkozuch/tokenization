from datasets import load_dataset


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

    dataset = (
        load_dataset('json', data_files=[corpus_path], split="train")
        .filter(lambda x: x["meta"]["quality"] == "HIGH")
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
