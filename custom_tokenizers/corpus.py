from datasets import load_dataset


def get_training_corpus(corpus_path: str):
    dataset = (
        load_dataset('json', data_files=[corpus_path], split="train")
        .filter(lambda x: x["meta"]["quality"] == "HIGH")
    )
    return dataset["text"]
