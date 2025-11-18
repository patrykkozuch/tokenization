import glob

from datasets import load_dataset


def get_training_corpus(corpus_path: str):
    data_files = list(glob.glob(corpus_path))
    dataset = (
        load_dataset('json', data_files=data_files, split="train", num_proc=20)
        .filter(lambda x: x.get("meta", {}).get("quality", "") == "HIGH", num_proc=20)
    )

    for i in range(0, len(dataset), 1000):
        batch = dataset[i: i + 1000]["text"]
        yield batch
