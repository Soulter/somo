import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from somo.train import Config, DatasetConfig, load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config.",
    )
    return parser.parse_args()


def iter_local_text(dataset: DatasetConfig):
    assert dataset.data_path is not None, "local dataset must have data_path"
    with open(dataset.data_path, "r", encoding="utf-8") as f:
        yield f.read()


def iter_jsonl_text(dataset: DatasetConfig, max_documents: int):
    count = 0
    assert dataset.data_path is not None, "jsonl dataset must have data_path"
    with open(dataset.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            text = row.get(dataset.text_column)
            if not text:
                continue

            yield text
            count += 1
            if count >= max_documents:
                break


def iter_hf_text(
    dataset_config: DatasetConfig,
    max_documents: int,
    shuffle_buffer_size: int,
    seed: int,
):
    assert dataset_config.dataset_name is not None, "HF dataset must have dataset_name"
    dataset = load_dataset(
        dataset_config.dataset_name,
        dataset_config.dataset_config,
        split=dataset_config.dataset_split,
        streaming=True,
    )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=seed,
        )

    count = 0
    for row in dataset:
        text = row.get(dataset_config.text_column)
        if not text:
            continue

        yield text
        count += 1
        if count >= max_documents:
            break


def iter_parquet_text(
    dataset_config: DatasetConfig,
    max_documents: int,
    shuffle_buffer_size: int,
    seed: int,
):
    dataset = load_dataset(
        "parquet",
        data_files=str(dataset_config.data_path),
        split=dataset_config.dataset_split,
        streaming=True,
    )
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            seed=seed,
        )

    count = 0
    for row in dataset:
        text = row.get(dataset_config.text_column)
        if not text:
            continue

        yield text
        count += 1
        if count >= max_documents:
            break


def iter_dataset_text(
    dataset_config: DatasetConfig,
    max_documents: int,
    shuffle_buffer_size: int,
    seed: int,
):
    if dataset_config.data_source == "local":
        yield from iter_local_text(dataset_config)
        return

    if dataset_config.data_source == "jsonl":
        yield from iter_jsonl_text(dataset_config, max_documents)
        return

    if dataset_config.data_source == "hf":
        yield from iter_hf_text(
            dataset_config,
            max_documents=max_documents,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        return

    if dataset_config.data_source == "parquet":
        yield from iter_parquet_text(
            dataset_config,
            max_documents=max_documents,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
        )
        return

    raise ValueError(
        f"unknown tokenizer dataset source: "
        f"{dataset_config.name} uses {dataset_config.data_source}"
    )


def get_text_iterator(config: Config):
    assert config.train_datasets is not None, "config must have train_datasets"
    total_weight = sum(dataset.weight for dataset in config.train_datasets)
    lengths = []

    for dataset in config.train_datasets:
        if dataset.data_source == "local":
            lengths.append(1)
            continue

        length = round(config.tokenizer_train_max_documents * dataset.weight / total_weight)
        lengths.append(max(1, length))

    def iterator():
        assert config.train_datasets is not None, "config must have train_datasets"
        for i, dataset in enumerate(config.train_datasets):
            shuffle_buffer_size = (
                dataset.shuffle_buffer_size
                if dataset.shuffle_buffer_size is not None
                else config.shuffle_buffer_size
            )
            yield from iter_dataset_text(
                dataset,
                max_documents=lengths[i],
                shuffle_buffer_size=shuffle_buffer_size,
                seed=config.seed + i,
            )

    return iterator(), sum(lengths)


def train_tokenizer(config_path: Path):
    config = load_config(config_path)

    # Byte Pair Encoding
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))  # <unk>: unknown
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=config.tokenizer_vocab_size,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevel.alphabet(),
    )

    texts, length = get_text_iterator(config)
    tokenizer.train_from_iterator(texts, trainer=trainer, length=length)

    config.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(config.tokenizer_path))

    print("saved tokenizer to", config.tokenizer_path)
    print("vocab size:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args.config)
