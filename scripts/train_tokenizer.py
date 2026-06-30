import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


@dataclass
class RawDatasetConfig:
    name: str
    data_source: str
    weight: float = 1.0

    data_path: Path | None = None
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"

    text_column: str = "text"
    shuffle_buffer_size: int | None = None


@dataclass
class TokenizerConfig:
    raw_datasets: list[RawDatasetConfig]
    tokenizer_path: Path
    tokenizer_vocab_size: int = 8192
    tokenizer_train_max_documents: int = 100_000
    shuffle_buffer_size: int = 10_000
    seed: int = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the tokenizer YAML config.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return path


def parse_raw_datasets(values: dict) -> list[RawDatasetConfig]:
    dataset_values = values.get("raw_datasets") or values.get("train_datasets")
    if not dataset_values:
        raise ValueError("raw_datasets is required in tokenizer config")

    datasets = []
    for item in dataset_values:
        item = dict(item)
        if item.get("data_path") is not None:
            item["data_path"] = resolve_path(item["data_path"])
        datasets.append(RawDatasetConfig(**item))
    return datasets


def load_tokenizer_config(path: Path) -> TokenizerConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    raw_datasets = parse_raw_datasets(values)
    tokenizer_path = resolve_path(values.get("tokenizer_path"))
    if tokenizer_path is None:
        raise ValueError("tokenizer_path is required in tokenizer config")

    return TokenizerConfig(
        raw_datasets=raw_datasets,
        tokenizer_path=tokenizer_path,
        tokenizer_vocab_size=values.get("tokenizer_vocab_size", 8192),
        tokenizer_train_max_documents=values.get("tokenizer_train_max_documents", 100_000),
        shuffle_buffer_size=values.get("shuffle_buffer_size", 10_000),
        seed=values.get("seed", 42),
    )


def iter_local_text(dataset: RawDatasetConfig):
    assert dataset.data_path is not None, "local dataset must have data_path"
    with open(dataset.data_path, "r", encoding="utf-8") as f:
        yield f.read()


def iter_jsonl_text(dataset: RawDatasetConfig, max_documents: int):
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
    dataset_config: RawDatasetConfig,
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
    dataset_config: RawDatasetConfig,
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
    dataset_config: RawDatasetConfig,
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


def get_text_iterator(config: TokenizerConfig):
    total_weight = sum(dataset.weight for dataset in config.raw_datasets)
    lengths = []

    for dataset in config.raw_datasets:
        if dataset.data_source == "local":
            lengths.append(1)
            continue

        length = round(config.tokenizer_train_max_documents * dataset.weight / total_weight)
        lengths.append(max(1, length))

    def iterator():
        for i, dataset in enumerate(config.raw_datasets):
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
    config = load_tokenizer_config(config_path)

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
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
