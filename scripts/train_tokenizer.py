import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from datasets import load_dataset
from somo.train import Config

from somo.train import load_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config.",
    )
    return parser.parse_args()


def iter_local_text(config):
    with open(config.data_path, "r", encoding="utf-8") as f:
        yield f.read()


def iter_jsonl_text(config: Config):
    count = 0
    with open(config.data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            text = row.get(config.text_column)
            if not text:
                continue

            yield text
            count += 1
            if count >= config.tokenizer_train_max_documents:
                break


def iter_hf_text(config: Config):
    if config.dataset_name is None:
        raise ValueError("dataset_name is required when data_source is 'hf'")

    dataset = load_dataset(
        config.dataset_name,
        config.dataset_config,
        split=config.dataset_split,
        streaming=True,
    )
    if config.shuffle_buffer_size > 0:
        dataset = dataset.shuffle(
            buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
        )

    count = 0
    for row in dataset:
        text = row.get(config.text_column)
        if not text:
            continue

        yield text
        count += 1
        if count >= config.tokenizer_train_max_documents:
            break


def get_text_iterator(config: Config):
    if config.data_source == "local":
        return iter_local_text(config), 1
    if config.data_source == "jsonl":
        return iter_jsonl_text(config), config.tokenizer_train_max_documents
    if config.data_source == "hf":
        return iter_hf_text(config), config.tokenizer_train_max_documents

    raise ValueError(f"unknown data_source: {config.data_source}")


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
