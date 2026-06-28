import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer

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

    tokenizer.train([str(config.data_path)], trainer)

    config.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(config.tokenizer_path))

    print("saved tokenizer to", config.tokenizer_path)
    print("vocab size:", tokenizer.get_vocab_size())


if __name__ == "__main__":
    args = parse_args()
    train_tokenizer(args.config)
