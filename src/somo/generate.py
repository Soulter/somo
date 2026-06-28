import argparse
from pathlib import Path
from dataclasses import dataclass

import torch

from .data import CharTokenizer, read_text
from .model import GPT, GPTConfig
from .train import get_device, load_config


@dataclass
class GenerateConfig:
    train_config_path: Path
    prompt: str = "ROMEO:"
    max_new_tokens: int = 500
    temperature: float = 0.8


def load_model(
    checkpoint_path: Path,
    tokenizer: CharTokenizer,
    device: str,
) -> GPT:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = checkpoint.get("model_config")
    if model_config is None:
        raise KeyError("checkpoint is missing model_config")

    gpt_config = GPTConfig(**model_config)
    if gpt_config.vocab_size != tokenizer.vocab_size:
        raise ValueError(
            "checkpoint vocab_size does not match tokenizer vocab_size: "
            f"{gpt_config.vocab_size} != {tokenizer.vocab_size}"
        )

    model = GPT(gpt_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main(config: GenerateConfig):
    device = get_device()
    train_config = load_config(config.train_config_path)

    text = read_text(train_config.data_path)
    tokenizer = CharTokenizer(text)
    model = load_model(train_config.checkpoint_path, tokenizer, device)

    prompt_ids = tokenizer.encode(config.prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    output_ids = model.generate(
        idx,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
    )[0].tolist()

    print(tokenizer.decode(output_ids))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(GenerateConfig(train_config_path=args.config))
