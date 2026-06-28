import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import yaml

from .data import CharTokenizer, get_batch, make_data, read_text
from .model import GPT, GPTConfig


@dataclass
class Config:
    batch_size: int = 32
    seq_len: int = 128
    max_steps: int = 1000
    eval_interval: int = 100
    eval_iters: int = 20
    checkpoint_interval: int = 500
    learning_rate: float = 3e-4
    data_path: Path = Path("data/tinyshakespeare.txt")
    checkpoint_path: Path = Path("checkpoints/tiny.pt")
    resume_path: Path | None = None
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.0


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None

    path = Path(value)
    if path.is_absolute():
        return path

    return path


def load_config(path: str | Path) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        values = yaml.safe_load(f) or {}

    values["data_path"] = resolve_path(values.get("data_path", Config.data_path))
    values["checkpoint_path"] = resolve_path(
        values.get("checkpoint_path", Config.checkpoint_path)
    )
    values["resume_path"] = resolve_path(values.get("resume_path"))

    return Config(**values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config.",
    )
    return parser.parse_args()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "mps" if torch.backends.mps.is_available() else device
    return device


def config_to_dict(config):
    values = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


def save_checkpoint(
    config: Config,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    gpt_config: GPTConfig,
):
    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": config_to_dict(config),
            "model_config": asdict(gpt_config),
        },
        config.checkpoint_path,
    )
    print(f"saved checkpoint to {config.checkpoint_path}")


def load_checkpoint(
    path: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> int:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    step = checkpoint["step"]
    print(f"loaded checkpoint from {path} at step {step}")
    return step


def train(config: Config):
    # prepare data
    device = get_device()
    text = read_text(config.data_path)
    tokenizer = CharTokenizer(text)
    train_data, val_data = make_data(text, tokenizer)

    print("vocab_size:", tokenizer.vocab_size)
    print("train tokens:", len(train_data))
    print("val tokens:", len(val_data))

    # prepare model
    gpt_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        seq_len=config.seq_len,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        d_model=config.d_model,
        dropout=config.dropout,
    )
    model = GPT(gpt_config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"params: {num_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
    )

    start_step = 0
    if config.resume_path is not None:
        start_step = load_checkpoint(config.resume_path, model, optimizer, device)

    @torch.no_grad()
    def estimate_loss():
        model.eval()

        out = {}
        for split, data in [("train", train_data), ("val", val_data)]:
            losses = []
            for _ in range(config.eval_iters):
                x, y = get_batch(data, config.batch_size, config.seq_len, device)
                _, loss = model(x, y)
                losses.append(loss.item())

            out[split] = sum(losses) / len(losses)

        model.train()
        return out

    for step in range(start_step, config.max_steps):
        if step % config.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )

        x, y = get_batch(train_data, config.batch_size, config.seq_len, device)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)  # clear grad
        loss.backward()
        optimizer.step()

        if (
            config.checkpoint_interval > 0
            and (step + 1) % config.checkpoint_interval == 0
        ):
            save_checkpoint(config, model, optimizer, step + 1, gpt_config)

    if (
        config.checkpoint_interval <= 0
        or config.max_steps % config.checkpoint_interval != 0
    ):
        save_checkpoint(config, model, optimizer, config.max_steps, gpt_config)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
