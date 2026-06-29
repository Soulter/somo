import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from .data import (
    JsonlTokenBatcher,
    StreamingTokenBatcher,
    get_batch,
    make_data,
    read_text,
)
from .tokenizers.bpe import BPETokenizer
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
    min_lr: float = 3e-5
    warmup_steps: int = 100
    seed: int = 42
    grad_clip: float = 1.0
    data_source: str = "local"
    data_path: Path = Path("data/tinyshakespeare.txt")
    dataset_name: str | None = None
    dataset_config: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    shuffle_buffer_size: int = 10_000
    tokenizer_path: Path = Path("tokenizers/tiny-bpe.json")
    tokenizer_vocab_size: int = 2048
    tokenizer_train_max_documents: int = 100_000
    checkpoint_path: Path = Path("checkpoints/tiny.pt")
    resume_path: Path | None = None
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    dropout: float = 0.0
    grad_accum_steps: int = 1 # micro batch
    log_dir: Path | None = None
    precision: str = "bf16"

def get_autocast_context(device: str, precision: str):
    if device == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if device == "cuda" and precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()

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
    values["tokenizer_path"] = resolve_path(
        values.get("tokenizer_path", Config.tokenizer_path)
    )
    values["checkpoint_path"] = resolve_path(
        values.get("checkpoint_path", Config.checkpoint_path)
    )
    values["resume_path"] = resolve_path(values.get("resume_path"))

    log_dir = f"{str(values.get('log_dir')).rstrip('/')}/{datetime.now().strftime('%Y%m%d_%H%M%S')}" if values.get("log_dir") else None
    values["log_dir"] = resolve_path(log_dir)

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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# dynamic lr
def get_lr(step: int, config: Config) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps

    decay_steps = config.max_steps - config.warmup_steps
    if decay_steps <= 0:
        return config.min_lr

    decay_step = min(step - config.warmup_steps, decay_steps)
    decay_ratio = decay_step / decay_steps
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


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
    set_seed(config.seed)
    device = get_device()
    print(f"we will using {device}.")
    tokenizer = BPETokenizer(config.tokenizer_path)

    print("vocab_size:", tokenizer.vocab_size)

    is_streaming = config.data_source in {"hf", "jsonl", "parquet"}

    if config.data_source == "local":
        text = read_text(config.data_path)
        train_data, val_data = make_data(text, tokenizer)
        train_batcher = None
        print("train tokens:", len(train_data))
        print("val tokens:", len(val_data))
    elif config.data_source == "hf":
        if config.dataset_name is None:
            raise ValueError("dataset_name is required when data_source is 'hf'")
        train_data = None
        val_data = None
        train_batcher = StreamingTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            dataset_split=config.dataset_split,
            text_column=config.text_column,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
        )
        eval_batcher = StreamingTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            dataset_split=config.dataset_split,
            text_column=config.text_column,
            shuffle_buffer_size=max(1, config.shuffle_buffer_size // 10),
            seed=config.seed + 1,
        )
        print(
            "hf dataset:",
            config.dataset_name,
            config.dataset_config,
            config.dataset_split,
        )
    elif config.data_source == "jsonl":
        train_data = None
        val_data = None
        train_batcher = JsonlTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            data_path=config.data_path,
            text_column=config.text_column,
            line_mod=10,
            line_remainders=set(range(1, 10)),
        )
        eval_batcher = JsonlTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            data_path=config.data_path,
            text_column=config.text_column,
            line_mod=10,
            line_remainders={0},
        )
        print("jsonl dataset:", config.data_path)
    elif config.data_source == "parquet":
        train_data = None
        val_data = None
        data_files = str(config.data_path)
        train_batcher = StreamingTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            dataset_name="parquet",
            dataset_config=None,
            dataset_split="train",
            text_column=config.text_column,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
            data_files=data_files,
        )
        eval_batcher = StreamingTokenBatcher(
            tokenizer=tokenizer,
            batch_size=config.batch_size,
            seq_len=config.seq_len,
            device=device,
            dataset_name="parquet",
            dataset_config=None,
            dataset_split="train",
            text_column=config.text_column,
            shuffle_buffer_size=max(1, config.shuffle_buffer_size // 10),
            seed=config.seed + 1,
            data_files=data_files,
        )
        print("parquet dataset:", data_files)
    else:
        raise ValueError(f"unknown data_source: {config.data_source}")

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

        if is_streaming:
            losses = []
            for _ in range(config.eval_iters):
                x, y = eval_batcher.next_batch()
                with get_autocast_context(device, config.precision):
                    _, loss = model(x, y)
                losses.append(loss.item())

            mean_loss = sum(losses) / len(losses)
            model.train()
            return {"train": mean_loss, "val": mean_loss}

        assert train_data is not None and val_data is not None
        out = {}
        for split, data in [("train", train_data), ("val", val_data)]:
            losses = []
            for _ in range(config.eval_iters):
                x, y = get_batch(data, config.batch_size, config.seq_len, device)
                with get_autocast_context(device, config.precision):
                    _, loss = model(x, y)
                losses.append(loss.item())

            out[split] = sum(losses) / len(losses)

        model.train()
        return out

    # prepare tensorboard writer
    writer = None
    if config.log_dir is not None:
        writer = SummaryWriter(config.log_dir)

    for step in range(start_step, config.max_steps):
        # dynamic lr
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # output logs
        if step % config.eval_interval == 0:
            losses = estimate_loss()
            print(
                f"step {step}: "
                f"train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, "
                f"lr {lr:.2e}"
            )
            if writer is not None:
                writer.add_scalar("loss/train", losses["train"], step)
                writer.add_scalar("loss/val", losses["val"], step)
                writer.add_scalar("lr", lr, step)   

        optimizer.zero_grad(set_to_none=True)  # clear grad

        loss_accum = 0.0
        for micro_step in range(config.grad_accum_steps):
            if is_streaming and train_batcher:
                x, y = train_batcher.next_batch()
            else:
                assert train_data is not None
                x, y = get_batch(train_data, config.batch_size, config.seq_len, device)

            with get_autocast_context(device, config.precision):
                logits, loss = model(x, y)

            raw_loss = loss.item()
            loss_accum += raw_loss
            if writer is not None:
                # note: step is started from 0, so we don't need (step - 1) * config.grad_accum_steps
                writer.add_scalar("loss/step_micro", raw_loss, step * config.grad_accum_steps + micro_step)
            # here, finally the loss is: (grad(loss_1) + grad(loss_2) + ... + grad(loss_N)) / N
            # because pytorch accumulates the grad but not cover.
            loss = loss / config.grad_accum_steps
            loss.backward()

        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if writer is not None:
            mean_loss = loss_accum / config.grad_accum_steps
            writer.add_scalar("loss/step", mean_loss, step)

            tokens_per_step = config.batch_size * config.seq_len * config.grad_accum_steps
            seen_tokens = (step + 1) * tokens_per_step
            writer.add_scalar("tokens/seen", seen_tokens, step)
            writer.add_scalar("loss/step_by_tokens", mean_loss, seen_tokens)

            if torch.cuda.is_available():
                writer.add_scalar(
                    "gpu/memory_allocated_mb",
                    torch.cuda.memory_allocated() / 1024**2,
                    step,
                )
                writer.add_scalar(
                    "gpu/memory_reserved_mb",
                    torch.cuda.memory_reserved() / 1024**2,
                    step,
                )
                writer.add_scalar(
                    "gpu/max_memory_allocated_mb",
                    torch.cuda.max_memory_allocated() / 1024**2,
                    step,
                )

        if (
            config.checkpoint_interval > 0
            and (step + 1) % config.checkpoint_interval == 0
        ):
            save_checkpoint(config, model, optimizer, step + 1, gpt_config)

    if writer is not None:
        writer.close()

    if (
        config.checkpoint_interval <= 0
        or config.max_steps % config.checkpoint_interval != 0
    ):
        save_checkpoint(config, model, optimizer, config.max_steps, gpt_config)


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config)
