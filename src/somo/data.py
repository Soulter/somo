import torch
from pathlib import Path
from .tokenizers.tokenizer import BaseTokenizer
from .tokenizers.bpe import BPETokenizer


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_data(text: str, tokenizer: BaseTokenizer):
    ids = tokenizer.encode(text)
    data = torch.tensor(ids, dtype=torch.long)

    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data


def get_batch(data: torch.Tensor, batch_size: int, seq_len: int, device: str):
    # torch.randint(low, high, size)
    # generate batch_size random batches in the data, and return their's start index
    ix = torch.randint(0, len(data) - seq_len - 1, (batch_size,))

    x = torch.stack([data[i : i + seq_len] for i in ix])

    y = torch.stack([data[i + 1 : i + seq_len + 1] for i in ix])

    return x.to(device), y.to(device)  # B, T


if __name__ == "__main__":
    text = read_text("data/tinyshakespeare.txt")
    tokenizer = BPETokenizer(text)
    train_data, val_data = make_data(text, tokenizer)

    x, y = get_batch(
        train_data,
        batch_size=4,
        seq_len=16,
        device="cpu",
    )

    print(x.shape)
    print(y.shape)
    print(x[0])
    print(y[0])
