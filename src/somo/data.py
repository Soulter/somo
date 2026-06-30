import json
import torch
import random
from pathlib import Path
from datasets import load_dataset

from .tokenizers.tokenizer import BaseTokenizer
from .tokenizers.bpe import BPETokenizer


def read_text(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


class TensorTokenBatcher:
    def __init__(
        self,
        data: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: str,
    ):
        if len(data) <= seq_len + 1:
            raise ValueError(
                f"data is too short for seq_len={seq_len}: got {len(data)} tokens"
            )

        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device

    def next_batch(self):
        ix = torch.randint(
            0,
            len(self.data) - self.seq_len - 1,
            (self.batch_size,),
        )
        x = torch.stack([self.data[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.data[i + 1 : i + self.seq_len + 1] for i in ix])
        return x.to(self.device), y.to(self.device)


class StreamingTokenBatcher:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        batch_size: int,
        seq_len: int,
        device: str,
        dataset_name: str,
        dataset_config: str | None,
        dataset_split: str,
        text_column: str,
        shuffle_buffer_size: int = 10_000,
        seed: int = 42,
        data_files: str | list[str] | None = None,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.text_column = text_column
        self.buffer: list[int] = []

        dataset_kwargs = {}
        if data_files is not None:
            dataset_kwargs["data_files"] = data_files

        self.dataset = load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            streaming=True,
            **dataset_kwargs,
        )
        if shuffle_buffer_size > 0:
            self.dataset = self.dataset.shuffle(
                buffer_size=shuffle_buffer_size,
                seed=seed,
            )
        self.iterator = iter(self.dataset)

    def _fill_buffer(self, min_tokens: int):
        while len(self.buffer) < min_tokens:
            row = next(self.iterator)
            text = row.get(self.text_column)
            if not text:
                continue

            ids = self.tokenizer.encode(text)
            if not ids:
                continue

            self.buffer.extend(ids)

    def next_batch(self):
        needed_tokens = self.batch_size * (self.seq_len + 1)
        self._fill_buffer(needed_tokens)

        chunk = self.buffer[:needed_tokens]
        self.buffer = self.buffer[needed_tokens:]

        data = torch.tensor(chunk, dtype=torch.long)
        data = data.view(self.batch_size, self.seq_len + 1)

        x = data[:, :-1]
        y = data[:, 1:]

        return x.to(self.device), y.to(self.device)


class JsonlTokenBatcher:
    def __init__(
        self,
        tokenizer: BaseTokenizer,
        batch_size: int,
        seq_len: int,
        device: str,
        data_path: str | Path,
        text_column: str,
        line_mod: int | None = None,
        line_remainders: set[int] | None = None,
    ):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.data_path = Path(data_path)
        self.text_column = text_column
        self.line_mod = line_mod
        self.line_remainders = line_remainders
        self.buffer: list[int] = []
        self.iterator = self._iter_text()

        if self.line_mod is not None and self.line_mod <= 0:
            raise ValueError("line_mod must be positive")
        if not self.data_path.exists():
            raise FileNotFoundError(f"jsonl data not found: {self.data_path}")

    def _use_line(self, line_index: int) -> bool:
        if self.line_mod is None or self.line_remainders is None:
            return True

        return line_index % self.line_mod in self.line_remainders

    def _iter_text(self):
        while True:
            with self.data_path.open("r", encoding="utf-8") as f:
                for line_index, line in enumerate(f):
                    if not self._use_line(line_index):
                        continue

                    if not line.strip():
                        continue

                    row = json.loads(line)
                    text = row.get(self.text_column)
                    if text:
                        yield text

    def _fill_buffer(self, min_tokens: int):
        while len(self.buffer) < min_tokens:
            text = next(self.iterator)
            ids = self.tokenizer.encode(text)
            if not ids:
                continue

            self.buffer.extend(ids)

    def next_batch(self):
        needed_tokens = self.batch_size * (self.seq_len + 1)
        self._fill_buffer(needed_tokens)

        chunk = self.buffer[:needed_tokens]
        self.buffer = self.buffer[needed_tokens:]

        data = torch.tensor(chunk, dtype=torch.long)
        data = data.view(self.batch_size, self.seq_len + 1)

        x = data[:, :-1]
        y = data[:, 1:]

        return x.to(self.device), y.to(self.device)

class MixedTokenBatcher:
    def __init__(self, sources: list[dict], seed: int = 42):
        self.names = [source["name"] for source in sources]
        self.weights = [source["weight"] for source in sources]
        self.batchers = [source["batcher"] for source in sources]
        self.rng = random.Random(seed)
        self.counts = {name: 0 for name in self.names}

    def next_batch(self):
        i = self.rng.choices(
            range(len(self.batchers)),
            weights=self.weights,
            k=1,
        )[0]

        name = self.names[i]
        self.counts[name] += 1
        return self.batchers[i].next_batch()
