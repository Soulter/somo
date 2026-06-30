import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


@dataclass
class TokenizedShard:
    tokens_path: Path
    index_path: Path
    meta_path: Path
    dtype: str
    num_tokens: int
    num_docs: int

    @classmethod
    def from_meta(cls, meta_path: Path):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        base = meta_path.parent
        files = meta.get("files", {})

        return cls(
            tokens_path=base / files.get("tokens", meta_path.name.replace(".meta.json", ".tokens")),
            index_path=base / files.get("index", meta_path.name.replace(".meta.json", ".index")),
            meta_path=meta_path,
            dtype=meta["dtype"],
            num_tokens=int(meta["num_tokens"]),
            num_docs=int(meta["num_docs"]),
        )

    def open_tokens(self):
        if self.dtype not in DTYPES:
            raise ValueError(f"unsupported token dtype: {self.dtype}")
        return np.memmap(
            self.tokens_path,
            dtype=DTYPES[self.dtype],
            mode="r",
            shape=(self.num_tokens,),
        )


class TokenizedDataset:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"tokenized dataset not found: {self.path}")

        self.shards = [
            TokenizedShard.from_meta(path)
            for path in sorted(self.path.glob("*.meta.json"))
            if path.name != "dataset.meta.json"
        ]
        if not self.shards:
            raise FileNotFoundError(f"no *.meta.json shards found in {self.path}")

        self.total_tokens = sum(shard.num_tokens for shard in self.shards)
        self.total_docs = sum(shard.num_docs for shard in self.shards)
        self._tokens: dict[int, np.memmap] = {}

    def _tokens_for_shard(self, shard_index: int):
        if shard_index not in self._tokens:
            self._tokens[shard_index] = self.shards[shard_index].open_tokens()
        return self._tokens[shard_index]

    def sample(self, seq_len: int, rng: random.Random):
        usable_shards = [
            i for i, shard in enumerate(self.shards) if shard.num_tokens > seq_len + 1
        ]
        if not usable_shards:
            raise ValueError(
                f"{self.path} has no shard long enough for seq_len={seq_len}"
            )

        weights = [self.shards[i].num_tokens for i in usable_shards]
        shard_index = rng.choices(usable_shards, weights=weights, k=1)[0]
        shard = self.shards[shard_index]
        tokens = self._tokens_for_shard(shard_index)

        start = rng.randrange(0, shard.num_tokens - seq_len - 1)
        window = np.asarray(tokens[start : start + seq_len + 1], dtype=np.int64)

        x = torch.from_numpy(window[:-1])
        y = torch.from_numpy(window[1:])
        return x, y


class TokenizedMixtureDataset(IterableDataset):
    def __init__(
        self,
        sources: list[dict],
        seq_len: int,
        seed: int,
    ):
        self.names = [source["name"] for source in sources]
        self.weights = [source["weight"] for source in sources]
        self.datasets = [source["dataset"] for source in sources]
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + worker_id)

        while True:
            index = rng.choices(
                range(len(self.datasets)),
                weights=self.weights,
                k=1,
            )[0]
            yield self.datasets[index].sample(self.seq_len, rng)


def make_tokenized_dataloader(
    sources: list[dict],
    batch_size: int,
    seq_len: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    dataset = TokenizedMixtureDataset(
        sources=sources,
        seq_len=seq_len,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
