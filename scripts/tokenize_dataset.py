import argparse
import glob
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from datasets import load_dataset

from somo.tokenizers.bpe import BPETokenizer


DTYPES = {
    "uint16": np.uint16,
    "uint32": np.uint32,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-format",
        choices=["text", "jsonl", "parquet", "hf"],
        required=True,
    )
    parser.add_argument("--input-path", action="append", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size-tokens", type=int, default=100_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_input_paths(values: list[str] | None) -> list[str]:
    paths = []
    for value in values or []:
        matches = sorted(glob.glob(value))
        paths.extend(matches or [value])
    return paths


def iter_text_files(paths: list[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            yield f.read()


def iter_jsonl(paths: list[str], text_column: str):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                row = json.loads(line)
                text = row.get(text_column)
                if isinstance(text, str) and text.strip():
                    yield text


def iter_parquet(paths: list[str], text_column: str):
    dataset = load_dataset(
        "parquet",
        data_files=paths if len(paths) != 1 else paths[0],
        split="train",
        streaming=True,
    )
    for row in dataset:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def iter_hf(
    dataset_name: str,
    dataset_config: str | None,
    dataset_split: str,
    text_column: str,
):
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )
    for row in dataset:
        text = row.get(text_column)
        if isinstance(text, str) and text.strip():
            yield text


def iter_documents(args):
    paths = resolve_input_paths(args.input_path)

    if args.input_format in {"text", "jsonl", "parquet"} and not paths:
        raise ValueError("--input-path is required for text/jsonl/parquet input")

    if args.input_format == "text":
        yield from iter_text_files(paths)
        return

    if args.input_format == "jsonl":
        yield from iter_jsonl(paths, args.text_column)
        return

    if args.input_format == "parquet":
        yield from iter_parquet(paths, args.text_column)
        return

    if args.input_format == "hf":
        if args.dataset_name is None:
            raise ValueError("--dataset-name is required for hf input")
        yield from iter_hf(
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
        )
        return

    raise ValueError(f"unknown input format: {args.input_format}")


class TokenizedShardWriter:
    def __init__(
        self,
        output_dir: Path,
        shard_id: int,
        dtype: str,
        tokenizer_path: Path,
        tokenizer_vocab_size: int,
    ):
        self.output_dir = output_dir
        self.shard_id = shard_id
        self.dtype = dtype
        self.tokenizer_path = tokenizer_path
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.name = f"{shard_id:05d}"
        self.tokens_path = output_dir / f"{self.name}.tokens"
        self.index_path = output_dir / f"{self.name}.index"
        self.meta_path = output_dir / f"{self.name}.meta.json"
        self.token_file = self.tokens_path.open("wb")
        self.offsets = [0]
        self.num_tokens = 0
        self.num_docs = 0

    def add(self, ids: list[int]):
        if not ids:
            return

        array = np.asarray(ids, dtype=DTYPES[self.dtype])
        array.tofile(self.token_file)
        self.num_tokens += len(ids)
        self.num_docs += 1
        self.offsets.append(self.num_tokens)

    def close(self):
        self.token_file.close()

        np.asarray(self.offsets, dtype=np.uint64).tofile(self.index_path)
        metadata = {
            "format": "somo-tokenized-dataset",
            "version": 1,
            "dtype": self.dtype,
            "num_tokens": self.num_tokens,
            "num_docs": self.num_docs,
            "tokenizer_path": str(self.tokenizer_path),
            "tokenizer_vocab_size": self.tokenizer_vocab_size,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {
                "tokens": self.tokens_path.name,
                "index": self.index_path.name,
            },
        }
        self.meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def choose_dtype(tokenizer: BPETokenizer, requested_dtype: str | None) -> str:
    if requested_dtype is not None:
        return requested_dtype
    if tokenizer.vocab_size <= np.iinfo(np.uint16).max:
        return "uint16"
    return "uint32"


def prepare_output_dir(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def tokenize_dataset(args):
    tokenizer = BPETokenizer(args.tokenizer)
    dtype = choose_dtype(tokenizer, args.dtype)
    max_token_id = np.iinfo(DTYPES[dtype]).max
    if tokenizer.vocab_size - 1 > max_token_id:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} does not fit dtype={dtype}"
        )

    prepare_output_dir(args.output, args.overwrite)

    shard_id = 0
    writer = TokenizedShardWriter(
        output_dir=args.output,
        shard_id=shard_id,
        dtype=dtype,
        tokenizer_path=args.tokenizer,
        tokenizer_vocab_size=tokenizer.vocab_size,
    )

    total_docs = 0
    total_tokens = 0
    for text in iter_documents(args):
        ids = tokenizer.encode(text)
        if not ids:
            continue

        if (
            writer.num_docs > 0
            and writer.num_tokens + len(ids) > args.shard_size_tokens
        ):
            writer.close()
            shard_id += 1
            writer = TokenizedShardWriter(
                output_dir=args.output,
                shard_id=shard_id,
                dtype=dtype,
                tokenizer_path=args.tokenizer,
                tokenizer_vocab_size=tokenizer.vocab_size,
            )

        writer.add(ids)
        total_docs += 1
        total_tokens += len(ids)

        if total_docs % 10_000 == 0:
            print(
                f"docs={total_docs:,} tokens={total_tokens:,} shards={shard_id + 1}",
                flush=True,
            )

        if args.max_documents is not None and total_docs >= args.max_documents:
            break

    writer.close()

    dataset_metadata = {
        "format": "somo-tokenized-dataset",
        "version": 1,
        "num_shards": shard_id + 1,
        "num_tokens": total_tokens,
        "num_docs": total_docs,
        "dtype": dtype,
        "tokenizer_path": str(args.tokenizer),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "dataset.meta.json").write_text(
        json.dumps(dataset_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"saved tokenized dataset to {args.output}")
    print(f"docs: {total_docs:,}")
    print(f"tokens: {total_tokens:,}")
    print(f"shards: {shard_id + 1}")
    print(f"dtype: {dtype}")


if __name__ == "__main__":
    tokenize_dataset(parse_args())
