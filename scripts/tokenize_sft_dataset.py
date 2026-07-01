import argparse
import json
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path
from somo.tokenizers.tokenizer import BaseTokenizer
from somo.tokenizers.bpe import BPETokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-size-tokens", type=int, default=50_000_000)
    parser.add_argument("--max-documents", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def encode_messages_for_sft(messages: list[dict], tokenizer: BaseTokenizer):
    input_ids = []
    labels = []

    for message in messages:
        role = message["role"]
        content = message["content"].strip()

        prefix = f"<|im_start|>{role}\n"
        suffix = "<|im_end|>\n"

        prefix_ids = tokenizer.encode(prefix)
        content_ids = tokenizer.encode(content)
        suffix_ids = tokenizer.encode(suffix)

        ids = prefix_ids + content_ids + suffix_ids
        input_ids.extend(ids)

        if role == "assistant":
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids)
            labels.extend(suffix_ids)
        else:
            labels.extend([-100] * len(ids))

    assert len(input_ids) == len(labels)
    return input_ids, labels

class SFTShardWriter:
    def __init__(self, output_dir, shard_id, token_dtype, tokenizer_path, vocab_size):
        self.output_dir = output_dir
        self.shard_id = shard_id
        self.token_dtype = token_dtype
        self.name = f"{shard_id:05d}"

        self.tokens_path = output_dir / f"{self.name}.tokens"
        self.labels_path = output_dir / f"{self.name}.labels"
        self.index_path = output_dir / f"{self.name}.index"
        self.meta_path = output_dir / f"{self.name}.meta.json"

        self.token_file = self.tokens_path.open("wb")
        self.label_file = self.labels_path.open("wb")

        self.offsets = [0]
        self.num_tokens = 0
        self.num_docs = 0
        self.tokenizer_path = tokenizer_path
        self.vocab_size = vocab_size

    def add(self, input_ids, labels):
        assert len(input_ids) == len(labels)

        np.asarray(input_ids, dtype=self.token_dtype).tofile(self.token_file)
        np.asarray(labels, dtype=np.int32).tofile(self.label_file)

        self.num_tokens += len(input_ids)
        self.num_docs += 1
        self.offsets.append(self.num_tokens)

    def close(self):
        self.token_file.close()
        self.label_file.close()

        np.asarray(self.offsets, dtype=np.uint64).tofile(self.index_path)

        meta = {
            "format": "somo-sft-tokenized-dataset",
            "version": 1,
            "dtype": str(np.dtype(self.token_dtype)),
            "label_dtype": "int32",
            "num_tokens": self.num_tokens,
            "num_docs": self.num_docs,
            "tokenizer_path": str(self.tokenizer_path),
            "tokenizer_vocab_size": self.vocab_size,
            "files": {
                "tokens": self.tokens_path.name,
                "labels": self.labels_path.name,
                "index": self.index_path.name,
            },
        }

        self.meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

def tokenize_sft_dataset(args):
    tokenizer = BPETokenizer(args.tokenizer)

    token_dtype = np.uint16
    if tokenizer.vocab_size > np.iinfo(np.uint16).max:
        token_dtype = np.uint32

    if args.output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {args.output}")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    shard_id = 0
    writer = SFTShardWriter(
        args.output,
        shard_id,
        token_dtype,
        args.tokenizer,
        tokenizer.vocab_size,
    )

    total_docs = 0
    total_tokens = 0
    supervised_tokens = 0

    with args.input.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="tokenize sft"):
            if not line.strip():
                continue

            row = json.loads(line)
            messages = row["messages"]

            input_ids, labels = encode_messages_for_sft(messages, tokenizer)
            if not input_ids:
                continue

            if writer.num_docs > 0 and writer.num_tokens + len(input_ids) > args.shard_size_tokens:
                writer.close()
                shard_id += 1
                writer = SFTShardWriter(
                    args.output,
                    shard_id,
                    token_dtype,
                    args.tokenizer,
                    tokenizer.vocab_size,
                )

            writer.add(input_ids, labels)

            total_docs += 1
            total_tokens += len(input_ids)
            supervised_tokens += sum(label != -100 for label in labels)

            if args.max_documents is not None and total_docs >= args.max_documents:
                break

    writer.close()

    meta = {
        "format": "somo-sft-tokenized-dataset",
        "version": 1,
        "num_docs": total_docs,
        "num_tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
    }

    (args.output / "dataset.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def main():
    tokenize_sft_dataset(parse_args())


if __name__ == "__main__":
    main()