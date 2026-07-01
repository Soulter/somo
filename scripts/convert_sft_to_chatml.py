import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


ROLE_MAP = {
    "human": "user",
    "user": "user",
    "prompter": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "model": "assistant",
    "bot": "assistant",
    "system": "system",
}

RESPONSE_KEYS = ("response", "output", "answer", "completion", "chosen")
PROMPT_KEYS = ("prompt", "instruction", "question", "query")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert common SFT schemas to ChatML JSONL."
    )
    parser.add_argument(
        "--input-format",
        choices=["jsonl", "parquet", "hf"],
        required=True,
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        action="append",
        default=None,
        help="Input file path/glob. Can be repeated. For globs, quoting is recommended.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/sft_chatml/chatml.jsonl"),
    )
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def flatten(values: list[list[str]] | None) -> list[str]:
    items = []
    for group in values or []:
        items.extend(group)
    return items


def resolve_input_paths(values: list[list[str]] | None) -> list[str]:
    paths = []
    for value in flatten(values):
        matches = sorted(glob.glob(value))
        paths.extend(matches or [value])
    return paths


def text_from_content(content: Any) -> str | None:
    if isinstance(content, str):
        text = content.strip()
        return text or None

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text") or item.get("content")
                if isinstance(value, str):
                    parts.append(value)
        text = "\n".join(part.strip() for part in parts if part and part.strip())
        return text or None

    return None


def normalize_role(role: Any) -> str | None:
    if not isinstance(role, str):
        return None
    return ROLE_MAP.get(role.strip().lower())


def normalize_messages(raw_messages: Any) -> list[dict[str, str]] | None:
    if not isinstance(raw_messages, list):
        return None

    messages = []
    for turn in raw_messages:
        if not isinstance(turn, dict):
            return None

        role = normalize_role(turn.get("role") or turn.get("from"))
        content = text_from_content(turn.get("content") or turn.get("value"))
        if role is None or content is None:
            return None
        messages.append({"role": role, "content": content})

    if not messages or not any(message["role"] == "assistant" for message in messages):
        return None
    return messages


def first_string(row: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def prompt_response_to_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    prompt = first_string(row, PROMPT_KEYS)
    response = first_string(row, RESPONSE_KEYS)
    if prompt is None or response is None:
        return None

    extra_input = row.get("input")
    if isinstance(extra_input, str) and extra_input.strip():
        prompt = f"{prompt}\n\n{extra_input.strip()}"

    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]


def extract_messages(row: dict[str, Any]) -> list[dict[str, str]] | None:
    for key in ("messages", "conversations", "conversation"):
        messages = normalize_messages(row.get(key))
        if messages is not None:
            return messages
    return prompt_response_to_messages(row)


def render_chatml(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = message["role"]
        content = message["content"].strip()
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(parts)


def iter_jsonl(paths: list[str]):
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def iter_parquet(paths: list[str]):
    dataset = load_dataset(
        "parquet",
        data_files=paths if len(paths) != 1 else paths[0],
        split="train",
        streaming=True,
    )
    yield from dataset


def iter_hf(dataset_name: str, dataset_config: str | None, dataset_split: str):
    dataset = load_dataset(
        dataset_name,
        dataset_config,
        split=dataset_split,
        streaming=True,
    )
    yield from dataset


def iter_rows(args):
    paths = resolve_input_paths(args.input_path)

    if args.input_format in {"jsonl", "parquet"} and not paths:
        raise ValueError("--input-path is required for jsonl/parquet input")

    if args.input_format == "jsonl":
        yield from iter_jsonl(paths)
        return

    if args.input_format == "parquet":
        yield from iter_parquet(paths)
        return

    if args.input_format == "hf":
        if args.dataset_name is None:
            raise ValueError("--dataset-name is required for hf input")
        yield from iter_hf(args.dataset_name, args.dataset_config, args.dataset_split)
        return

    raise ValueError(f"unknown input format: {args.input_format}")


def prepare_output(path: Path, overwrite: bool):
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def convert(args):
    prepare_output(args.output, args.overwrite)

    total = 0
    written = 0
    skipped = 0
    source_name = args.source_name or args.dataset_name

    with args.output.open("w", encoding="utf-8") as f:
        for row in tqdm(iter_rows(args), desc="convert", unit="rows"):
            total += 1
            messages = extract_messages(row)
            if messages is None:
                skipped += 1
                continue

            out = {
                "text": render_chatml(messages),
                "messages": messages,
            }
            if source_name is not None:
                out["dataset"] = source_name
            elif isinstance(row.get("dataset"), str):
                out["dataset"] = row["dataset"]

            for key in ("source", "prompt_id", "category"):
                value = row.get(key)
                if value is not None:
                    out[key] = value

            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            written += 1

            if args.max_examples is not None and written >= args.max_examples:
                break

    summary = {
        "format": "somo-chatml-jsonl",
        "version": 1,
        "output": str(args.output),
        "total_rows_seen": total,
        "rows_written": written,
        "rows_skipped": skipped,
    }
    summary_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        f"wrote {written:,} rows to {args.output} "
        f"(skipped {skipped:,}, saw {total:,})",
        flush=True,
    )
    print(f"summary: {summary_path}", flush=True)


def main():
    convert(parse_args())


if __name__ == "__main__":
    main()
