import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import fsspec
import pyarrow.parquet as pq
from datasets import load_dataset


MODELSCOPE = "https://modelscope.cn/datasets"

SOURCES = {
    "ultrafineweb_zh": {
        "kind": "parquet",
        "urls": [
            f"{MODELSCOPE}/OpenBMB/Ultra-FineWeb/resolve/master/data/ultrafineweb_zh/ultrafineweb-zh-part-{part:03d}-of-256.parquet"
            for part in range(1, 12)
        ],
        "text_key": "content",
    },
    "cci4_zh": {
        "kind": "json",
        "urls": [
            f"{MODELSCOPE}/BAAI/CCI4.0-M2-Base-v1/resolve/master/zh_cc-high-loss0/{group:03d}_{part:05d}.jsonl"
            for group in range(2)
            for part in range(8)
        ],
        "known_counts": {
            "000_00000.jsonl": 460_860,
        },
        "text_key": "text",
    },
    "nemotron_cc_math_4plus": {
        "kind": "parquet",
        "urls": [
            f"{MODELSCOPE}/nv-community/Nemotron-CC-Math-v1/resolve/master/4plus/part_{part:06d}.parquet"
            for part in range(46)
        ],
        "text_key": "text",
    },
}


def parse_count(value: str) -> int:
    value = value.strip().lower().replace("_", "")
    multiplier = 1
    if value.endswith("k"):
        multiplier = 1_000
        value = value[:-1]
    elif value.endswith("m"):
        multiplier = 1_000_000
        value = value[:-1]
    return int(float(value) * multiplier)


def format_count(count: int) -> str:
    if count >= 1_000_000 and count % 1_000_000 == 0:
        return f"{count // 1_000_000}m"
    if count >= 1_000 and count % 1_000 == 0:
        return f"{count // 1_000}k"
    return str(count)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def write_jsonl_row(f, row: dict):
    f.write(json.dumps(row, ensure_ascii=False) + "\n")


def output_path(root: Path, name: str, target: int) -> Path:
    return root / "pretrain" / f"{name}_{format_count(target)}.jsonl"


def seed_from_existing(root: Path, name: str, target_path: Path, target: int) -> int:
    if target_path.exists():
        return count_lines(target_path)

    source_dir = root / "pretrain"
    best_path = None
    best_count = 0
    for path in source_dir.glob(f"{name}_*.jsonl"):
        rows = count_lines(path)
        if rows < target and rows > best_count:
            best_path = path
            best_count = rows

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if best_path is not None:
        shutil.copyfile(best_path, target_path)
        print(f"{name}: seeded {best_count} rows from {best_path}", flush=True)
        return best_count

    target_path.touch()
    return 0


def format_pretrain_row(name: str, row: dict, text_key: str) -> dict | None:
    text = row.get(text_key)
    if not isinstance(text, str) or not text.strip():
        return None
    metadata = row.get("metadata")
    return {
        "text": text,
        "dataset": name,
        "source": row.get("source") or (metadata.get("source") if isinstance(metadata, dict) else None),
        "metadata": metadata,
    }


def iter_parquet_rows(name: str, spec: dict, skip: int):
    fs = fsspec.filesystem("https")
    remaining_skip = skip
    columns = [spec["text_key"]]
    if name == "ultrafineweb_zh":
        columns.extend(["source", "score"])

    for url in spec["urls"]:
        print(f"{name}: reading {url}", flush=True)
        with fs.open(url, "rb", block_size=2**20) as f:
            parquet_file = pq.ParquetFile(f)
            for row_group in range(parquet_file.metadata.num_row_groups):
                row_count = parquet_file.metadata.row_group(row_group).num_rows
                if remaining_skip >= row_count:
                    remaining_skip -= row_count
                    continue

                table = parquet_file.read_row_group(row_group, columns=columns)
                rows = table.to_pylist()
                start = remaining_skip
                remaining_skip = 0
                for row in rows[start:]:
                    out = format_pretrain_row(name, row, spec["text_key"])
                    if out is not None:
                        yield out


def iter_jsonl_rows(name: str, spec: dict, skip: int):
    remaining_skip = skip
    for url in spec["urls"]:
        filename = url.rsplit("/", 1)[-1]
        known_count = spec.get("known_counts", {}).get(filename)
        if known_count is not None and remaining_skip >= known_count:
            remaining_skip -= known_count
            print(f"{name}: skipped {filename} ({known_count} known rows)", flush=True)
            continue

        print(f"{name}: reading {url}", flush=True)
        req = Request(url, headers={"User-Agent": "somo-dataset-sampler/0.1"})
        with urlopen(req, timeout=60) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                row = json.loads(raw_line.decode("utf-8"))
                out = format_pretrain_row(name, row, spec["text_key"])
                if out is None:
                    continue
                if remaining_skip > 0:
                    remaining_skip -= 1
                    continue
                yield out


def iter_hf_rows(name: str, spec: dict, skip: int):
    token = os.environ.get("HF_TOKEN")
    dataset = load_dataset(
        spec["dataset_name"],
        spec.get("dataset_config"),
        split=spec.get("dataset_split", "train"),
        streaming=True,
        token=token,
    )

    remaining_skip = skip
    for row in dataset:
        out = format_pretrain_row(name, row, spec["text_key"])
        if out is None:
            continue
        if remaining_skip > 0:
            remaining_skip -= 1
            continue
        yield out


def iter_source_rows(name: str, spec: dict, skip: int):
    if spec["kind"] == "parquet":
        yield from iter_parquet_rows(name, spec, skip)
    elif spec["kind"] == "json":
        yield from iter_jsonl_rows(name, spec, skip)
    elif spec["kind"] == "hf":
        yield from iter_hf_rows(name, spec, skip)
    else:
        raise ValueError(f"unsupported source kind: {spec['kind']}")


def extend_source(root: Path, name: str, target: int) -> tuple[Path, int]:
    spec = SOURCES[name]
    path = output_path(root, name, target)
    existing = seed_from_existing(root, name, path, target)
    if existing >= target:
        print(f"{name}: already has {existing} rows at {path}", flush=True)
        return path, existing

    print(f"{name}: extending {existing} -> {target} rows at {path}", flush=True)
    seen = 0
    written = existing
    progress_interval = max(10_000, target // 20)

    with path.open("a", encoding="utf-8") as f:
        for row in iter_source_rows(name, spec, existing):
            seen += 1
            write_jsonl_row(f, row)
            written += 1
            if written % progress_interval == 0:
                print(f"{name}: {written} rows...", flush=True)
            if written >= target:
                break

    print(f"{name}: wrote {written} rows to {path}", flush=True)
    return path, written


def rebuild_mixed_pretrain(root: Path, source_paths: list[Path]) -> tuple[Path, int]:
    mixed_path = root / "pretrain" / "mixed_pretrain_sample.jsonl"
    total = 0
    with mixed_path.open("w", encoding="utf-8") as out:
        for path in source_paths:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    total += 1
    print(f"mixed_pretrain: wrote {total} rows to {mixed_path}", flush=True)
    return mixed_path, total


def update_summary(root: Path, pretrain_summary: dict):
    summary_path = root / "sample_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}

    summary["pretrain"] = pretrain_summary
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary: {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/samples"))
    parser.add_argument("--target", type=parse_count, default=parse_count("1m"))
    parser.add_argument("--sources", nargs="+", default=["ultrafineweb_zh", "cci4_zh"])
    parser.add_argument("--no-mix", action="store_true", help="Do not rebuild mixed_pretrain_sample.jsonl.")
    args = parser.parse_args()

    source_paths = []
    pretrain_summary = {}
    for name in args.sources:
        if name not in SOURCES:
            raise ValueError(f"unknown source: {name}")
        path, rows = extend_source(args.output_dir, name, args.target)
        source_paths.append(path)
        pretrain_summary[name] = {
            "path": str(path),
            "rows": rows,
            "target": args.target,
        }

    if not args.no_mix:
        mixed_path, mixed_rows = rebuild_mixed_pretrain(args.output_dir, source_paths)
        pretrain_summary["mixed_pretrain"] = {
            "path": str(mixed_path),
            "rows": mixed_rows,
        }
    update_summary(args.output_dir, pretrain_summary)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
