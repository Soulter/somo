# Somo

A small language model from scratch.

## Initialize

```bash
uv sync
```

## Environment

Server: A100 * 8 (currently I just using 1 GPU)

`configs/train_fineweb_10bt_100m.yaml` is adjust to A100 * 1

## Run

### tiny shakespeare

```bash
uv run python scripts/train_tokenizer.py --config configs/tokenizer_tiny.yaml
uv run python scripts/tokenize_dataset.py \
  --input-format text \
  --input-path data/tinyshakespeare.txt \
  --tokenizer tokenizers/tiny-bpe.json \
  --output datasets/tinyshakespeare \
  --overwrite
uv run python -m somo.train --config configs/train_tiny.yaml
uv run python -m somo.generate --config configs/train_tiny.yaml
```

### fineweb-edu 10bt

~0.1B model size

```bash
uv run python scripts/train_tokenizer.py --config configs/tokenizer_fineweb_8k.yaml
uv run python scripts/tokenize_dataset.py \
  --input-format parquet \
  --input-path 'data/fineweb-edu-modelscope/sample/10BT/train/*.parquet' \
  --text-column text \
  --tokenizer tokenizers/fineweb-8k-bpe.json \
  --output datasets/fineweb_edu_10bt/train \
  --overwrite
uv run python scripts/tokenize_dataset.py \
  --input-format parquet \
  --input-path 'data/fineweb-edu-modelscope/sample/10BT/val/*.parquet' \
  --text-column text \
  --tokenizer tokenizers/fineweb-8k-bpe.json \
  --output datasets/fineweb_edu_10bt/val \
  --overwrite
uv run python -m somo.train --config configs/train_fineweb_10bt_100m.yaml
uv run python -m somo.generate --config configs/train_fineweb_10bt_100m.yaml
# new screen
uv run tensorboard --logdir runs
# use ssh tunnel if you need
ssh -L 6006:localhost:6006 user@server
```

#### Performance Metrics

- Using hand-writing `scaled_dot_product_attention`: 3.127min ~ 20steps
- Using pytoch-implementation `scaled_dot_product_attention`: 1.227min ~ 20steps

## Test Reminder

### Train

- DataLoader
- Weight Tying
- DDP

### Pre-train

- Pretrain Dataset Recipe(Fineweb-edu; CCI4; Nemotron-CC-Math-v1; )
- Pretrain Annealing (Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations; Scaling Law with Learning Rate Annealing; MiniCPM; TinyLlama; Mid-Training of Large Language Models: A Survey)
- GQA
- RoPE

### Post-train

#### SFT

- HuggingFaceTB/smol-smoltalk
- HuggingFaceH4/no_robots
- HuggingFaceH4/ultrachat_200k
- liumindmind/NekoQA-10K

## Thanks

Partially inspired from [nanotron](https://github.com/huggingface/nanotron).
