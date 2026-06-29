# Somo

A small language model from scratch.

## Initialize

```bash
uv sync
```

## Run

### tiny shakespeare

```bash
uv run python scripts/train_tokenizer.py --config configs/train_tiny.yaml
uv run python -m somo.train --config configs/train_tiny.yaml
uv run python -m somo.generate --config configs/train_tiny.yaml
```

### fineweb-edu 10bt

~0.1B model size

```bash
uv run python scripts/train_tokenizer.py --config configs/train_fineweb_10bt_100m.yaml
uv run python -m somo.train --config configs/train_fineweb_10bt_100m.yaml
uv run python -m somo.generate --config configs/train_fineweb_10bt_100m.yaml
# new screen
uv run tensorboard --logdir runs
# use ssh tunnel if you need
ssh -L 6006:localhost:6006 user@server
```
