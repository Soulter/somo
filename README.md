# Somo

A small language model from scratch.

## Initialize

```bash
uv sync
```

## Run

```bash
uv run python -m somo.train --config configs/train_tiny.yaml
uv run python -m somo.generate --config configs/train_tiny.yaml
```