from pathlib import Path

from tokenizers import Tokenizer
from .tokenizer import BaseTokenizer


class BPETokenizer(BaseTokenizer):
    def __init__(self, path: str | Path):
        """path: tokenizer path"""
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)
