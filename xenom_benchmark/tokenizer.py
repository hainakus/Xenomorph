"""Tokenizer loading for token-level MLM benchmarking.

The benchmark needs the exact same BPE/k-mer tokenizer used by the model so that
masking, inference and metric computation all share the same token space.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Union

import requests

from .utils import logger, repo_root


class BenchmarkTokenizer:
    """Thin wrapper around a Hugging Face fast tokenizer loaded from JSON."""

    def __init__(self, tokenizer_path: Union[str, Path]) -> None:
        from tokenizers import Tokenizer

        self.path = Path(tokenizer_path)
        self._tokenizer = Tokenizer.from_file(str(self.path))
        self.vocab_size = self._tokenizer.get_vocab_size()

    @classmethod
    def from_huggingface(cls, model_id: str, cache_dir: Optional[Path] = None) -> "BenchmarkTokenizer":
        """Download ``tokenizer.json`` from a Hugging Face hub and load it."""
        if cache_dir is None:
            cache_dir = repo_root() / ".benchmark_cache" / "tokenizers"
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        safe_name = model_id.replace("/", "--")
        local_path = cache_dir / f"{safe_name}_tokenizer.json"

        if local_path.exists():
            logger.info("Using cached tokenizer for %s: %s", model_id, local_path)
            return cls(local_path)

        url = f"https://huggingface.co/{model_id}/resolve/main/tokenizer.json"
        logger.info("Downloading tokenizer for %s from %s", model_id, url)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        local_path.write_bytes(response.content)
        return cls(local_path)

    @classmethod
    def from_model_dir(cls, model_dir: Union[str, Path]) -> "BenchmarkTokenizer":
        """Load a local ``tokenizer.json`` from a model directory."""
        model_dir = Path(model_dir)
        tokenizer_path = model_dir / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"No tokenizer.json found in {model_dir}")
        return cls(tokenizer_path)

    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """Tokenize a sequence and return token ids."""
        return self._tokenizer.encode(sequence, add_special_tokens=add_special_tokens).ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token ids to a string without adding spaces between tokens."""
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def token_strings(self) -> List[str]:
        """Return the token string for every id in the vocabulary."""
        vocab = self._tokenizer.get_vocab()
        # vocab is {token: id}; invert it.
        id_to_token = [""] * self.vocab_size
        for token, idx in vocab.items():
            id_to_token[idx] = token
        return id_to_token

    def token_string(self, token_id: int) -> str:
        """Return the token string for a single id."""
        return self._tokenizer.id_to_token(token_id) or ""
