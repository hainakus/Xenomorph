"""Deterministic masking strategies for MLM benchmarks."""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .tokenizer import BenchmarkTokenizer


@dataclass
class MaskingResult:
    """Result of masking a DNA window."""

    original: str
    masked: str
    positions: List[int]
    mask_token: str
    seed: int = 0
    # Token-level fields (populated when a tokenizer is available).
    token_level: bool = False
    tokens: Optional[List[str]] = None
    token_ids: Optional[List[int]] = None
    masked_token_positions: Optional[List[int]] = None
    masked_token_ids: Optional[List[int]] = None


@dataclass
class MaskingStrategy:
    """Configuration for deterministic per-position or per-token MLM masking."""

    mask_prob: float = 0.15
    mask_token: str = "["
    tokenizer: Optional[BenchmarkTokenizer] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError(f"mask_prob must be in [0, 1], got {self.mask_prob}")

    @classmethod
    def for_model(
        cls,
        model_id: str,
        mask_prob: float = 0.15,
        tokenizer: Optional[BenchmarkTokenizer] = None,
    ) -> "MaskingStrategy":
        """Return the appropriate mask token for a model id."""
        lower = model_id.lower()
        if "mgm" in lower or "mini-genome" in lower:
            token = "["
        else:
            token = "<mask>"
        return cls(mask_prob=mask_prob, mask_token=token, tokenizer=tokenizer)

    def mask_sequence(self, sequence: str, seed: int) -> MaskingResult:
        """Mask positions in `sequence` deterministically using `seed`.

        The same (sequence, seed, mask_prob) triple always produces the same
        mask positions, which makes benchmark results reproducible.

        If a tokenizer is configured, masking is performed at the token level
        (whole BPE/k-mer tokens are replaced), which is the correct setup for
        literature-comparable MLM cross-entropy/perplexity.
        """
        if not sequence:
            raise ValueError("cannot mask an empty sequence")

        if self.tokenizer is not None:
            return self._mask_tokens(sequence, seed)
        return self._mask_characters(sequence, seed)

    def _mask_tokens(self, sequence: str, seed: int) -> MaskingResult:
        """Mask whole tokens in `sequence`."""
        tokens = self.tokenizer.token_strings()
        token_ids = self.tokenizer.encode(sequence, add_special_tokens=False)
        if not token_ids:
            raise ValueError("Tokenizer produced no tokens for input")

        rng = random.Random(seed)
        positions = [i for i in range(len(token_ids)) if rng.random() < self.mask_prob]

        if not positions:
            positions = [rng.randrange(len(token_ids))]

        seen = set()
        unique_positions: List[int] = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique_positions.append(p)

        masked_token_ids = [token_ids[p] for p in unique_positions]

        # Build the masked sequence by concatenating token strings, replacing
        # selected tokens with the model's mask token.  Special tokens such as
        # <mask> are not split by BPE, so the server tokenization will see the
        # same number of tokens and the mask at the same positions.
        masked_strings: List[str] = []
        for i, tid in enumerate(token_ids):
            if i in seen:
                masked_strings.append(self.mask_token)
            else:
                masked_strings.append(tokens[tid])

        masked = "".join(masked_strings)

        # Map token positions back to character positions for display/legacy
        # base-level metrics.
        char_positions: List[int] = []
        offset = 0
        for i, tid in enumerate(token_ids):
            token_len = len(tokens[tid])
            if i in seen:
                char_positions.append(offset)
            offset += token_len

        return MaskingResult(
            original=sequence,
            masked=masked,
            positions=char_positions,
            mask_token=self.mask_token,
            seed=seed,
            token_level=True,
            tokens=tokens,
            token_ids=token_ids,
            masked_token_positions=unique_positions,
            masked_token_ids=masked_token_ids,
        )

    def _mask_characters(self, sequence: str, seed: int) -> MaskingResult:
        """Mask individual characters (legacy base-reconstruction benchmark)."""
        rng = random.Random(seed)
        positions = [i for i in range(len(sequence)) if rng.random() < self.mask_prob]

        if not positions:
            # Edge case: nothing was masked; force at least one mask.
            positions = [rng.randrange(len(sequence))]

        seen = set()
        unique_positions: List[int] = []
        for p in positions:
            if p not in seen:
                seen.add(p)
                unique_positions.append(p)

        chars = list(sequence)
        for p in unique_positions:
            chars[p] = self.mask_token

        return MaskingResult(
            original=sequence,
            masked="".join(chars),
            positions=unique_positions,
            mask_token=self.mask_token,
            seed=seed,
        )


def mask_sequences(
    sequences: Sequence[str],
    strategy: MaskingStrategy,
    seed: int,
) -> List[MaskingResult]:
    """Mask a list of sequences with per-sequence seeds derived from `seed`."""
    results: List[MaskingResult] = []
    for i, seq in enumerate(sequences):
        results.append(strategy.mask_sequence(seq, seed + i))
    return results
