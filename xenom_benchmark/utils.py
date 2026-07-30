"""Shared helpers, constants and validation utilities."""

import hashlib
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

BASES = ("A", "C", "G", "T")
BASE_SET = set(BASES)
DNA_TO_INDEX = {b: i for i, b in enumerate(BASES)}
MASK_TOKENS = ("<mask>", "[MASK]", "[")

DEFAULT_SEQ_LEN = 512
DEFAULT_MASK_PROB = 0.15
DEFAULT_N_SAMPLES = 88
DEFAULT_FRAGMENT_SIZE = 1_048_576

DEFAULT_GRPC_ADDR = "94.237.108.145:50051"
DEFAULT_TIMEOUT = 120
BENCHMARK_VERSION = "1.0.0"

logger = logging.getLogger("xenom_benchmark")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger for the benchmark."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


def repo_root() -> Path:
    """Return the repository root (parent of the `xenom_benchmark` package)."""
    return Path(__file__).resolve().parent.parent


def add_scripts_to_path() -> None:
    """Ensure generated gRPC stubs under `scripts/xenom_eval` are importable."""
    stubs_dir = str(repo_root() / "scripts" / "xenom_eval")
    if stubs_dir not in sys.path:
        sys.path.insert(0, stubs_dir)


def sha256_file(path: Union[str, Path]) -> str:
    """Return the SHA-256 hex digest of a file, reading in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now_iso() -> str:
    """Return the current UTC timestamp as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def is_valid_base(char: str) -> bool:
    """Return True if `char` is a canonical DNA base."""
    return char in BASE_SET


def validate_bases(seq: str, label: str = "sequence") -> None:
    """Raise ValueError if `seq` contains non-DNA characters."""
    invalid = set(seq) - BASE_SET
    if invalid:
        raise ValueError(f"{label} contains invalid bases: {sorted(invalid)}")


def validate_confidence(confidence: float) -> None:
    """Raise ValueError if `confidence` is NaN, Inf or outside [0, 1]."""
    if np.isnan(confidence) or np.isinf(confidence):
        raise ValueError(f"confidence is not finite: {confidence}")
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence out of [0, 1] range: {confidence}")


def validate_predictions(
    original: str,
    masked: str,
    predicted: str,
    mask_positions: Sequence[int],
    confidence: float,
) -> List[str]:
    """Validate an inference response and return a list of warnings."""
    warnings: List[str] = []

    if not predicted:
        raise ValueError("predicted sequence is empty")

    validate_confidence(confidence)

    if len(mask_positions) != masked.count("[") + masked.count("<"):
        # The mask token may be a single char or a substring; the count is only
        # a sanity check and not a hard requirement for BPE tokenizers.
        pass

    if len(predicted) != len(original):
        warnings.append(
            f"predicted length ({len(predicted)}) != original length ({len(original)}); "
            "alignment will be used to map mask positions."
        )

    invalid = set(predicted) - BASE_SET
    if invalid:
        warnings.append(f"predicted sequence contains non-DNA characters: {sorted(invalid)}")

    duplicates = len(mask_positions) - len(set(mask_positions))
    if duplicates:
        warnings.append(f"{duplicates} duplicate mask positions detected")

    return warnings


def safe_mean(values: Sequence[float]) -> float:
    """Return the mean of `values` or 0.0 if the list is empty."""
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.mean(arr))


def safe_log(x: float, eps: float = 1e-12) -> float:
    """Return log(max(x, eps)) to avoid -inf on zero probabilities."""
    return float(np.log(max(x, eps)))


def frequency_map(values: Sequence[str]) -> Dict[str, float]:
    """Return normalized frequencies of the given categorical values."""
    total = len(values)
    if total == 0:
        return {b: 0.0 for b in BASES}
    counts: Dict[str, int] = {b: 0 for b in BASES}
    for v in values:
        if v in counts:
            counts[v] += 1
    return {b: counts[b] / total for b in BASES}


def ensure_dir(path: Union[str, Path]) -> Path:
    """Create `path` if it does not exist and return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
