#!/usr/bin/env python3
"""Register a DNABERT-2 model for from-scratch training.

Downloads only `config.json` and `tokenizer.json` from a Hugging Face model
repo and stores them (encrypted) in the local model cache.  `weights.enc` is
written as an empty file, which the node and miner interpret as "train this
model from scratch".

The encryption uses the same scheme as the Rust `model-crypto` crate:
AES-256-GCM with a 12-byte random nonce prepended to the ciphertext.  The key
is derived from `XENO_MODEL_KEY` (or a default devnet string) exactly like the
node and miner do.

Usage:
    python scripts/register_dnabert2_from_scratch.py \
        --models-dir ~/.xenom/models \
        --model-id zhihan1996/DNABERT-2-117M

After running, point `XENO_DEFAULT_MODEL_ID` or `--active-model-id` to the same
`model_id` and start the node.  The node will see the empty `weights.enc` and
serve a randomly-initialized DNABERT-2.
"""

import argparse
import hashlib
import os
import sys
import urllib.request
from pathlib import Path
from typing import Optional


try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as e:  # pragma: no cover
    print("Missing Python dependency: cryptography>=3.0", file=sys.stderr)
    print("Install with: pip install cryptography", file=sys.stderr)
    raise SystemExit(1) from e

DEFAULT_KEY_SEED = "xenom-devnet-model-key"
NONCE_LEN = 12
HF_HUB_URL = "https://huggingface.co"
REQUEST_TIMEOUT = 300


def derive_encryption_key() -> bytes:
    """Mirror model_crypto::derive_encryption_key."""
    seed = os.environ.get("XENO_MODEL_KEY", DEFAULT_KEY_SEED).strip()

    if len(seed) == 64:
        try:
            decoded = bytes.fromhex(seed)
            if len(decoded) == 32:
                return decoded
        except ValueError:
            pass

    return hashlib.sha256(seed.encode()).digest()


def encrypt(data: bytes, key: bytes) -> bytes:
    """AES-256-GCM; Rust format is `nonce || ciphertext`."""
    nonce = os.urandom(NONCE_LEN)
    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, data, None)
    return nonce + ciphertext


def key_hash(key: bytes) -> bytes:
    return hashlib.sha256(key).digest()


def sanitize_model_id(model_id: str) -> str:
    """Mirror ModelStorage::sanitize_id."""
    return "".join("_" if c in "/\\: \0" else c for c in model_id)


def hf_resolve_url(model_id: str, filename: str) -> str:
    return f"{HF_HUB_URL}/{model_id}/resolve/main/{filename}"


def download_file(url: str) -> bytes:
    """Download a single file using stdlib urllib (no external dependencies)."""
    for variant in (url, f"{url}?download=true"):
        try:
            with urllib.request.urlopen(variant, timeout=REQUEST_TIMEOUT) as response:
                return response.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
    raise RuntimeError(f"Could not download {url} (HTTP 404)")


def register_from_scratch(
    models_dir: Path,
    model_id: str,
    hf_model_id: str,
) -> None:
    key = derive_encryption_key()

    print(f"Downloading config.json for {hf_model_id} ...")
    config = download_file(hf_resolve_url(hf_model_id, "config.json"))
    print(f"Downloading tokenizer.json for {hf_model_id} ...")
    tokenizer = download_file(hf_resolve_url(hf_model_id, "tokenizer.json"))

    safe_id = sanitize_model_id(model_id)
    model_path = models_dir / safe_id
    model_path.mkdir(parents=True, exist_ok=True)

    (model_path / "config.enc").write_bytes(encrypt(config, key))
    (model_path / "tokenizer.enc").write_bytes(encrypt(tokenizer, key))
    (model_path / "weights.enc").write_bytes(encrypt(b"", key))  # from-scratch marker
    (model_path / "model.keyhash").write_bytes(key_hash(key))

    print(f"Registered from-scratch DNABERT-2 at {model_path}")
    print(f"  config.enc      {len(config):>10} bytes plaintext")
    print(f"  tokenizer.enc   {len(tokenizer):>10} bytes plaintext")
    print(f"  weights.enc              0 bytes (from-scratch marker)")
    print(f"  model.keyhash           32 bytes")
    print()
    print("Set your node to use this model, e.g.")
    print(f"  --active-model-id {model_id}")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models-dir",
        required=True,
        type=Path,
        help="Directory where the node stores its model cache (e.g. ~/.xenom/models).",
    )
    parser.add_argument(
        "--model-id",
        default="zhihan1996/DNABERT-2-117M",
        help="Model id the node should use for this checkpoint (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-model-id",
        help="Hugging Face repo to download config/tokenizer from. Defaults to --model-id.",
    )
    args = parser.parse_args(argv)

    hf_model_id = args.hf_model_id or args.model_id
    register_from_scratch(args.models_dir, args.model_id, hf_model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
