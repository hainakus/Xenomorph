#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# node.sh - Build and start the Xenom node (kaspa-daemon)
# Usage: ./scripts/node.sh [--devnet|--testnet|--mainnet] [--build]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$REPO_ROOT/target/release/xenom"
DATA_DIR="$HOME/.xenom"
NETWORK="--devnet"
BUILD=0

for arg in "$@"; do
  case $arg in
    --devnet|--testnet|--mainnet) NETWORK="$arg" ;;
    --build) BUILD=1 ;;
  esac
done

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ $BUILD -eq 1 || ! -f "$BINARY" ]]; then
  echo "[node] Building xenom..."
  cargo build --release -p xenom --manifest-path "$REPO_ROOT/Cargo.toml"
fi

# ── Data dir ──────────────────────────────────────────────────────────────────
mkdir -p "$DATA_DIR"

echo "[node] Starting Xenom node ($NETWORK)..."
echo "[node] Data dir : $DATA_DIR"
echo "[node] gRPC     : 127.0.0.1:18610"
echo "[node] P2P      : 0.0.0.0:18611"
echo ""

exec "$BINARY" \
  $NETWORK \
  --appdir "$DATA_DIR" \
  --rpclisten=127.0.0.1:18610 \
  --listen=0.0.0.0:18611 \
  --utxoindex
