#!/usr/bin/env bash
# mainnet.sh - Xenomorph Mainnet with EVM + Genetics L2 + Bioproof
set -euo pipefail

# ── keygen subcommand ─────────────────────────────────────────────────────────
# Usage: ./mainnet.sh keygen
# Generates a fresh secp256k1 keypair + derives its xenom: mainnet address.
#
# Usage: ./mainnet.sh addr <64-hex-privkey>
# Derives the xenom: mainnet address from an existing private key.
#
# Key architecture:
#   VALIDATOR_PRIVKEY / SETTLEMENT_PRIVKEY  → secp256k1 hex (L2 encryption/signing)
#   xenom: address (funding/payout)         → derived from same key via X-only P2PK
#   MINING_ADDR                             → xenom-wallet Schnorr keypair (separate)

_ensure_worker_bin() {
  BIN_TMP=./target/release
  if [[ ! -x "$BIN_TMP/genetics-l2-worker" ]]; then
    echo "Building genetics-l2-worker..."
    cargo build --release --bin genetics-l2-worker 2>&1 | tail -3
  fi
}

if [[ "${1:-}" == "keygen" ]]; then
  _ensure_worker_bin
  echo ""
  echo "=== Generating new secp256k1 keypair ==="
  "$BIN_TMP/genetics-l2-worker" --gen-key
  echo ""
  echo "Set env vars before running mainnet.sh:"
  echo "  export VALIDATOR_PRIVKEY=<private key above>"
  echo "  export COMMITTER_PRIVKEY=<private key above>  # or a separate funded key"
  echo "  export MINING_ADDR=<xenom: address from xenom-wallet (separate Schnorr key)>"
  exit 0
fi

if [[ "${1:-}" == "addr" ]]; then
  if [[ -z "${2:-}" ]]; then
    echo "Usage: ./mainnet.sh addr <64-hex-privkey>"
    exit 1
  fi
  _ensure_worker_bin
  echo ""
  echo "=== Deriving xenom: address from private key ==="
  WORKER_PRIVKEY="${2}" "$BIN_TMP/genetics-l2-worker" --show-addr
  exit 0
fi

echo "=== Xenomorph Mainnet Setup ==="
echo "EVM + Genetics L2 + Bioproof + GPU Mining"

# Config
BIN=./target/release
MINING_ADDR=${MINING_ADDR:-}              # required: set MINING_ADDR env var
NODE_RPC=127.0.0.1:36669                 # Mainnet gRPC port
COORDINATOR_URL=http://localhost:8091
EVM_RPC=127.0.0.1:8545
STRATUM_PORT=5555
DATA_DIR=~/.xenom-mainnet
EVM_CHAIN_ID=${EVM_CHAIN_ID:-7777}        # Xenom mainnet EVM chain ID

# Required private keys (set these env vars before running)
: "${COMMITTER_PRIVKEY:?Set COMMITTER_PRIVKEY (64 hex chars secp256k1 key for anchor committer funding address)}"
: "${VALIDATOR_PRIVKEY:?Set VALIDATOR_PRIVKEY (64 hex chars secp256k1 key for the L2 validator)}"
: "${MINING_ADDR:?Set MINING_ADDR (xenom: mainnet mining address)}"
# Share the same key across coordinator/validator/settlement so decryption works
COORDINATOR_PRIVKEY="$VALIDATOR_PRIVKEY"
SETTLEMENT_PRIVKEY="$VALIDATOR_PRIVKEY"
WORKER_PRIVKEY="$VALIDATOR_PRIVKEY"
export COMMITTER_PRIVKEY VALIDATOR_PRIVKEY COORDINATOR_PRIVKEY SETTLEMENT_PRIVKEY WORKER_PRIVKEY

# Create data dirs
mkdir -p "$DATA_DIR" /tmp/xenom-logs

# Build binaries
echo "Building binaries..."
cargo build --release \
  --bin xenom \
  --bin xenom-stratum-bridge \
  --bin genetics-l2-coordinator \
  --bin genetics-l2-fetcher \
  --bin genetics-l2-validator \
  --bin genetics-l2-worker \
  --bin genetics-l2-settlement \
  --bin genome-miner \
  --bin xenom-evm-node \
  --bin xenom-anchor-committer

XENOM_PID=""

# 1. Start Xenom Mainnet Node
echo "Starting Xenom Mainnet Node..."
# $BIN/xenom --datadir $DATA_DIR --utxoindex --rpclisten $NODE_RPC \
#   > /tmp/xenom-logs/xenom-mainnet.log 2>&1 &
# XENOM_PID=$!
# sleep 10

# 2. Start Genetics L2 Coordinator
echo "Starting Genetics L2 Coordinator..."
$BIN/genetics-l2-coordinator \
  --db-path $DATA_DIR/genetics-l2.db \
  --listen 0.0.0.0:8091 \
  --scripts-dir scripts \
  --datasets-dir $DATA_DIR/datasets \
  > /tmp/xenom-logs/coordinator.log 2>&1 &
COORDINATOR_PID=$!
sleep 5

# 3. Start Xenom EVM Node (Mainnet, DAA-tied to L1)
echo "Starting Xenom EVM Node..."
$BIN/xenom-evm-node \
  --chain-id $EVM_CHAIN_ID \
  --rpc-addr $EVM_RPC \
  --state-dir $DATA_DIR/evm-state \
  --l1-node $NODE_RPC \
  > /tmp/xenom-logs/evm-node.log 2>&1 &
EVM_PID=$!
sleep 5

# 4. Start Stratum Bridge (Mainnet + L2)
echo "Starting Stratum Bridge..."
$BIN/xenom-stratum-bridge \
  --mining-address $MINING_ADDR \
  --rpcserver $NODE_RPC \
  --listen 0.0.0.0:$STRATUM_PORT \
  --db-path $DATA_DIR/pool.db \
  --l2-coordinator $COORDINATOR_URL \
  --l2-theme genetics \
  > /tmp/xenom-logs/stratum-bridge.log 2>&1 &
STRATUM_PID=$!
sleep 5

# 5. Start Genetics L2 Fetcher (Kaggle, NIH, etc.)
echo "Starting L2 Fetcher..."
$BIN/genetics-l2-fetcher \
  --coordinator $COORDINATOR_URL \
  --sra --igsr --gnomad --gdc --clinvar \
  > /tmp/xenom-logs/fetcher.log 2>&1 &
FETCHER_PID=$!
sleep 5

# 6. Start L2 Validator
echo "Starting L2 Validator..."
VALIDATOR_PRIVKEY=$VALIDATOR_PRIVKEY \
$BIN/genetics-l2-validator \
  --coordinator $COORDINATOR_URL \
  --coordinator-key-file $DATA_DIR/genetics-l2.db.key \
  --poll-ms 10000 \
  --score-tolerance 0.05 \
  > /tmp/xenom-logs/validator.log 2>&1 &
VALIDATOR_PID=$!
sleep 5

# 7. Start L2 Settlement (anchors via EVM; anchor committer bridges EVM → L1)
echo "Starting L2 Settlement..."
$BIN/genetics-l2-settlement \
  --coordinator $COORDINATOR_URL \
  --node "grpc://$NODE_RPC" \
  --evm-node http://$EVM_RPC \
  --submit \
  --poll-ms 15000 \
  --quorum 1 \
  --score-tolerance 0.05 \
  > /tmp/xenom-logs/settlement.log 2>&1 &
SETTLEMENT_PID=$!
sleep 5

# 8. Start Anchor Committer (L2 → L1)
echo "Starting Anchor Committer..."
COMMITTER_PRIVKEY=$COMMITTER_PRIVKEY \
$BIN/xenom-anchor-committer \
  --evm-node $EVM_RPC \
  --node $NODE_RPC \
  --state-dir $DATA_DIR/anchor-state \
  --poll-ms 10000 \
  --submit \
  > /tmp/xenom-logs/anchor-committer.log 2>&1 &
ANCHOR_PID=$!
sleep 5

# 9. Start genetics-l2-worker (standalone L2 job processor)
echo "Starting genetics-l2-worker..."
L2_WORK_DIR="$DATA_DIR/l2-work"
mkdir -p "$L2_WORK_DIR"
$BIN/genetics-l2-worker \
  --coordinator $COORDINATOR_URL \
  --work-root "$L2_WORK_DIR" \
  --poll-ms 5000 \
  > /tmp/xenom-logs/l2-worker.log 2>&1 &
WORKER_PID=$!
sleep 2

echo ""
echo "=== Xenomorph Mainnet Running ==="
echo "Node RPC:     $NODE_RPC"
echo "EVM RPC:      $EVM_RPC"
echo "Stratum:      stratum+tcp://127.0.0.1:$STRATUM_PORT"
echo "Coordinator:  $COORDINATOR_URL"
echo "Data:         $DATA_DIR"
echo "Logs:         /tmp/xenom-logs/*"
echo ""
echo "PIDs: $XENOM_PID $COORDINATOR_PID $EVM_PID $STRATUM_PID $FETCHER_PID $VALIDATOR_PID $SETTLEMENT_PID $ANCHOR_PID $WORKER_PID"
echo ""
echo "Press Ctrl+C to stop."

trap "kill $XENOM_PID $COORDINATOR_PID $EVM_PID $STRATUM_PID $FETCHER_PID $VALIDATOR_PID $SETTLEMENT_PID $ANCHOR_PID $WORKER_PID 2>/dev/null || true" INT TERM

wait
