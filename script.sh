#!/usr/bin/env bash

set -euo pipefail

echo "=== Xenom Devnet Setup ==="

read -r -p "BIN path [./target/release]: " BIN
BIN="${BIN:-./target/release}"

read -r -p "Private Key: " PRIVKEY
echo ""

read -r -p "Mining Address: " MINING_ADDR
read -r -p "Node RPC (e.g. 127.0.0.1:16110): " NODE_RPC

read -r -p "Coordinator URL [http://localhost:8091]: " COORDINATOR
COORDINATOR="${COORDINATOR:-http://localhost:8091}"

echo ""
read -r -p "Kaggle API Key (username:token) [optional, press Enter to skip]: " KAGGLE_KEY

export BIN
export PRIVKEY
export MINING_ADDR
export NODE_RPC
export COORDINATOR
export KAGGLE_KEY

echo ""
echo "=== Environment ==="
echo "BIN=$BIN"
echo "MINING_ADDR=$MINING_ADDR"
echo "NODE_RPC=$NODE_RPC"
echo "COORDINATOR=$COORDINATOR"
echo ""

if [ ! -d "$BIN" ]; then
  echo "Error: BIN directory not found: $BIN"
  exit 1
fi

mkdir -p /tmp/xenom-logs

echo "=== Setting up Python virtual environment ==="
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv || {
    echo "Warning: Failed to create venv, will try system Python"
  }
fi

if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
  echo "Installing Python dependencies for BirdCLEF..."
  pip install --quiet --upgrade pip
  
  # Install essential dependencies (Perch/TensorFlow optional, will use BirdNET fallback)
  echo "Installing: kagglehub, librosa, numpy..."
  pip install --quiet kagglehub librosa numpy soundfile || echo "Warning: Some packages failed to install"
  
  # Try to install birdnet-analyzer as fallback
  echo "Installing birdnet-analyzer (fallback classifier)..."
  pip install --quiet birdnet-analyzer 2>/dev/null || echo "Note: birdnet-analyzer not installed, will use stub mode"
else
  echo "Warning: venv not available, using system Python"
  if command -v pip3 &> /dev/null; then
    pip3 install --quiet --break-system-packages kagglehub tensorflow librosa numpy 2>/dev/null || \
    pip3 install --quiet --user kagglehub tensorflow librosa numpy 2>/dev/null || \
    echo "Warning: pip3 install failed, continuing anyway..."
  fi
fi
echo ""

PIDS=()

cleanup() {
  echo ""
  echo "=== Stopping processes ==="
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait || true
}

trap cleanup EXIT INT TERM

echo "=== Building ==="
cargo build --release \
  -p xenom \
  -p xenom-stratum-bridge \
  -p genetics-l2-coordinator \
  -p genetics-l2-fetcher \
  -p genetics-l2-validator \
  -p genetics-l2-settlement \
  -p genome-miner \
  -p xenom-evm-node

echo "=== Starting xenom node ==="
"$BIN/xenom" --devnet --utxoindex --rpclisten="$NODE_RPC"\
  > /tmp/xenom-logs/xenom.log 2>&1 &
PIDS+=($!)
sleep 3

echo "=== Starting genetics-l2-coordinator ==="
"$BIN/genetics-l2-coordinator" \
  --db-path /tmp/genetics-l2-nih2.db \
  --listen 0.0.0.0:8091 \
  > /tmp/xenom-logs/coordinator.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting xenom-evm-node ==="
"$BIN/xenom-evm-node" \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --block-time 2000 \
  > /tmp/xenom-logs/evm-node.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting xenom-stratum-bridge ==="
"$BIN/xenom-stratum-bridge" \
  --mining-address "$MINING_ADDR" \
  --rpcserver "$NODE_RPC" \
  --listen 0.0.0.0:5555 \
  --l2-coordinator "$COORDINATOR" \
  --db-path /tmp/genetics-l2-nih2-pool.db \
  --l2-theme birdclef \
  --devnet \
  > /tmp/xenom-logs/stratum-bridge.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-fetcher ==="
FETCHER_CMD="$BIN/genetics-l2-fetcher --coordinator $COORDINATOR --horizon --poll-secs 300"
if [ -n "$KAGGLE_KEY" ]; then
  FETCHER_CMD="$FETCHER_CMD --kaggle-key $KAGGLE_KEY --competition birdclef-2025"
fi
eval "$FETCHER_CMD" > /tmp/xenom-logs/fetcher.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-validator ==="
"$BIN/genetics-l2-validator" \
  --private-key "$PRIVKEY" \
  --coordinator "$COORDINATOR" \
  --poll-ms 10000 \
  --score-tolerance 0.05 \
  > /tmp/xenom-logs/validator.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-settlement ==="
"$BIN/genetics-l2-settlement" \
  --coordinator "$COORDINATOR" \
  --node "grpc://$NODE_RPC" \
  --private-key "$PRIVKEY" \
  --submit \
  --devnet \
  --poll-ms 15000 \
  > /tmp/xenom-logs/settlement.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genome-miner gpu ==="
"$BIN/genome-miner" gpu \
  --devnet \
  --mining-address "$MINING_ADDR" \
  --stratum stratum+tcp://127.0.0.1:5555 \
  --gpu 0 \
  --l2-coordinator "$COORDINATOR" \
  --l2-private-key "$PRIVKEY" \
  --l2-perch-script scripts/perch_infer.py \
  > /tmp/xenom-logs/miner.log 2>&1 &
PIDS+=($!)
sleep 2

echo ""
echo "=== Services started ==="
echo "Logs:"
echo "  /tmp/xenom-logs/xenom.log"
echo "  /tmp/xenom-logs/coordinator.log"
echo "  /tmp/xenom-logs/evm-node.log"
echo "  /tmp/xenom-logs/stratum-bridge.log"
echo "  /tmp/xenom-logs/fetcher.log"
echo "  /tmp/xenom-logs/validator.log"
echo "  /tmp/xenom-logs/settlement.log"
echo "  /tmp/xenom-logs/miner.log"
echo ""
echo "To follow miner logs:"
echo "  tail -f /tmp/xenom-logs/miner.log"
echo ""
echo "Press Ctrl+C to stop everything."

wait
