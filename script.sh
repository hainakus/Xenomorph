#!/usr/bin/env bash

set -euo pipefail

echo "=== Xenom BirdCLEF Devnet Setup ==="
echo "L2 Work Encryption: ENABLED"
echo "Theme: BirdWatch (Blue UI)"
echo ""

read -r -p "BIN path [./target/release]: " BIN
BIN="${BIN:-./target/release}"

read -r -p "Private Key: " PRIVKEY
echo ""

read -r -p "Mining Address: " MINING_ADDR
read -r -p "Node RPC (e.g. 127.0.0.1:18110): " NODE_RPC
NODE_RPC="${NODE_RPC:-127.0.0.1:18110}"

read -r -p "Coordinator URL [http://localhost:8091]: " COORDINATOR
COORDINATOR="${COORDINATOR:-http://localhost:8091}"

echo ""
echo "Kaggle API Key: Auto-detected from ~/.kaggle/kaggle.json"
KAGGLE_KEY=""

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
echo "=== BirdCLEF Configuration ==="
echo "Theme: BirdWatch (Blue UI)"
echo "Competition: birdclef-2025"
echo "Model: Perch v2 (GPU/CUDA) / Stub fallback"
echo "Inference: GPU accelerated with TensorFlow CUDA"
echo "Encryption: ENABLED (2-layer)"
echo "  - Layer 1: Local output files (ChaCha20-Poly1305)"
echo "  - Layer 2: Submitted payload (ECIES)"
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
  echo "Installing Python dependencies for Perch v2 (CUDA)..."
  pip install --quiet --upgrade pip
  
  # Install TensorFlow with CUDA support for Perch v2 GPU inference
  echo "Installing: tensorflow (CUDA), kagglehub, librosa, numpy..."
  pip install --quiet tensorflow kagglehub librosa numpy soundfile || echo "Warning: Some packages failed to install"
  
  echo "Note: Perch v2 will use GPU if CUDA is available, otherwise CPU"
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

echo "=== Starting genetics-l2-fetcher (BirdCLEF) ==="
# Fetches BirdCLEF-2025 competition from Kaggle
# Uses ~/.kaggle/kaggle.json for authentication
FETCHER_CMD="$BIN/genetics-l2-fetcher --coordinator $COORDINATOR --horizon --poll-secs 300 --competition birdclef-2025"
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

echo "=== Starting genome-miner gpu (BirdCLEF + Encryption) ==="
# Processes BirdCLEF acoustic classification jobs
# Encryption enabled:
#   - Encrypts output files locally with worker privkey
#   - Encrypts submitted payload with coordinator pubkey
#   - Uses Perch v2 script for bird audio analysis (GPU mode with CUDA)
"$BIN/genome-miner" gpu \
  --devnet \
  --mining-address "$MINING_ADDR" \
  --stratum stratum+tcp://127.0.0.1:5555 \
  --gpu 0 \
  --l2-coordinator "$COORDINATOR" \
  --l2-private-key "$PRIVKEY" \
  --l2-gpu \
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
echo "=== BirdCLEF L2 System ==="
echo "Pool UI: http://localhost:5555 (Blue BirdWatch theme)"
echo "Coordinator: $COORDINATOR"
echo "Competition: BirdCLEF-2025 (Kaggle)"
echo ""
echo "Encryption Status: ACTIVE"
echo "  ✓ Output files encrypted on disk: /tmp/genome-miner-l2/{job_id}/output/*.enc"
echo "  ✓ Submitted payloads encrypted with coordinator pubkey"
echo "  ✓ Coordinator keypair: /tmp/genetics-l2-nih2.db.key"
echo ""
echo "To follow miner logs:"
echo "  tail -f /tmp/xenom-logs/miner.log"
echo ""
echo "To check coordinator pubkey:"
echo "  curl $COORDINATOR/pubkey"
echo ""
echo "Press Ctrl+C to stop everything."

wait
