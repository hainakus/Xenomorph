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
echo "Competition: birdclef-2026"
echo "Model: Google YAMNet (GPU/CUDA) / Stub fallback"
echo "Inference: GPU accelerated with TensorFlow CUDA"
echo "Encryption: ENABLED (2-layer)"
echo "  - Layer 1: Local output files (ChaCha20-Poly1305)"
echo "  - Layer 2: Submitted payload (ECIES)"
echo ""

# Kaggle API key setup (for coordinator only - miners will download from coordinator API)
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

if [ ! -f "$KAGGLE_JSON" ]; then
  echo "=== Kaggle API Setup ==="
  echo "Coordinator needs Kaggle credentials to download BirdCLEF datasets."
  echo "Miners will download datasets from coordinator API (no Kaggle key needed on miners)."
  echo ""
  read -p "Do you have a Kaggle API key? (y/n): " has_key
  
  if [ "$has_key" = "y" ] || [ "$has_key" = "Y" ]; then
    echo ""
    echo "Please enter your Kaggle credentials:"
    read -p "Kaggle username: " kaggle_user
    read -p "Kaggle API key: " kaggle_key
    
    mkdir -p "$KAGGLE_DIR"
    cat > "$KAGGLE_JSON" <<EOF
{"username":"$kaggle_user","key":"$kaggle_key"}
EOF
    chmod 600 "$KAGGLE_JSON"
    echo "✓ Kaggle credentials saved to $KAGGLE_JSON"
  else
    echo "Skipping Kaggle setup. Coordinator will use stub data."
    echo "To add credentials later, create: $KAGGLE_JSON"
  fi
  echo ""
else
  echo "✓ Kaggle credentials found at $KAGGLE_JSON"
fi
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
  
  # Install TensorFlow with CUDA support for YAMNet GPU inference
  echo "Installing: tensorflow (CUDA), tensorflow-hub, librosa, numpy..."
  pip install --quiet tensorflow tensorflow-hub librosa numpy soundfile || echo "Warning: Some packages failed to install"
  
  echo "Note: YAMNet will use GPU if CUDA is available, otherwise CPU"
else
  echo "Warning: venv not available, using system Python"
  if command -v pip3 &> /dev/null; then
    pip3 install --quiet --break-system-packages kaggle kagglehub tensorflow librosa numpy 2>/dev/null || \
    pip3 install --quiet --user kaggle kagglehub tensorflow librosa numpy 2>/dev/null || \
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

# Pre-download BirdCLEF-2026 dataset to cache before starting pool/miner
BIRDCLEF_CACHE="/tmp/kaggle-datasets/_cache/birdclef-2026"
if [ ! -f "$BIRDCLEF_CACHE/.ready" ]; then
  echo "=== Pre-downloading BirdCLEF-2026 dataset (required before mining) ==="
  mkdir -p "$BIRDCLEF_CACHE"
  if kaggle competitions download -c birdclef-2026 -p "$BIRDCLEF_CACHE/" 2>&1 | tee -a /tmp/xenom-logs/coordinator.log; then
    echo "Extracting BirdCLEF-2026..."
    for z in "$BIRDCLEF_CACHE"/*.zip; do
      [ -f "$z" ] && unzip -o -q "$z" -d "$BIRDCLEF_CACHE/" && rm -f "$z"
    done
    touch "$BIRDCLEF_CACHE/.ready"
    echo "BirdCLEF-2026 dataset ready: $(find $BIRDCLEF_CACHE -name '*.ogg' -o -name '*.wav' | wc -l) audio files"
  else
    echo "Warning: BirdCLEF-2026 download failed, miners will use stub"
  fi
else
  echo "=== BirdCLEF-2026 dataset already cached ==="
  echo "Audio files: $(find $BIRDCLEF_CACHE -name '*.ogg' -o -name '*.wav' 2>/dev/null | wc -l)"
fi

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

echo "=== Starting genetics-l2-fetcher (BirdCLEF only) ==="
# Fetches ONLY BirdCLEF-2026 competition from Kaggle
# Uses ~/.kaggle/kaggle.json for authentication
# --kaggle-only disables NIH and other default sources
"$BIN/genetics-l2-fetcher" \
  --coordinator "$COORDINATOR" \
  --competition birdclef-2026 \
  --kaggle-only \
  --poll-secs 300 \
  > /tmp/xenom-logs/fetcher.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-validator ==="
# Validator needs coordinator privkey to decrypt encrypted results
COORDINATOR_PRIVKEY_FILE="/tmp/genetics-l2-nih2.db.key"
COORDINATOR_PRIVKEY=""
if [ -f "$COORDINATOR_PRIVKEY_FILE" ]; then
  COORDINATOR_PRIVKEY=$(cat "$COORDINATOR_PRIVKEY_FILE")
fi

"$BIN/genetics-l2-validator" \
  --private-key "$PRIVKEY" \
  --coordinator "$COORDINATOR" \
  --coordinator-privkey "$COORDINATOR_PRIVKEY" \
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
  --no-tui \
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
echo "Competition: BirdCLEF-2026 (Kaggle)"
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
