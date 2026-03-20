#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Xenom L2 Genetics Devnet Setup ==="
echo "L2 Work Encryption: ENABLED"
echo "Primary:   Genomics — VCF → normalization → annotation → gene scoring → cohort grouping"
echo "Reference: GRCh38"
echo ""

read -r -p "BIN path [./target/release]: " BIN
BIN="${BIN:-./target/release}"

read -r -p "Private Key (L2 worker/settlement): " PRIVKEY
echo ""

read -r -p "Mining Address: " MINING_ADDR
read -r -p "Node RPC (e.g. 127.0.0.1:18110): " NODE_RPC
NODE_RPC="${NODE_RPC:-127.0.0.1:18110}"

read -r -p "Coordinator URL [http://localhost:8091]: " COORDINATOR
COORDINATOR="${COORDINATOR:-http://localhost:8091}"

echo ""
echo "--- L1 Anchor Committer ---"
echo "  To commit EVM checkpoints to L1, enter the private key whose address"
echo "  corresponds to your mining address (it must hold UTXOs from block rewards)."
echo "  Leave blank to run in dry-run mode (checkpoints logged, not submitted)."
read -r -p "Anchor Committer Key [blank = dry-run]: " ANCHOR_KEY
if [[ -n "$ANCHOR_KEY" ]]; then
  ANCHOR_SUBMIT=1
else
  ANCHOR_SUBMIT=0
  echo "  (dry-run: anchor commits disabled)"
fi
echo ""

export BIN
export MINING_ADDR
export NODE_RPC
export COORDINATOR

# Export private key as env vars — never pass as CLI args (ps aux / /proc/$PID/cmdline exposure)
export SETTLEMENT_PRIVKEY="$PRIVKEY"
export VALIDATOR_PRIVKEY="$PRIVKEY"
export L2_PRIVKEY="$PRIVKEY"
export WORKER_PRIVKEY="$PRIVKEY"   # genetics-l2-worker standalone L2 worker
export COMMITTER_PRIVKEY="${ANCHOR_KEY:-}" # xenom-anchor-committer L2→L1 checkpoint commits
unset PRIVKEY ANCHOR_KEY  # clear originals after mapping to named vars

echo ""
echo "=== Environment ==="
echo "BIN=$BIN"
echo "MINING_ADDR=$MINING_ADDR"
echo "NODE_RPC=$NODE_RPC"
echo "COORDINATOR=$COORDINATOR"
echo ""
echo "=== Configuration ==="
echo "Primary:   Genomics (GRCh38 VCF → Ensembl VEP → gene scores)"
echo "Encryption: ENABLED (ChaCha20-Poly1305 local + ECIES submit)"
echo ""

# ── Dataset sources ──────────────────────────────────────────────────────────
echo "=== Dataset Sources ==="
echo "--- Genomics (primary, GRCh38, all open-access) ---"
read -r -p "Enable SRA    (NCBI SRA — GIAB benchmark VCFs)         ? [Y/n]: " _ans
OPT_SRA=1;    [[ "$_ans" =~ ^[Nn]$ ]] && OPT_SRA=0

read -r -p "Enable IGSR   (1000 Genomes 30x phased VCFs, chr1-X)   ? [Y/n]: " _ans
OPT_IGSR=1;   [[ "$_ans" =~ ^[Nn]$ ]] && OPT_IGSR=0

read -r -p "Enable gnomAD (gnomAD v4.1 exome AF annotation)        ? [Y/n]: " _ans
OPT_GNOMAD=1; [[ "$_ans" =~ ^[Nn]$ ]] && OPT_GNOMAD=0

read -r -p "Enable GDC    (TCGA open-access cancer cohorts)         ? [Y/n]: " _ans
OPT_GDC=1;    [[ "$_ans" =~ ^[Nn]$ ]] && OPT_GDC=0

read -r -p "Enable ClinVar (weekly GRCh38 clinical variant VCF)    ? [Y/n]: " _ans
OPT_CLINVAR=1; [[ "$_ans" =~ ^[Nn]$ ]] && OPT_CLINVAR=0

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
  echo "Installing Python dependencies for genomics pipeline..."
  pip install --quiet --upgrade pip
  
  # Genomics pipeline (genome_annotate.py)
  echo "Installing: numpy, pandas, requests..."
  pip install --quiet numpy pandas requests || echo "Warning: Some packages failed to install"
else
  echo "Warning: venv not available, using system Python"
  if command -v pip3 &> /dev/null; then
    pip3 install --quiet --break-system-packages numpy pandas requests 2>/dev/null || \
    pip3 install --quiet --user numpy pandas requests 2>/dev/null || \
    echo "Warning: pip3 install failed, continuing anyway..."
  fi
fi
echo ""

# ── NVIDIA GPU detection ─────────────────────────────────────────────────────
GPU_AVAILABLE=0
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null 2>&1; then
  GPU_AVAILABLE=1
  CUDA_MAJ=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+' | head -1)
  CUDA_MAJ="${CUDA_MAJ:-12}"
  echo "=== NVIDIA GPU detected (CUDA ${CUDA_MAJ}.x) — enabling GPU acceleration ==="
  export USE_GPU=1
  echo "  Installing cupy-cuda${CUDA_MAJ}x for Python GPU acceleration..."
  if [ -d "venv" ]; then
    pip install --quiet "cupy-cuda${CUDA_MAJ}x" 2>/dev/null \
      || pip install --quiet cupy 2>/dev/null \
      || echo "  Warning: cupy install failed — Python will use numpy/CPU"
  else
    pip3 install --quiet --break-system-packages "cupy-cuda${CUDA_MAJ}x" 2>/dev/null \
      || echo "  Warning: cupy install failed — Python will use numpy/CPU"
  fi
else
  GPU_AVAILABLE=0
  export USE_GPU=0
  echo "=== No NVIDIA GPU detected — using CPU mining and numpy/CPU pipeline ==="
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
  -p genetics-l2-worker \
  -p genetics-l2-settlement \
  -p genome-miner \
  -p xenom-evm-node \
  -p xenom-anchor-committer

echo "=== Starting xenom node ==="
"$BIN/xenom" --devnet --utxoindex --rpclisten="$NODE_RPC"\
  > /tmp/xenom-logs/xenom.log 2>&1 &
PIDS+=($!)
sleep 3

# Persistent data directory (survives reboots)
XENOM_DATA="${XENOM_DATA_DIR:-$HOME/.local/share/xenom}"
KAGGLE_DATASETS_DIR="$XENOM_DATA/kaggle-datasets"
COORDINATOR_DB="$XENOM_DATA/genetics-l2.db"
EVM_STATE_DIR="$XENOM_DATA/evm-state"
COMMITTER_STATE_DIR="$XENOM_DATA/evm-committer"
mkdir -p "$KAGGLE_DATASETS_DIR" "$EVM_STATE_DIR" "$COMMITTER_STATE_DIR"

echo "=== Starting genetics-l2-coordinator ==="
echo "    DB:       $COORDINATOR_DB"
echo "    Datasets: $KAGGLE_DATASETS_DIR"
"$BIN/genetics-l2-coordinator" \
  --db-path "$COORDINATOR_DB" \
  --listen 0.0.0.0:8091 \
  --scripts-dir "$SCRIPT_DIR/scripts" \
  --datasets-dir "$KAGGLE_DATASETS_DIR" \
  > /tmp/xenom-logs/coordinator.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting xenom-evm-node ==="
echo "    state: $EVM_STATE_DIR"
"$BIN/xenom-evm-node" \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --block-time 2000 \
  --state-dir "$EVM_STATE_DIR" \
  > /tmp/xenom-logs/evm-node.log 2>&1 &
PIDS+=($!)
sleep 2

STRATUM_THEME="genetics"

echo "=== Starting xenom-stratum-bridge (theme: $STRATUM_THEME) ==="
"$BIN/xenom-stratum-bridge" \
  --mining-address "$MINING_ADDR" \
  --rpcserver "$NODE_RPC" \
  --listen 0.0.0.0:5555 \
  --db-path /tmp/genetics-l2-nih2-pool.db \
  --l2-coordinator "$COORDINATOR" \
  --l2-theme "$STRATUM_THEME" \
  --devnet \
  > /tmp/xenom-logs/stratum-bridge.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-fetcher ==="
FETCHER_ARGS=(
  --coordinator "$COORDINATOR"
  --poll-secs 300
)
[[ $OPT_SRA      -eq 1 ]] && FETCHER_ARGS+=(--sra)     || true
[[ $OPT_IGSR     -eq 1 ]] && FETCHER_ARGS+=(--igsr)    || true
[[ $OPT_GNOMAD   -eq 1 ]] && FETCHER_ARGS+=(--gnomad)  || true
[[ $OPT_GDC      -eq 1 ]] && FETCHER_ARGS+=(--gdc)     || true
[[ $OPT_CLINVAR  -eq 1 ]] && FETCHER_ARGS+=(--clinvar) || true
ACTIVE_FETCHER_SRC=""
[[ $OPT_SRA     -eq 1 ]] && ACTIVE_FETCHER_SRC+="sra, "    || true
[[ $OPT_IGSR    -eq 1 ]] && ACTIVE_FETCHER_SRC+="igsr, "   || true
[[ $OPT_GNOMAD  -eq 1 ]] && ACTIVE_FETCHER_SRC+="gnomad, " || true
[[ $OPT_GDC     -eq 1 ]] && ACTIVE_FETCHER_SRC+="gdc, "    || true
[[ $OPT_CLINVAR -eq 1 ]] && ACTIVE_FETCHER_SRC+="clinvar"  || true
ACTIVE_FETCHER_SRC="${ACTIVE_FETCHER_SRC%, }"
echo "    sources: ${ACTIVE_FETCHER_SRC:-(none)}"
"$BIN/genetics-l2-fetcher" "${FETCHER_ARGS[@]}" \
  > /tmp/xenom-logs/fetcher.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-validator ==="
# Coordinator key file — generated by coordinator on first start at {db}.key
COORDINATOR_KEY_FILE="$COORDINATOR_DB.key"

"$BIN/genetics-l2-validator" \
  --coordinator "$COORDINATOR" \
  --coordinator-key-file "$COORDINATOR_KEY_FILE" \
  --poll-ms 10000 \
  --score-tolerance 0.05 \
  > /tmp/xenom-logs/validator.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-settlement ==="
"$BIN/genetics-l2-settlement" \
  --coordinator "$COORDINATOR" \
  --node "grpc://$NODE_RPC" \
  --submit \
  --devnet \
  --poll-ms 15000 \
  --quorum 1 \
  --score-tolerance 0.05 \
  > /tmp/xenom-logs/settlement.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting xenom-anchor-committer (L2 checkpoint → L1 tx.payload) ==="
echo "    evm-node:  http://127.0.0.1:8545"
echo "    l1-node:   grpc://$NODE_RPC"
echo "    state-dir: $COMMITTER_STATE_DIR"
if [[ "$ANCHOR_SUBMIT" -eq 1 ]]; then
  echo "    mode:      SUBMIT (live L1 anchor commits)"
else
  echo "    mode:      dry-run (no L1 submission — set Anchor Committer Key to enable)"
fi
COMMITTER_ARGS=(--evm-node 127.0.0.1:8545 --node "$NODE_RPC" --state-dir "$COMMITTER_STATE_DIR" --poll-ms 10000 --devnet)
[[ "$ANCHOR_SUBMIT" -eq 1 ]] && COMMITTER_ARGS+=(--submit) || true
"$BIN/xenom-anchor-committer" "${COMMITTER_ARGS[@]}" \
  > /tmp/xenom-logs/anchor-committer.log 2>&1 &
PIDS+=($!)
sleep 2

if [ "$GPU_AVAILABLE" -eq 1 ]; then
  echo "=== Starting genome-miner gpu (wgpu Vulkan/CUDA PoW + L2 GPU inline) ==="
  "$BIN/genome-miner" gpu \
    --devnet \
    --mining-address "$MINING_ADDR" \
    --stratum stratum+tcp://127.0.0.1:5555 \
    --no-tui \
    --l2-coordinator "$COORDINATOR" \
    --l2-gpu \
    > /tmp/xenom-logs/miner.log 2>&1 &
else
  echo "=== Starting genome-miner mine (PoW CPU + L2 inline) ==="
  "$BIN/genome-miner" mine \
    --devnet \
    --mining-address "$MINING_ADDR" \
    --stratum stratum+tcp://127.0.0.1:5555 \
    --no-tui \
    --l2-coordinator "$COORDINATOR" \
    > /tmp/xenom-logs/miner.log 2>&1 &
fi
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-worker (standalone L2 worker) ==="
L2_WORK_DIR="$XENOM_DATA/l2-work"
mkdir -p "$L2_WORK_DIR"
"$BIN/genetics-l2-worker" \
  --coordinator "$COORDINATOR" \
  --work-root "$L2_WORK_DIR" \
  --poll-ms 5000 \
  > /tmp/xenom-logs/l2-worker.log 2>&1 &
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
echo "  /tmp/xenom-logs/l2-worker.log"
echo "  /tmp/xenom-logs/anchor-committer.log"
echo ""
ACTIVE_SOURCES=""
[[ $OPT_SRA      -eq 1 ]] && ACTIVE_SOURCES+="sra, "
[[ $OPT_IGSR     -eq 1 ]] && ACTIVE_SOURCES+="igsr, "
[[ $OPT_GNOMAD   -eq 1 ]] && ACTIVE_SOURCES+="gnomad, "
[[ $OPT_GDC      -eq 1 ]] && ACTIVE_SOURCES+="gdc, "
[[ $OPT_CLINVAR  -eq 1 ]] && ACTIVE_SOURCES+="clinvar, "
ACTIVE_SOURCES="${ACTIVE_SOURCES%, }"

echo "=== Xenom L2 Genetics — running ==="
echo "Coordinator:  $COORDINATOR"
echo "Stratum:      stratum+tcp://127.0.0.1:5555 (theme: $STRATUM_THEME)"
echo "EVM RPC:      http://127.0.0.1:8545"
echo "Checkpoint:   xenom-anchor-committer (L2 block→L1 tx.payload, poll 10s)"
echo "Datasets:     ${ACTIVE_SOURCES:-none}"
echo "Pipeline:     variant_annotation_grch38 (GRCh38)"
echo ""
echo "Encryption:   ACTIVE"
echo "  ✓ Output files encrypted: /tmp/genome-miner-l2/{job_id}/output/*.enc"
echo "  ✓ Payloads encrypted with coordinator pubkey"
echo ""
echo "Logs (miner): tail -f /tmp/xenom-logs/miner.log"
echo "Logs (L2):    tail -f /tmp/xenom-logs/l2-worker.log"
echo "Pubkey:       curl $COORDINATOR/pubkey"
echo "Jobs:         curl $COORDINATOR/jobs"
echo ""
echo "Press Ctrl+C to stop everything."

wait

