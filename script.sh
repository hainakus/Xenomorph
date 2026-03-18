#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Xenom L2 Genetics Devnet Setup ==="
echo "L2 Work Encryption: ENABLED"
echo "Primary:   Genomics — VCF → normalization → annotation → gene scoring → cohort grouping"
echo "Optional:  BirdCLEF acoustic classification (GPU/CUDA)"
echo "Reference: GRCh38"
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
export MINING_ADDR
export NODE_RPC
export COORDINATOR
export KAGGLE_KEY

# Export private key as env vars — never pass as CLI args (ps aux / /proc/$PID/cmdline exposure)
export SETTLEMENT_PRIVKEY="$PRIVKEY"
export VALIDATOR_PRIVKEY="$PRIVKEY"
export L2_PRIVKEY="$PRIVKEY"
unset PRIVKEY  # clear original after mapping to named vars

echo ""
echo "=== Environment ==="
echo "BIN=$BIN"
echo "MINING_ADDR=$MINING_ADDR"
echo "NODE_RPC=$NODE_RPC"
echo "COORDINATOR=$COORDINATOR"
echo ""
echo "=== Configuration ==="
echo "Primary:   Genomics (GRCh38 VCF → Ensembl VEP → gene scores)"
echo "Optional:  BirdCLEF-2026 acoustic classification (YAMNet/Perch, GPU)"
echo "Encryption: ENABLED (ChaCha20-Poly1305 local + ECIES submit)"
echo ""

# Kaggle API key setup (for coordinator only - miners will download from coordinator API)
KAGGLE_DIR="$HOME/.kaggle"
KAGGLE_JSON="$KAGGLE_DIR/kaggle.json"

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
echo "--- Acoustic (optional) ---"
read -r -p "Enable BirdCLEF-2026 (Kaggle acoustic classification)  ? [y/N]: " _ans
OPT_BIRDCLEF=0; [[ "$_ans" =~ ^[Yy]$ ]] && OPT_BIRDCLEF=1
echo ""

if [[ $OPT_BIRDCLEF -eq 1 ]]; then
  if [ ! -f "$KAGGLE_JSON" ]; then
    echo "=== Kaggle API Setup ==="
    echo "BirdCLEF requires Kaggle credentials to download the dataset."
    echo ""
    read -p "Do you have a Kaggle API key? (y/n): " has_key
    if [ "$has_key" = "y" ] || [ "$has_key" = "Y" ]; then
      echo ""
      read -p "Kaggle username: " kaggle_user
      read -p "Kaggle API key: "  kaggle_key
      mkdir -p "$KAGGLE_DIR"
      cat > "$KAGGLE_JSON" <<EOF
{"username":"$kaggle_user","key":"$kaggle_key"}
EOF
      chmod 600 "$KAGGLE_JSON"
      echo "✓ Kaggle credentials saved to $KAGGLE_JSON"
    else
      echo "No Kaggle key — disabling BirdCLEF."
      OPT_BIRDCLEF=0
    fi
    echo ""
  else
    echo "✓ Kaggle credentials found at $KAGGLE_JSON"
  fi
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
  
  # Acoustic classification (YAMNet/Perch) + genomics pipeline (genome_annotate.py)
  echo "Installing: tensorflow, torch, torchaudio, timm, numpy, pandas, soundfile, requests..."
  pip install --quiet tensorflow tensorflow-hub numpy soundfile torch torchaudio timm pandas requests || echo "Warning: Some packages failed to install"
  
  echo "Note: YAMNet will use GPU if CUDA is available, otherwise CPU"
else
  echo "Warning: venv not available, using system Python"
  if command -v pip3 &> /dev/null; then
    pip3 install --quiet --break-system-packages kaggle kagglehub tensorflow numpy torch torchaudio timm pandas requests 2>/dev/null || \
    pip3 install --quiet --user kaggle kagglehub tensorflow numpy torch torchaudio timm pandas requests 2>/dev/null || \
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

# Persistent data directory (survives reboots)
XENOM_DATA="${XENOM_DATA_DIR:-$HOME/.local/share/xenom}"
KAGGLE_DATASETS_DIR="$XENOM_DATA/kaggle-datasets"
COORDINATOR_DB="$XENOM_DATA/genetics-l2.db"
mkdir -p "$KAGGLE_DATASETS_DIR"

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

# Pre-download BirdCLEF-2026 dataset (only when BirdCLEF source is enabled)
if [[ $OPT_BIRDCLEF -eq 1 ]]; then
  BIRDCLEF_CACHE="$KAGGLE_DATASETS_DIR/_cache/birdclef-2026"
  mkdir -p "$BIRDCLEF_CACHE"
  AUDIO_COUNT=$(find "$BIRDCLEF_CACHE" \( -name '*.ogg' -o -name '*.wav' -o -name '*.mp3' \) 2>/dev/null | wc -l | tr -d ' ')
  if [ "$AUDIO_COUNT" -gt "0" ]; then
    echo "=== BirdCLEF-2026 already cached: $AUDIO_COUNT audio files ==="
  elif [ -f "$BIRDCLEF_CACHE/.ready" ]; then
    echo "=== BirdCLEF-2026 cache marked ready ==="
  else
    echo "=== Downloading BirdCLEF-2026 dataset → $BIRDCLEF_CACHE ==="
    LOCK_FILE="$KAGGLE_DATASETS_DIR/_birdclef-2026.lock"
    if [ ! -f "$LOCK_FILE" ]; then
      touch "$LOCK_FILE"
      if kaggle competitions download -c birdclef-2026 --path "$BIRDCLEF_CACHE/" 2>&1 | tee -a /tmp/xenom-logs/coordinator.log; then
        echo "Extracting BirdCLEF-2026..."
        for z in "$BIRDCLEF_CACHE"/*.zip; do
          [ -f "$z" ] && unzip -n -q "$z" -d "$BIRDCLEF_CACHE/" && rm -f "$z"
        done
        touch "$BIRDCLEF_CACHE/.ready"
        AUDIO_COUNT=$(find "$BIRDCLEF_CACHE" \( -name '*.ogg' -o -name '*.wav' \) | wc -l | tr -d ' ')
        echo "BirdCLEF-2026 ready: $AUDIO_COUNT audio files"
      else
        echo "Warning: BirdCLEF-2026 download failed — skipping acoustic jobs"
        OPT_BIRDCLEF=0
      fi
      rm -f "$LOCK_FILE"
    else
      echo "Warning: download lock exists, skipping BirdCLEF"
    fi
  fi
fi

echo "=== Starting xenom-evm-node ==="
"$BIN/xenom-evm-node" \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --block-time 2000 \
  > /tmp/xenom-logs/evm-node.log 2>&1 &
PIDS+=($!)
sleep 2

STRATUM_THEME="genetics"
[[ $OPT_BIRDCLEF -eq 1 ]] && STRATUM_THEME="birdclef"

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
[[ $OPT_BIRDCLEF -eq 1 ]] && FETCHER_ARGS+=(--competition birdclef-2026)
[[ $OPT_BIRDCLEF -eq 0 ]] && FETCHER_ARGS+=(--kaggle-only)  # suppress kaggle if not opted in
[[ $OPT_SRA      -eq 1 ]] && FETCHER_ARGS+=(--sra)
[[ $OPT_IGSR     -eq 1 ]] && FETCHER_ARGS+=(--igsr)
[[ $OPT_GNOMAD   -eq 1 ]] && FETCHER_ARGS+=(--gnomad)
[[ $OPT_GDC      -eq 1 ]] && FETCHER_ARGS+=(--gdc)
[[ $OPT_CLINVAR  -eq 1 ]] && FETCHER_ARGS+=(--clinvar)
ACTIVE_FETCHER_SRC="$([ $OPT_BIRDCLEF -eq 1 ] && echo 'birdclef-2026, ')$([ $OPT_SRA -eq 1 ] && echo 'sra, ')$([ $OPT_IGSR -eq 1 ] && echo 'igsr, ')$([ $OPT_GNOMAD -eq 1 ] && echo 'gnomad, ')$([ $OPT_GDC -eq 1 ] && echo 'gdc, ')$([ $OPT_CLINVAR -eq 1 ] && echo 'clinvar')"
echo "    sources: ${ACTIVE_FETCHER_SRC%, }"
"$BIN/genetics-l2-fetcher" "${FETCHER_ARGS[@]}" \
  > /tmp/xenom-logs/fetcher.log 2>&1 &
PIDS+=($!)
sleep 2

echo "=== Starting genetics-l2-validator ==="
# Validator needs coordinator privkey to decrypt encrypted results
COORDINATOR_PRIVKEY_FILE="/tmp/genetics-l2-nih2.db.key"
if [ -f "$COORDINATOR_PRIVKEY_FILE" ]; then
  export COORDINATOR_PRIVKEY
  COORDINATOR_PRIVKEY=$(cat "$COORDINATOR_PRIVKEY_FILE")
fi

"$BIN/genetics-l2-validator" \
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
ACTIVE_SOURCES=""
[[ $OPT_SRA      -eq 1 ]] && ACTIVE_SOURCES+="sra, "
[[ $OPT_IGSR     -eq 1 ]] && ACTIVE_SOURCES+="igsr, "
[[ $OPT_GNOMAD   -eq 1 ]] && ACTIVE_SOURCES+="gnomad, "
[[ $OPT_GDC      -eq 1 ]] && ACTIVE_SOURCES+="gdc, "
[[ $OPT_CLINVAR  -eq 1 ]] && ACTIVE_SOURCES+="clinvar, "
[[ $OPT_BIRDCLEF -eq 1 ]] && ACTIVE_SOURCES+="birdclef-2026"
ACTIVE_SOURCES="${ACTIVE_SOURCES%, }"

echo "=== Xenom L2 Genetics — running ==="
echo "Coordinator:  $COORDINATOR"
echo "Stratum:      stratum+tcp://127.0.0.1:5555 (theme: $STRATUM_THEME)"
echo "EVM RPC:      http://127.0.0.1:8545"
echo "Datasets:     ${ACTIVE_SOURCES:-none}"
echo "Pipeline:     variant_annotation_grch38 (GRCh38)"
echo ""
echo "Encryption:   ACTIVE"
echo "  ✓ Output files encrypted: /tmp/genome-miner-l2/{job_id}/output/*.enc"
echo "  ✓ Payloads encrypted with coordinator pubkey"
echo ""
echo "Logs:         tail -f /tmp/xenom-logs/miner.log"
echo "Pubkey:       curl $COORDINATOR/pubkey"
echo "Jobs:         curl $COORDINATOR/jobs"
echo ""
echo "Press Ctrl+C to stop everything."

wait
