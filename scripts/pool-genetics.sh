#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# pool-genetics.sh - Build and start the Genetics L2 pool
#
# Starts 3 processes:
#   1. genetics-l2-coordinator  (REST API for job management)
#   2. xenom-stratum-bridge     (Stratum pool + L2 dispatcher)
#   3. genetics-l2-fetcher      (seeds jobs - runs once then exits)
#
# Usage:
#   ./scripts/pool-genetics.sh [options]
#
# Options:
#   --mining-address ADDR   Pool reward address (required)
#   --rpc HOST:PORT         Node gRPC endpoint (default: 127.0.0.1:18610)
#   --coordinator-port N    L2 coordinator port (default: 8091)
#   --stratum-port N        Stratum listen port (default: 5555)
#   --db PATH               Coordinator DB path (default: /tmp/genetics-l2.db)
#   --build                 Force rebuild of all binaries
#
#   Dataset sources (fetcher flags):
#   --competition SLUG      Kaggle competition to seed (default: birdclef-2026)
#   --kaggle-key USER:TOKEN Kaggle API key (overrides KAGGLE_KEY env / ~/.kaggle/kaggle.json)
#   --genomics              Enable all 5 genomics sources (SRA + IGSR + gnomAD + GDC + ClinVar)
#   --sra                   Seed NCBI SRA GRCh38 VCF jobs (GIAB benchmarks)
#   --igsr                  Seed 1000 Genomes / IGSR GRCh38 30x phased VCF jobs
#   --gnomad                Seed gnomAD v4.1 exome VCF allele-frequency jobs
#   --gdc                   Fetch open-access TCGA/GDC cancer cohort jobs
#   --clinvar               Seed ClinVar weekly GRCh38 VCF clinical annotation jobs
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO_ROOT/target/debug"
RELEASE_BIN="$REPO_ROOT/target/release"

# ── Defaults ──────────────────────────────────────────────────────────────────
MINING_ADDRESS=""
NODE_RPC="127.0.0.1:18610"
COORDINATOR_PORT="8091"
STRATUM_PORT="5555"
COMPETITION="birdclef-2026"
DB_PATH="/tmp/genetics-l2.db"
KAGGLE_KEY="${KAGGLE_KEY:-}"
BUILD=0

# Dataset source flags (all off by default; --genomics enables all 5)
OPT_SRA=0
OPT_IGSR=0
OPT_GNOMAD=0
OPT_GDC=0
OPT_CLINVAR=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mining-address)   MINING_ADDRESS="$2"; shift 2 ;;
    --rpc)              NODE_RPC="$2";        shift 2 ;;
    --coordinator-port) COORDINATOR_PORT="$2"; shift 2 ;;
    --stratum-port)     STRATUM_PORT="$2";    shift 2 ;;
    --competition)      COMPETITION="$2";     shift 2 ;;
    --db)               DB_PATH="$2";         shift 2 ;;
    --kaggle-key)       KAGGLE_KEY="$2";      shift 2 ;;
    --build)            BUILD=1;              shift 1 ;;
    --genomics)         OPT_SRA=1; OPT_IGSR=1; OPT_GNOMAD=1; OPT_GDC=1; OPT_CLINVAR=1; shift 1 ;;
    --sra)              OPT_SRA=1;            shift 1 ;;
    --igsr)             OPT_IGSR=1;           shift 1 ;;
    --gnomad)           OPT_GNOMAD=1;         shift 1 ;;
    --gdc)              OPT_GDC=1;            shift 1 ;;
    --clinvar)          OPT_CLINVAR=1;        shift 1 ;;
    *) shift 1 ;;
  esac
done

if [[ -z "$MINING_ADDRESS" ]]; then
  echo "ERROR: --mining-address is required"
  echo "Usage: $0 --mining-address xenom:qYOURPOOLADDRESS [options]"
  exit 1
fi

COORDINATOR_URL="http://localhost:$COORDINATOR_PORT"

# ── Build ─────────────────────────────────────────────────────────────────────
if [[ $BUILD -eq 1 ]] || \
   [[ ! -f "$BIN/genetics-l2-coordinator" ]] || \
   [[ ! -f "$BIN/xenom-stratum-bridge" ]] || \
   [[ ! -f "$BIN/genetics-l2-fetcher" ]]; then
  echo "[pool] Building binaries..."
  cargo build \
    -p genetics-l2-coordinator \
    -p xenom-stratum-bridge \
    -p genetics-l2-fetcher \
    --manifest-path "$REPO_ROOT/Cargo.toml"
fi

# ── Cleanup on exit ───────────────────────────────────────────────────────────
PIDS=()
cleanup() {
  echo ""
  echo "[pool] Shutting down..."
  for pid in "${PIDS[@]+${PIDS[@]}}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait
  echo "[pool] Done."
}
trap cleanup INT TERM EXIT

# ── 1. Coordinator ────────────────────────────────────────────────────────────
if lsof -i ":$COORDINATOR_PORT" -sTCP:LISTEN &>/dev/null; then
  echo "[pool] genetics-l2-coordinator already running on :$COORDINATOR_PORT - skipping."
else
  echo "[pool] Starting genetics-l2-coordinator on :$COORDINATOR_PORT..."
  "$BIN/genetics-l2-coordinator" \
    --db-path "$DB_PATH" \
    --listen "0.0.0.0:$COORDINATOR_PORT" \
    &
  PIDS+=($!)
  sleep 1
fi

# ── 2. Stratum bridge ─────────────────────────────────────────────────────────
if lsof -i ":$STRATUM_PORT" -sTCP:LISTEN &>/dev/null; then
  echo "[pool] xenom-stratum-bridge already running on :$STRATUM_PORT - skipping."
else
  echo "[pool] Starting xenom-stratum-bridge on :$STRATUM_PORT..."
  "$BIN/xenom-stratum-bridge" \
    --mining-address "$MINING_ADDRESS" \
    --rpcserver "$NODE_RPC" \
    --listen "0.0.0.0:$STRATUM_PORT" \
    --l2-coordinator "$COORDINATOR_URL" \
    --l2-theme genetics \
    --pool-name "Xenom Genetics Pool" \
    --devnet \
    &
  PIDS+=($!)
  sleep 1
fi

# ── 3. Fetcher (seed jobs - once) ─────────────────────────────────────────────
FETCHER_ARGS=(--coordinator "$COORDINATOR_URL" --competition "$COMPETITION")
[[ -n "$KAGGLE_KEY" ]] && FETCHER_ARGS+=(--kaggle-key "$KAGGLE_KEY")
[[ $OPT_SRA    -eq 1 ]] && FETCHER_ARGS+=(--sra)
[[ $OPT_IGSR   -eq 1 ]] && FETCHER_ARGS+=(--igsr)
[[ $OPT_GNOMAD -eq 1 ]] && FETCHER_ARGS+=(--gnomad)
[[ $OPT_GDC    -eq 1 ]] && FETCHER_ARGS+=(--gdc)
[[ $OPT_CLINVAR -eq 1 ]] && FETCHER_ARGS+=(--clinvar)

echo "[pool] Seeding jobs via genetics-l2-fetcher (args: ${FETCHER_ARGS[*]})..."
"$BIN/genetics-l2-fetcher" "${FETCHER_ARGS[@]}" && echo "[pool] Seed complete."

# ── Summary ───────────────────────────────────────────────────────────────────
ACTIVE_SOURCES="kaggle:$COMPETITION"
[[ $OPT_SRA    -eq 1 ]] && ACTIVE_SOURCES+=", sra"
[[ $OPT_IGSR   -eq 1 ]] && ACTIVE_SOURCES+=", igsr"
[[ $OPT_GNOMAD -eq 1 ]] && ACTIVE_SOURCES+=", gnomad"
[[ $OPT_GDC    -eq 1 ]] && ACTIVE_SOURCES+=", gdc"
[[ $OPT_CLINVAR -eq 1 ]] && ACTIVE_SOURCES+=", clinvar"

POOL_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || echo '127.0.0.1')"

echo ""
echo "══════════════════════════════════════════════════"
echo " Genetics L2 Pool running"
echo "══════════════════════════════════════════════════"
echo " Stratum       stratum+tcp://$POOL_IP:$STRATUM_PORT"
echo " Coordinator   $COORDINATOR_URL"
echo " Datasets      $ACTIVE_SOURCES"
echo " Pool address  $MINING_ADDRESS"
echo "══════════════════════════════════════════════════"
echo ""
echo "Miner command:"
echo "  ./genome-miner mine --devnet \\"
echo "    --mining-address <YOUR_ADDRESS> \\"
echo "    --stratum stratum+tcp://$POOL_IP:$STRATUM_PORT \\"
echo "    --l2-coordinator $COORDINATOR_URL \\"
echo "    --l2-private-key <HEX_KEY>"
echo ""
echo "Press Ctrl+C to stop."
wait
