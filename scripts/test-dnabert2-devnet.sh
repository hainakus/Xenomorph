#!/usr/bin/env bash
# Run an end-to-end devnet test with the real DNABERT-2 trainer.
# This is a HITL (human-in-the-loop) test: it downloads the 110M model and
# performs real MLM training, so it needs a machine with at least ~8 GB RAM
# and several CPU cores.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the devnet with --trainer=dnabert2 and validate real training.

Options:
  -d, --duration <sec>   How long to wait for training metrics (default: 180)
  -t, --threads <n>      CPU threads for the miner (default: 4)
  --no-build             Skip image build
  --no-cleanup           Leave containers running after the test
  -q, --quiet            Minimal output
  -v, --verbose          Debug output
  -h, --help             Show this help and exit
"

DURATION=180
THREADS=4
NO_BUILD=0
NO_CLEANUP=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--duration) DURATION="$2"; shift 2 ;;
        -t|--threads) THREADS="$2"; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --no-cleanup) NO_CLEANUP=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

load_env
require_docker

export XENO_MINER_TRAINER=dnabert2
export XENO_MINER_THREADS="$THREADS"

qlog "Starting DNABERT-2 end-to-end devnet test"
qlog "Trainer: $XENO_MINER_TRAINER, threads: $XENO_MINER_THREADS, duration: ${DURATION}s"

# Ensure a clean state for reproducible metrics.
if [[ "$NO_CLEANUP" == "0" ]]; then
    qlog "Cleaning any existing devnet"
    "$SCRIPT_DIR/cleanup-devnet.sh" --volumes >/dev/null 2>&1 || true
fi

BUILD_FLAG=""
if [[ "$NO_BUILD" == "1" ]]; then
    BUILD_FLAG="--no-build"
fi

START_TIME=$(date +%s)

qlog "Starting devnet..."
"$SCRIPT_DIR/quick-devnet.sh" $BUILD_FLAG

# Wait until the miner container is running and producing logs.
qlog "Waiting for miner to start (max 60s)..."
for _ in $(seq 1 60); do
    if docker_compose ps -q xeno-miner &>/dev/null; then
        break
    fi
    sleep 1
done

if ! docker_compose ps -q xeno-miner &>/dev/null; then
    err "xeno-miner container did not start"
    exit 1
fi

# The seed-node needs time to download the DNABERT-2 model before it can serve
# checkpoints. Poll the miner logs for the "Loaded DNABERT-2" message.
qlog "Waiting for model download and trainer initialisation (timeout ${DURATION}s)..."
MODEL_LOADED=0
for _ in $(seq 1 "$DURATION"); do
    if docker_compose logs --no-log-prefix --tail=20 xeno-miner 2>/dev/null | grep -q "Loaded DNABERT-2 model checkpoint"; then
        MODEL_LOADED=1
        break
    fi
    sleep 1
done

if [[ "$MODEL_LOADED" == "0" ]]; then
    err "Miner did not load the DNABERT-2 checkpoint within ${DURATION}s"
    docker_compose logs --no-log-prefix --tail=50 xeno-miner || true
    exit 1
fi

# Wait for at least one block to be submitted.
qlog "Waiting for training blocks (timeout ${DURATION}s)..."
BLOCKS_BEFORE=0
for _ in $(seq 1 "$DURATION"); do
    BLOCKS_BEFORE="$(docker_compose logs --no-log-prefix --tail=100 xeno-miner 2>/dev/null | grep -c 'Submitted block' || true)"
    if [[ "$BLOCKS_BEFORE" -gt 0 ]]; then
        break
    fi
    sleep 1
done

if [[ "$BLOCKS_BEFORE" == "0" ]]; then
    err "Miner did not submit any blocks within ${DURATION}s"
    docker_compose logs --no-log-prefix --tail=100 xeno-miner || true
    exit 1
fi

# Allow a few more blocks to be produced so we can gather metrics.
qlog "Collecting metrics for 60s..."
sleep 60

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Extract metrics from miner logs.
MINER_LOGS="$(docker_compose logs --no-log-prefix xeno-miner 2>/dev/null || true)"
BLOCKS_TOTAL="$(echo "$MINER_LOGS" | grep -c 'Submitted block' || true)"
LOSS_LINES="$(echo "$MINER_LOGS" | grep -E 'loss [0-9]+\.[0-9]+->[0-9]+\.[0-9]+' || true)"
LAST_LOSS="$(echo "$MINER_LOGS" | grep -E 'loss [0-9]+\.[0-9]+->[0-9]+\.[0-9]+' | tail -n1 || true)"
MEMORY_STATS="$(docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}' xeno-miner 2>/dev/null | tail -n1 || true)"

if [[ -n "$LAST_LOSS" ]]; then
    ok "Real training loss observed: $LAST_LOSS"
else
    warn "Could not find explicit loss values in miner logs; check RUST_LOG level"
fi

ok "Blocks submitted: $BLOCKS_TOTAL in ${ELAPSED}s"

# JSON summary for CI.
if [[ "${XENO_QUIET:-0}" == "1" ]]; then
    echo "{\"blocks\":$BLOCKS_TOTAL,\"elapsed_seconds\":$ELAPSED,\"last_loss\":\"${LAST_LOSS//\"/\\\"}\",\"memory\":\"${MEMORY_STATS//\"/\\\"}\"}"
else
    echo "--- DNABERT-2 devnet test summary ---"
    echo "Blocks submitted: $BLOCKS_TOTAL"
    echo "Elapsed time:     ${ELAPSED}s"
    echo "Blocks/min:       $(awk "BEGIN {printf \"%.2f\", $BLOCKS_TOTAL * 60 / $ELAPSED}")"
    echo "Last loss log:    $LAST_LOSS"
    echo "Memory (xeno-miner): $MEMORY_STATS"
    echo "-------------------------------------"
fi

if [[ "$NO_CLEANUP" == "0" ]]; then
    qlog "Cleaning up devnet"
    "$SCRIPT_DIR/cleanup-devnet.sh" --volumes >/dev/null 2>&1 || true
else
    qlog "Leaving devnet running (--no-cleanup)"
fi

ok "DNABERT-2 end-to-end devnet test passed"
