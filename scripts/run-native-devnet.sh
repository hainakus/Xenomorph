#!/usr/bin/env bash
# Run a native (non-Docker) Xenomorph AI devnet: xeno-node, xeno-seed, xeno-miner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the Xenomorph devnet using locally built binaries (no Docker).

Options:
  -b, --build              Build release binaries before starting (default)
  --no-build               Skip cargo build
  -t, --trainer <name>     Miner trainer: mock, cpu, dnabert2
                           (default: \$XENO_MINER_TRAINER or mock)
  -d, --data-dir <dir>     Base data directory
                           (default: \$XENO_DATA_DIR or ./devnet-data-native)
  --anvil                  Start a local anvil instance for EVM/governance tests
  -q, --quiet              Minimal output
  -v, --verbose            Debug output
  -h, --help               Show this help and exit
"

BUILD=1
TRAINER="${XENO_MINER_TRAINER:-mock}"
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../devnet-data-native}"
START_ANVIL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -t|--trainer) TRAINER="$2"; shift 2 ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --anvil) START_ANVIL=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

if [[ "$TRAINER" != "mock" && "$TRAINER" != "cpu" && "$TRAINER" != "dnabert2" ]]; then
    err "Unknown trainer: $TRAINER. Use mock, cpu, or dnabert2."
    exit 1
fi

load_env

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Resolve absolute path for data directory.
DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
NODE_DATA_DIR="$DATA_DIR/node"
SEED_DATA_DIR="$DATA_DIR/models"
MINER_DATA_DIR="$DATA_DIR/miner"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$NODE_DATA_DIR" "$SEED_DATA_DIR" "$MINER_DATA_DIR" "$LOG_DIR"

BIN_PREFIX="${XENO_BIN_PREFIX:-$REPO_ROOT/target/release}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
    qlog "Building release binaries..."
    cargo build --release -p xenom -p seed-node -p xenom-miner
fi

require_command "$BIN_PREFIX/xenom"
require_command "$BIN_PREFIX/seed-node"
require_command "$BIN_PREFIX/xenom-miner"

# -----------------------------------------------------------------------------
# Port configuration
# -----------------------------------------------------------------------------
NODE_RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
NODE_P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
SEED_GRPC_PORT="${XENO_SEED_GRPC_PORT:-50051}"
MINER_WS_PORT="${XENO_MINER_RPC_PORT:-17110}"

if ! is_port_free "$NODE_RPC_PORT" || ! is_port_free "$NODE_P2P_PORT" || \
   ! is_port_free "$SEED_GRPC_PORT" || ! is_port_free "$MINER_WS_PORT"; then
    err "One or more required ports are already in use: $NODE_RPC_PORT, $NODE_P2P_PORT, $SEED_GRPC_PORT, $MINER_WS_PORT"
    exit 1
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
PIDS=()

port_open() {
    local port="$1"
    # shellcheck disable=SC2210
    (exec 3<>/dev/tcp/127.0.0.1/"$port") 2>/dev/null
}

wait_for_port() {
    local port="$1"
    local timeout_secs="${2:-60}"
    local pid="${3:-}"
    local i=0
    qlog "Waiting for port $port..."
    while (( i < timeout_secs )); do
        if port_open "$port"; then
            return 0
        fi
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            err "Process $pid exited while waiting for port $port"
            return 1
        fi
        sleep 1
        i=$((i + 1))
    done
    err "Timed out waiting for port $port to open"
    return 1
}

cleanup() {
    qlog "Stopping native devnet..."
    local pid
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done

    # Give the processes a few seconds to shut down cleanly, then SIGKILL.
    for pid in "${PIDS[@]}"; do
        local waited=0
        while kill -0 "$pid" 2>/dev/null && (( waited < 10 )); do
            sleep 1
            waited=$((waited + 1))
        done
        kill -9 "$pid" 2>/dev/null || true
    done
    exit 0
}

trap cleanup INT TERM EXIT

# -----------------------------------------------------------------------------
# Optional anvil for EVM/governance tests
# -----------------------------------------------------------------------------
if [[ "$START_ANVIL" == "1" ]]; then
    require_command anvil
    qlog "Starting anvil..."
    RUST_LOG="${RUST_LOG:-info}" anvil \
        --host=127.0.0.1 \
        --port="${XENO_ANVIL_PORT:-8545}" \
        --block-time="${XENO_BLOCK_TIME:-12}" \
        > "$LOG_DIR/anvil.log" 2>&1 &
    PIDS+=("$!")
    qlog "anvil started (pid $!)"
fi

# -----------------------------------------------------------------------------
# xeno-node
# -----------------------------------------------------------------------------
qlog "Starting xeno-node..."
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" \
    --devnet \
    --utxoindex \
    --appdir="$NODE_DATA_DIR" \
    --logdir="$LOG_DIR/node" \
    --rpclisten="0.0.0.0:$NODE_RPC_PORT" \
    --listen="0.0.0.0:$NODE_P2P_PORT" \
    --disable-upnp \
    --nodnsseed \
    > "$LOG_DIR/xeno-node.log" 2>&1 &
NODE_PID=$!
PIDS+=("$NODE_PID")
qlog "xeno-node started (pid $NODE_PID)"

wait_for_port "$NODE_RPC_PORT" 60 "$NODE_PID"

# -----------------------------------------------------------------------------
# xeno-seed (genome + model server, gRPC + miner WebSocket)
# -----------------------------------------------------------------------------
qlog "Starting xeno-seed..."
XENO_MODELS_DIR="$SEED_DATA_DIR" \
XENO_NODE_RPC="127.0.0.1:$NODE_RPC_PORT" \
XENO_GRPC_ADDR="0.0.0.0:$SEED_GRPC_PORT" \
XENO_MINER_WS_ADDR="0.0.0.0:$MINER_WS_PORT" \
XENO_DEFAULT_MODEL_ID="${XENO_DEFAULT_MODEL_ID:-multimolecule/dnabert2}" \
XENO_MODEL_KEY="${XENO_MODEL_KEY:-xenom-devnet-model-key}" \
RUST_LOG="${RUST_LOG:-info}" \
    "$BIN_PREFIX/seed-node" > "$LOG_DIR/xeno-seed.log" 2>&1 &
SEED_PID=$!
PIDS+=("$SEED_PID")
qlog "xeno-seed started (pid $SEED_PID)"

# The seed-node downloads the default model before opening the WebSocket port.
# Allow several minutes for the first download.
wait_for_port "$MINER_WS_PORT" 600 "$SEED_PID"

# -----------------------------------------------------------------------------
# xeno-miner
# -----------------------------------------------------------------------------
qlog "Starting xeno-miner (trainer=$TRAINER)..."
XENO_WALLET_PASSWORD="${XENO_WALLET_PASSWORD:-devnet-password}" \
RUST_LOG="${RUST_LOG:-info}" \
    "$BIN_PREFIX/xenom-miner" \
    --rpc-url "ws://127.0.0.1:$MINER_WS_PORT" \
    --model-id "${XENO_MINER_MODEL_ID:-multimolecule/dnabert2}" \
    --threads "${XENO_MINER_THREADS:-4}" \
    --trainer "$TRAINER" \
    --network devnet \
    --data-dir "$MINER_DATA_DIR" \
    --password "${XENO_WALLET_PASSWORD:-devnet-password}" \
    > "$LOG_DIR/xeno-miner.log" 2>&1 &
MINER_PID=$!
PIDS+=("$MINER_PID")
qlog "xeno-miner started (pid $MINER_PID)"

qok "Native devnet running. Logs in $LOG_DIR. Press Ctrl+C to stop."

# Wait for any background process. The trap will clean up on Ctrl+C.
wait "$NODE_PID" "$SEED_PID" "$MINER_PID" 2>/dev/null || true
