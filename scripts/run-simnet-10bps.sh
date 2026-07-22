#!/usr/bin/env bash
# Run a 10-BPS simnet stress test with mock miners.
#
# This starts two connected simnet nodes (so `is_synced` is satisfied) and one
# or more `--trainer mock` miners. Because the simnet target block time is 100 ms,
# the network will attempt to reach 10 blocks per second; the DAA will raise the
# difficulty as the mock miners push blocks in. Useful for observing difficulty
# variation under load.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run a 10 BPS simnet stress test:
  - two connected xeno-node processes (node1 mines, node2 is a peer)
  - N mock miners submitting blocks as fast as possible

Options:
  -b, --build                     Build release binaries before starting (default)
  --no-build                      Skip cargo build
  -n, --miners <n>                Number of mock miners to start (default: 1)
  -d, --data-dir <dir>            Base data directory
                                  (default: \$XENO_DATA_DIR or ./simnet-data-native)
  --duration <secs>               Stop miners after this many seconds (default: 60)
  -q, --quiet                     Minimal output
  -v, --verbose                   Debug output
  -h, --help                      Show this help and exit
"

BUILD=1
NUM_MINERS=1
DURATION=60
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../simnet-data-native}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -n|--miners) NUM_MINERS="$2"; shift 2 ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

load_env

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
NODE1_DATA_DIR="$DATA_DIR/node1"
NODE2_DATA_DIR="$DATA_DIR/node2"
MODELS_DIR="$DATA_DIR/models"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$NODE1_DATA_DIR" "$NODE2_DATA_DIR" "$MODELS_DIR" "$LOG_DIR"

BIN_PREFIX="${XENO_BIN_PREFIX:-$REPO_ROOT/target/release}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
    qlog "Building release binaries (xenom, xenom-miner)..."
    cargo build --release -p xenom
    cargo build --release -p xenom-miner
fi

require_command "$BIN_PREFIX/xenom"
require_command "$BIN_PREFIX/xenom-miner"

# -----------------------------------------------------------------------------
# Port configuration
# -----------------------------------------------------------------------------
NODE1_RPC_PORT=16110
NODE1_P2P_PORT=16111
NODE1_MINER_WS_PORT=17110
NODE2_RPC_PORT=16120
NODE2_P2P_PORT=16121

if ! is_port_free "$NODE1_RPC_PORT" || ! is_port_free "$NODE1_P2P_PORT" || ! is_port_free "$NODE1_MINER_WS_PORT" || \
   ! is_port_free "$NODE2_RPC_PORT" || ! is_port_free "$NODE2_P2P_PORT"; then
    err "One or more required ports are already in use"
    exit 1
fi

PIDS=()

port_open() {
    local port="$1"
    (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null
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
            err "Process $pid exited while waiting for $port"
            return 1
        fi
        sleep 1
        i=$((i + 1))
    done
    err "Timed out waiting for $port to open"
    return 1
}

wait_for_peer() {
    local port="$1"
    local timeout_secs="${2:-60}"
    local i=0
    qlog "Waiting for at least one peer on node RPC $port..."
    while (( i < timeout_secs )); do
        local peers
        peers="$(timeout 5 curl -s -X POST "http://127.0.0.1:$port" \
            -H 'Content-Type: application/json' \
            -d '{"jsonrpc":"2.0","method":"getConnectedPeerInfo","params":{},"id":1}' 2>/dev/null | jq '.result | length' 2>/dev/null || echo 0)"
        if [[ "$peers" -ge 1 ]]; then
            return 0
        fi
        sleep 1
        i=$((i + 1))
    done
    err "Timed out waiting for peer connection"
    return 1
}

cleanup() {
    qlog "Stopping simnet 10-BPS test..."
    local pid
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
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
# Node 1: full node with miner WebSocket
# -----------------------------------------------------------------------------
qlog "Starting simnet node 1 (RPC $NODE1_RPC_PORT, P2P $NODE1_P2P_PORT, miner WS $NODE1_MINER_WS_PORT)..."
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" \
    --simnet \
    --utxoindex \
    --appdir="$NODE1_DATA_DIR" \
    --logdir="$LOG_DIR/node1" \
    --rpclisten="0.0.0.0:$NODE1_RPC_PORT" \
    --listen="0.0.0.0:$NODE1_P2P_PORT" \
    --miner-ws-listen="0.0.0.0:$NODE1_MINER_WS_PORT" \
    --models-dir="$MODELS_DIR" \
    --disable-upnp \
    --nodnsseed \
    > "$LOG_DIR/node1.log" 2>&1 &
NODE1_PID=$!
PIDS+=("$NODE1_PID")
qlog "node1 started (pid $NODE1_PID)"

wait_for_port "$NODE1_RPC_PORT" 120 "$NODE1_PID"
# The miner WebSocket listener is bound only after the active model is downloaded,
# which can take several minutes on the first run.
wait_for_port "$NODE1_MINER_WS_PORT" 600 "$NODE1_PID"

# -----------------------------------------------------------------------------
# Node 2: peer only (no miner WebSocket, no model download)
# -----------------------------------------------------------------------------
qlog "Starting simnet node 2 as peer (P2P $NODE2_P2P_PORT -> 127.0.0.1:$NODE1_P2P_PORT)..."
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" \
    --simnet \
    --utxoindex \
    --appdir="$NODE2_DATA_DIR" \
    --logdir="$LOG_DIR/node2" \
    --rpclisten="0.0.0.0:$NODE2_RPC_PORT" \
    --listen="0.0.0.0:$NODE2_P2P_PORT" \
    --connect="127.0.0.1:$NODE1_P2P_PORT" \
    --addpeer="127.0.0.1:$NODE1_P2P_PORT" \
    --disable-upnp \
    --nodnsseed \
    > "$LOG_DIR/node2.log" 2>&1 &
NODE2_PID=$!
PIDS+=("$NODE2_PID")
qlog "node2 started (pid $NODE2_PID)"

wait_for_port "$NODE2_RPC_PORT" 120 "$NODE2_PID"
wait_for_peer "$NODE1_RPC_PORT" 120

# -----------------------------------------------------------------------------
# Mock miners
# -----------------------------------------------------------------------------
qlog "Starting $NUM_MINERS mock miner(s) against ws://127.0.0.1:$NODE1_MINER_WS_PORT..."
for i in $(seq 1 "$NUM_MINERS"); do
    MINER_DATA_DIR="$DATA_DIR/miner$i"
    mkdir -p "$MINER_DATA_DIR"
    XENO_WALLET_PASSWORD="${XENO_WALLET_PASSWORD:-simnet-password}" \
    RUST_LOG="${RUST_LOG:-info}" \
        "$BIN_PREFIX/xenom-miner" \
        --rpc-url "ws://127.0.0.1:$NODE1_MINER_WS_PORT" \
        --model-id "${XENO_MINER_MODEL_ID:-multimolecule/dnabert2}" \
        --threads "${XENO_MINER_THREADS:-4}" \
        --trainer mock \
        --network simnet \
        --data-dir "$MINER_DATA_DIR" \
        --password "${XENO_WALLET_PASSWORD:-simnet-password}" \
        > "$LOG_DIR/xeno-miner-$i.log" 2>&1 &
    MINER_PID=$!
    PIDS+=("$MINER_PID")
    qlog "xeno-miner $i started (pid $MINER_PID)"
done

qok "Simnet 10-BPS stress test running. Logs in $LOG_DIR."
qok "  Node1 RPC:  http://127.0.0.1:$NODE1_RPC_PORT"
qok "  Node1 miner WebSocket: ws://127.0.0.1:$NODE1_MINER_WS_PORT"
qok "  Node2 RPC:  http://127.0.0.1:$NODE2_RPC_PORT"
qok "  Miners:     $NUM_MINERS"
qok "  Press Ctrl+C to stop."

if [[ "$DURATION" -gt 0 ]]; then
    qlog "Running for $DURATION seconds..."
    sleep "$DURATION"
    qlog "Duration elapsed, stopping..."
    cleanup
else
    wait "$NODE1_PID" 2>/dev/null || true
fi
