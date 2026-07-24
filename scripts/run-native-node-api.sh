#!/usr/bin/env bash
# Run a Xenomorph devnet node + seed-node inference gRPC + OpenAI API gateway
# on a single server. No miner is started here; miners connect from other rigs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the Xenomorph node, seed-node (inference gRPC), and API gateway on one
host using locally built binaries. External miners connect to this node via
--miner-ws-listen.

Options:
  -b, --build                     Build release binaries before starting (default)
  --no-build                      Skip cargo build
  -d, --data-dir <dir>            Base data directory
                                  (default: \$XENO_DATA_DIR or ./devnet-data-native)
  --genome-file <path>            Path to .xenom packed GRCh38 genome file
                                  (optional; enables real Genome PoW)
  --bind-ip <ip>                  IP to bind node/seed/API sockets to
                                  (default: \$XENO_BIND_IP or 0.0.0.0)
  -q, --quiet                     Minimal output
  -v, --verbose                   Debug output
  -h, --help                      Show this help and exit
"

BUILD=1
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../devnet-data-native}"
GENOME_FILE="${XENO_GENOME_FILE:-}"
BIND_IP="${XENO_BIND_IP:-0.0.0.0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --genome-file) GENOME_FILE="$2"; shift 2 ;;
        --bind-ip) BIND_IP="$2"; shift 2 ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

load_env

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

# Honour XENO_BIND_IP from .env/env, but keep --bind-ip as fallback.
BIND_IP="${XENO_BIND_IP:-$BIND_IP}"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
NODE_DATA_DIR="$DATA_DIR/node"
SEED_DATA_DIR="$DATA_DIR/models"
REDIS_DATA_DIR="$DATA_DIR/redis"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$NODE_DATA_DIR" "$SEED_DATA_DIR" "$REDIS_DATA_DIR" "$LOG_DIR"

BIN_PREFIX="${XENO_BIN_PREFIX:-$REPO_ROOT/target/release}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
    qlog "Building release binaries (node, seed-node, api-gateway)..."
    cargo clean -p xenom
    cargo build --release -p xenom
    cargo clean -p seed-node
    cargo build --release -p seed-node
    cargo clean -p api-gateway
    cargo build --release -p api-gateway
fi

require_command "$BIN_PREFIX/xenom"
require_command "$BIN_PREFIX/seed-node"
require_command "$BIN_PREFIX/api-gateway"

# -----------------------------------------------------------------------------
# Port configuration
# -----------------------------------------------------------------------------
NODE_RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
NODE_P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
MINER_WS_PORT="${XENO_MINER_WS_PORT:-17110}"
SEED_GRPC_PORT="${XENO_SEED_GRPC_PORT:-50051}"
SEED_MINER_WS_PORT="${XENO_SEED_MINER_WS_PORT:-17111}"
API_PORT="${XENO_API_PORT:-3000}"
REDIS_PORT="${XENO_REDIS_PORT:-6379}"
MODEL_ID="${XENO_MODEL_ID:-multimolecule/dnabert2}"

# When binding to a specific IP, health-check waits on that IP; otherwise use localhost.
WAIT_HOST="127.0.0.1"
if [[ "$BIND_IP" != "0.0.0.0" && "$BIND_IP" != "127.0.0.1" ]]; then
    WAIT_HOST="$BIND_IP"
fi

# Local clients connect via localhost when binding to all interfaces, otherwise via the bound IP.
CONNECT_HOST="127.0.0.1"
if [[ "$BIND_IP" != "0.0.0.0" && "$BIND_IP" != "127.0.0.1" ]]; then
    CONNECT_HOST="$BIND_IP"
fi

if ! is_port_free "$NODE_RPC_PORT" || ! is_port_free "$NODE_P2P_PORT" || \
   ! is_port_free "$MINER_WS_PORT" || ! is_port_free "$SEED_GRPC_PORT" || \
   ! is_port_free "$SEED_MINER_WS_PORT" || ! is_port_free "$API_PORT"; then
    err "One or more required ports are already in use"
    exit 1
fi

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
PIDS=()

port_open() {
    local host="${1:-127.0.0.1}"
    local port="$2"
    (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null
}

wait_for_port() {
    local host="${1:-127.0.0.1}"
    local port="$2"
    local timeout_secs="${3:-60}"
    local pid="${4:-}"
    local i=0
    qlog "Waiting for $host:$port..."
    while (( i < timeout_secs )); do
        if port_open "$host" "$port"; then
            return 0
        fi
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
            err "Process $pid exited while waiting for $host:$port"
            return 1
        fi
        sleep 1
        i=$((i + 1))
    done
    err "Timed out waiting for $host:$port to open"
    return 1
}

cleanup() {
    qlog "Stopping node / seed / api-gateway..."
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
    if [[ -n "$REDIS_PID" ]]; then
        kill "$REDIS_PID" 2>/dev/null || true
    fi
    exit 0
}

trap cleanup INT TERM EXIT

# -----------------------------------------------------------------------------
# Redis for api-gateway
# -----------------------------------------------------------------------------
REDIS_URL="${REDIS_URL:-}"
REDIS_PID=""
if [[ -z "$REDIS_URL" ]]; then
    if has_command redis-server; then
        qlog "Starting local Redis on port $REDIS_PORT..."
        redis-server --port "$REDIS_PORT" --dir "$REDIS_DATA_DIR" --loglevel warning \
            > "$LOG_DIR/redis.log" 2>&1 &
        REDIS_PID=$!
        REDIS_URL="redis://127.0.0.1:$REDIS_PORT"
        wait_for_port 127.0.0.1 "$REDIS_PORT" 30 "$REDIS_PID"
    else
        err "REDIS_URL is not set and redis-server is not installed. Set REDIS_URL or install redis-server."
        exit 1
    fi
fi
export REDIS_URL

# -----------------------------------------------------------------------------
# xeno-node (full node + training coordinator / miner websocket)
# -----------------------------------------------------------------------------
qlog "Starting xeno-node..."
NODE_ARGS=(
    --devnet
    --utxoindex
    --appdir="$NODE_DATA_DIR"
    --logdir="$LOG_DIR/node"
    --rpclisten="$BIND_IP:$NODE_RPC_PORT"
    --listen="$BIND_IP:$NODE_P2P_PORT"
    --miner-ws-listen="$BIND_IP:$MINER_WS_PORT"
    --models-dir="$SEED_DATA_DIR"
    --disable-upnp
    --nodnsseed
)
[[ -n "$GENOME_FILE" ]] && NODE_ARGS+=(--genome-file="$GENOME_FILE")

RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" "${NODE_ARGS[@]}" \
    > "$LOG_DIR/xeno-node.log" 2>&1 &
NODE_PID=$!
PIDS+=("$NODE_PID")
qlog "xeno-node started (pid $NODE_PID)"

wait_for_port "$WAIT_HOST" "$NODE_RPC_PORT" 600 "$NODE_PID"
wait_for_port "$WAIT_HOST" "$MINER_WS_PORT" 600 "$NODE_PID"

# -----------------------------------------------------------------------------
# seed-node (inference gRPC for the API gateway)
# -----------------------------------------------------------------------------
# The seed-node also has a miner websocket; put it on a different port so it does
# not collide with the xeno-node miner websocket.
qlog "Starting seed-node (inference gRPC on port $SEED_GRPC_PORT)..."
XENO_MODELS_DIR="$SEED_DATA_DIR" \
XENO_NODE_RPC="$CONNECT_HOST:$NODE_RPC_PORT" \
XENO_NODE_P2P="$CONNECT_HOST:$NODE_P2P_PORT" \
XENO_GRPC_ADDR="$BIND_IP:$SEED_GRPC_PORT" \
XENO_MINER_WS_ADDR="$BIND_IP:$SEED_MINER_WS_PORT" \
XENO_SEED_HOST="$CONNECT_HOST" \
XENO_DEFAULT_MODEL_ID="$MODEL_ID" \
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/seed-node" \
    > "$LOG_DIR/seed-node.log" 2>&1 &
SEED_PID=$!
PIDS+=("$SEED_PID")
qlog "seed-node started (pid $SEED_PID)"

wait_for_port "$WAIT_HOST" "$SEED_GRPC_PORT" 120 "$SEED_PID"

# -----------------------------------------------------------------------------
# api-gateway (OpenAI-compatible HTTP API)
# -----------------------------------------------------------------------------
qlog "Starting api-gateway on port $API_PORT..."
SEED_NODE_ADDR="http://$CONNECT_HOST:$SEED_GRPC_PORT" \
API_HOST="$BIND_IP" \
API_PORT="$API_PORT" \
USDT_CONTRACT_ADDRESS="${USDT_CONTRACT_ADDRESS:-0x0000000000000000000000000000000000000000}" \
RPC_URL="${RPC_URL:-https://polygon-mumbai.infura.io/v3/YOUR_KEY}" \
GOVERNANCE_CONTRACT_ADDRESS="${GOVERNANCE_CONTRACT_ADDRESS:-0x0000000000000000000000000000000000000000}" \
GOVERNANCE_RPC_URL="${GOVERNANCE_RPC_URL:-https://polygon-mumbai.infura.io/v3/YOUR_KEY}" \
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/api-gateway" \
    > "$LOG_DIR/api-gateway.log" 2>&1 &
API_PID=$!
PIDS+=("$API_PID")
qlog "api-gateway started (pid $API_PID)"

wait_for_port "$WAIT_HOST" "$API_PORT" 60 "$API_PID"

qok "Node + seed + API gateway running."
qok "  Miner websocket:  ws://$BIND_IP:$MINER_WS_PORT"
qok "  OpenAI API:       http://$BIND_IP:$API_PORT"
qok "  Seed gRPC:        http://$BIND_IP:$SEED_GRPC_PORT"
qok "  Kaspa RPC:        http://$BIND_IP:$NODE_RPC_PORT"
qok "  Logs:             $LOG_DIR"
qok "Press Ctrl+C to stop."

wait "$NODE_PID" 2>/dev/null || true
