#!/usr/bin/env bash
# Run a Xenomorph devnet node (full node + training + inference gRPC) + OpenAI API gateway
# on a single server. No miner is started here; miners connect from other rigs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the Xenomorph node (full node + training + inference gRPC) and the API
gateway on one host using locally built binaries. External miners connect to
this node via --miner-ws-listen.

Options:
  -b, --build                     Build release binaries before starting (default)
  --no-build                      Skip cargo build
  -d, --data-dir <dir>            Base data directory
                                  (default: \$XENO_DATA_DIR or ./devnet-data-native)
  --genome-file <path>            Path to .xenom packed GRCh38 genome file
                                  (optional; enables real Genome PoW)
  --bind-ip <ip>                  IP to bind node/API sockets to
                                  (default: \$XENO_BIND_IP or 0.0.0.0)
  --quic-port <port>              QUIC bulk transfer server port
                                  (default: \$XENO_QUIC_PORT or 17111)
  --quic-external <addr:port>     External QUIC address announced to miners
                                  (default: auto-detected from bind/local IP)
  --quic-max-transfers <n>        Max concurrent QUIC checkpoint transfers
                                  (default: \$XENO_QUIC_MAX_TRANSFERS or 64)
  --lora                          Enable LoRA (default: on)
  --lora-rank <n>                 LoRA rank (default: 8)
  --lora-alpha <n>                LoRA alpha (default: 16)
  --lora-dropout <f>              LoRA dropout (default: 0)
  --lora-target-modules <list>    Comma-separated LoRA target modules
  -q, --quiet                     Minimal output
  -v, --verbose                   Debug output
  -h, --help                      Show this help and exit
"

BUILD=1
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../devnet-data-native}"
GENOME_FILE="${XENO_GENOME_FILE:-}"
BIND_IP="${XENO_BIND_IP:-0.0.0.0}"
QUIC_PORT="${XENO_QUIC_PORT:-17111}"
QUIC_EXTERNAL="${XENO_QUIC_EXTERNAL:-}"
QUIC_MAX_TRANSFERS="${XENO_QUIC_MAX_TRANSFERS:-64}"
LORA=1
LORA_RANK="${XENO_LORA_RANK:-8}"
LORA_ALPHA="${XENO_LORA_ALPHA:-16}"
LORA_DROPOUT="${XENO_LORA_DROPOUT:-0}"
LORA_TARGET_MODULES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --genome-file) GENOME_FILE="$2"; shift 2 ;;
        --bind-ip) BIND_IP="$2"; shift 2 ;;
        --quic-port) QUIC_PORT="$2"; shift 2 ;;
        --quic-external) QUIC_EXTERNAL="$2"; shift 2 ;;
        --quic-max-transfers) QUIC_MAX_TRANSFERS="$2"; shift 2 ;;
        --lora) LORA=1; shift ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --lora-target-modules) LORA_TARGET_MODULES="$2"; shift 2 ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

load_env

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

# Honour XENO_* from .env/env, keeping CLI flags as fallbacks.
BIND_IP="${XENO_BIND_IP:-$BIND_IP}"
QUIC_PORT="${XENO_QUIC_PORT:-$QUIC_PORT}"
QUIC_EXTERNAL="${XENO_QUIC_EXTERNAL:-$QUIC_EXTERNAL}"
QUIC_MAX_TRANSFERS="${XENO_QUIC_MAX_TRANSFERS:-$QUIC_MAX_TRANSFERS}"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
NODE_DATA_DIR="$DATA_DIR/node"
MODELS_DIR="$DATA_DIR/models"
REDIS_DATA_DIR="$DATA_DIR/redis"
LOG_DIR="$DATA_DIR/logs"

mkdir -p "$NODE_DATA_DIR" "$MODELS_DIR" "$REDIS_DATA_DIR" "$LOG_DIR"

BIN_PREFIX="${XENO_BIN_PREFIX:-$REPO_ROOT/target/release}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
    qlog "Building release binaries (xenom + inference gRPC, api-gateway)..."
    cargo clean -p xenom
    cargo build --release -p xenom
    cargo clean -p api-gateway
    cargo build --release -p api-gateway
fi

require_command "$BIN_PREFIX/xenom"
require_command "$BIN_PREFIX/api-gateway"

# -----------------------------------------------------------------------------
# Port configuration
# -----------------------------------------------------------------------------
NODE_RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
NODE_P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
MINER_WS_PORT="${XENO_MINER_WS_PORT:-17110}"
QUIC_PORT="${XENO_QUIC_PORT:-$QUIC_PORT}"
INFERENCE_GRPC_PORT="${XENO_INFERENCE_GRPC_PORT:-50051}"
API_PORT="${XENO_API_PORT:-3000}"
REDIS_PORT="${XENO_REDIS_PORT:-6379}"
MODEL_ID="${XENO_MODEL_ID:-xeno/mgm-1}"

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
   ! is_port_free "$MINER_WS_PORT" || ! is_port_free "$QUIC_PORT" || \
   ! is_port_free "$INFERENCE_GRPC_PORT" || ! is_port_free "$API_PORT"; then
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
    qlog "Stopping node / api-gateway..."
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
# xeno-node (full node + training coordinator + inference gRPC)
# -----------------------------------------------------------------------------
# If LoRA is requested, export the LoRA env vars so the node's ModelManager
# builds a LoRA-capable model and can aggregate LoRA gradients from miners.
if [[ "$LORA" == "1" ]]; then
    export XENO_LORA=1
    export XENO_LORA_RANK="$LORA_RANK"
    export XENO_LORA_ALPHA="$LORA_ALPHA"
    export XENO_LORA_DROPOUT="$LORA_DROPOUT"
    [[ -n "$LORA_TARGET_MODULES" ]] && export XENO_LORA_TARGET_MODULES="$LORA_TARGET_MODULES"
fi

# If no explicit external QUIC address was given, derive one from the bind IP
# or try to find a non-loopback local IP so miners receive a connectable endpoint.
if [[ -z "$QUIC_EXTERNAL" ]]; then
    if [[ "$BIND_IP" != "0.0.0.0" && "$BIND_IP" != "127.0.0.1" ]]; then
        QUIC_EXTERNAL="$BIND_IP:$QUIC_PORT"
    else
        if command -v hostname &>/dev/null && hostname -I &>/dev/null; then
            for ip in $(hostname -I); do
                if [[ "$ip" != 127.* && "$ip" != ::1* ]]; then
                    QUIC_EXTERNAL="$ip:$QUIC_PORT"
                    break
                fi
            done
        elif command -v ipconfig &>/dev/null; then
            ip=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)
            if [[ -n "$ip" ]]; then
                QUIC_EXTERNAL="$ip:$QUIC_PORT"
            fi
        fi
        if [[ -z "$QUIC_EXTERNAL" ]]; then
            qwarn "Could not auto-detect a non-loopback IP for QUIC. External miners may need --quic-external."
        fi
    fi
fi

qlog "Starting xeno-node..."
NODE_ARGS=(
    --devnet
    --appdir="$NODE_DATA_DIR"
    --logdir="$LOG_DIR/node"
    --rpclisten="$BIND_IP:$NODE_RPC_PORT"
    --listen="$BIND_IP:$NODE_P2P_PORT"
    --miner-ws-listen="$BIND_IP:$MINER_WS_PORT"
    --quic-listen="$BIND_IP:$QUIC_PORT"
    --quic-max-transfers="$QUIC_MAX_TRANSFERS"
    --inference-grpc-listen="$BIND_IP:$INFERENCE_GRPC_PORT"
    --models-dir="$MODELS_DIR"
    --disable-upnp
    --nodnsseed
    --addpeer="94.237.108.145:16111"
)
[[ -n "$QUIC_EXTERNAL" ]] && NODE_ARGS+=(--quic-external="$QUIC_EXTERNAL")
[[ -n "$GENOME_FILE" ]] && NODE_ARGS+=(--genome-file="$GENOME_FILE")

RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" "${NODE_ARGS[@]}" \
    > "$LOG_DIR/xeno-node.log" 2>&1 &
NODE_PID=$!
PIDS+=("$NODE_PID")
qlog "xeno-node started (pid $NODE_PID)"

wait_for_port "$WAIT_HOST" "$NODE_RPC_PORT" 600 "$NODE_PID"
wait_for_port "$WAIT_HOST" "$MINER_WS_PORT" 600 "$NODE_PID"
wait_for_port "$WAIT_HOST" "$INFERENCE_GRPC_PORT" 120 "$NODE_PID"

# -----------------------------------------------------------------------------
# api-gateway (OpenAI-compatible HTTP API)
# -----------------------------------------------------------------------------
qlog "Starting api-gateway on port $API_PORT..."
SEED_NODE_ADDR="http://$CONNECT_HOST:$INFERENCE_GRPC_PORT" \
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

qok "Node (unified) + API gateway running."
qok "  Miner websocket:  ws://$BIND_IP:$MINER_WS_PORT"
qok "  QUIC transfer:    ${QUIC_EXTERNAL:-$BIND_IP:$QUIC_PORT}"
qok "  OpenAI API:       http://$BIND_IP:$API_PORT"
qok "  Inference gRPC:   http://$BIND_IP:$INFERENCE_GRPC_PORT"
qok "  Kaspa RPC:        http://$BIND_IP:$NODE_RPC_PORT"
qok "  Logs:             $LOG_DIR"
qok "Press Ctrl+C to stop."

wait "$NODE_PID" 2>/dev/null || true
