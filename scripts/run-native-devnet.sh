#!/usr/bin/env bash
# Run a native (non-Docker) Xenomorph AI devnet: xeno-node, xeno-seed, xeno-miner.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the Xenomorph devnet using locally built binaries (no Docker).
The xenom node now also serves as the training coordinator / genome server.

Options:
  -b, --build                     Build release binaries before starting (default)
  --no-build                      Skip cargo build
  -t, --trainer <name>            Miner trainer: mock, cpu, dnabert2, mgm1, gpu, cuda, metal, rocm
                                  (default: \$XENO_MINER_TRAINER or mock)
  --features <features>           Extra cargo features for xenom-miner (e.g. cuda, metal).
                                  Overrides auto-detection. Also accepts \$XENO_MINER_FEATURES.
  --gpus <ids>                    Comma-separated GPU ordinals for multi-GPU training
                                  (default: \$XENO_MINER_GPUS or 0)
  --micro-batch-size <n>          Micro-batch size per GPU per accumulation step
                                  (default: \$XENO_MINER_MICRO_BATCH_SIZE or 1)
  --gradient-accumulation <n>     Number of gradient-accumulation steps
                                  (default: \$XENO_MINER_GRADIENT_ACCUMULATION or 1)
  --gradient-top-k-ratio <f>      FedAvg gradient compression ratio (1.0 = dense, 0.1 = 10%)
                                  (default: \$XENO_MINER_GRADIENT_TOP_K_RATIO or 1.0)
  --fp16                          Enable FP16 mixed precision
  --gradient-checkpointing        Enable gradient checkpointing (stub)
  --zero <n>                      ZeRO optimization level (stub, default 0)
  --max-seq-len <n>               Cap sequence length to save VRAM (default: 512)
  --lora                          Enable LoRA (default: on)
  --lora-rank <n>                 LoRA rank (default: 8)
  --lora-alpha <n>                LoRA alpha (default: 16)
  --lora-dropout <f>              LoRA dropout (default: 0)
  --lora-target-modules <list>    Comma-separated LoRA target modules
  -d, --data-dir <dir>            Base data directory
                                  (default: \$XENO_DATA_DIR or ./devnet-data-native)
  --anvil                         Start a local anvil instance for EVM/governance tests
  -q, --quiet                     Minimal output
  -v, --verbose                   Debug output
  -h, --help                      Show this help and exit
"

BUILD=1
TRAINER="${XENO_MINER_TRAINER:-mock}"
FEATURES="${XENO_MINER_FEATURES:-}"
GPUS="${XENO_MINER_GPUS:-0}"
MICRO_BATCH_SIZE="${XENO_MINER_MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${XENO_MINER_GRADIENT_ACCUMULATION:-1}"
GRADIENT_TOP_K_RATIO="${XENO_MINER_GRADIENT_TOP_K_RATIO:-1.0}"
FP16=0
GRADIENT_CHECKPOINTING=0
ZERO=0
MAX_SEQ_LEN=512
LORA=1
LORA_RANK="${XENO_LORA_RANK:-8}"
LORA_ALPHA="${XENO_LORA_ALPHA:-16}"
LORA_DROPOUT="${XENO_LORA_DROPOUT:-0}"
LORA_TARGET_MODULES=""
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../devnet-data-native}"
START_ANVIL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -t|--trainer) TRAINER="$2"; shift 2 ;;
        --features) FEATURES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --gradient-accumulation) GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --gradient-top-k-ratio) GRADIENT_TOP_K_RATIO="$2"; shift 2 ;;
        --fp16) FP16=1; shift ;;
        --gradient-checkpointing) GRADIENT_CHECKPOINTING=1; shift ;;
        --zero) ZERO="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --lora) LORA=1; shift ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --lora-target-modules) LORA_TARGET_MODULES="$2"; shift 2 ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --anvil) START_ANVIL=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

if [[ "$TRAINER" != "mock" && "$TRAINER" != "cpu" && "$TRAINER" != "dnabert2" && \
      "$TRAINER" != "mgm1" && "$TRAINER" != "gpu" && "$TRAINER" != "cuda" && \
      "$TRAINER" != "metal" && "$TRAINER" != "rocm" ]]; then
    err "Unknown trainer: $TRAINER. Use mock, cpu, dnabert2, mgm1, gpu, cuda, metal, or rocm."
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
    # Auto-detect GPU features for the miner. Explicit --features or XENO_MINER_FEATURES
    # always wins. For NVIDIA, we also require nvcc (CUDA toolkit) because candle-core's
    # CUDA backend is compiled in at build time.
    if [[ -z "$FEATURES" ]]; then
        if [[ "$TRAINER" == "cuda" || "$TRAINER" == "gpu" || "$TRAINER" == "dnabert2" ]]; then
            if has_command nvidia-smi && has_command nvcc; then
                FEATURES="cuda"
                qlog "NVIDIA GPUs + nvcc detected; building xenom-miner with --features cuda"
            else
                warn "GPU trainer selected but nvidia-smi or nvcc not found; building CPU miner. Set XENO_MINER_FEATURES=cuda and install the CUDA toolkit to use NVIDIA GPUs."
            fi
        elif [[ "$TRAINER" == "metal" ]]; then
            FEATURES="metal"
            qlog "Building xenom-miner with --features metal"
        fi
    else
        qlog "Building xenom-miner with explicit features: $FEATURES"
    fi

    qlog "Building release binaries..."
    # Clean node/miner crates to avoid stale release artifacts after source changes
    # (cargo relies on mtimes and git checkouts can leave them older than binaries).
    cargo clean -p xenom
    cargo build --release -p xenom
    cargo clean -p xenom-miner
    if [[ -n "$FEATURES" ]]; then
        cargo build --release -p xenom-miner --features "$FEATURES"
    else
        cargo build --release -p xenom-miner
    fi
fi

require_command "$BIN_PREFIX/xenom"
require_command "$BIN_PREFIX/xenom-miner"

# -----------------------------------------------------------------------------
# Port configuration
# -----------------------------------------------------------------------------
NODE_RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
NODE_P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
MINER_WS_PORT="${XENO_MINER_RPC_PORT:-17110}"

if ! is_port_free "$NODE_RPC_PORT" || ! is_port_free "$NODE_P2P_PORT" || \
   ! is_port_free "$MINER_WS_PORT"; then
    err "One or more required ports are already in use: $NODE_RPC_PORT, $NODE_P2P_PORT, $MINER_WS_PORT"
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
# If LoRA is requested, export the LoRA env vars so the node's ModelManager
# builds a LoRA-capable model and can aggregate LoRA gradients from miners.
if [[ "$LORA" == "1" ]]; then
    export XENO_LORA=1
    export XENO_LORA_RANK="$LORA_RANK"
    export XENO_LORA_ALPHA="$LORA_ALPHA"
    export XENO_LORA_DROPOUT="$LORA_DROPOUT"
    [[ -n "$LORA_TARGET_MODULES" ]] && export XENO_LORA_TARGET_MODULES="$LORA_TARGET_MODULES"
fi

qlog "Starting xeno-node..."
RUST_LOG="${RUST_LOG:-info}" "$BIN_PREFIX/xenom" \
    --devnet \
    --appdir="$NODE_DATA_DIR" \
    --logdir="$LOG_DIR/node" \
    --rpclisten="0.0.0.0:$NODE_RPC_PORT" \
    --listen="0.0.0.0:$NODE_P2P_PORT" \
    --miner-ws-listen="0.0.0.0:$MINER_WS_PORT" \
    --models-dir="$SEED_DATA_DIR" \
    --disable-upnp \
    --nodnsseed \
    > "$LOG_DIR/xeno-node.log" 2>&1 &
NODE_PID=$!
PIDS+=("$NODE_PID")
qlog "xeno-node started (pid $NODE_PID)"

NODE_RPC_TIMEOUT="${XENO_NODE_RPC_TIMEOUT:-600}"
wait_for_port "$NODE_RPC_PORT" "$NODE_RPC_TIMEOUT" "$NODE_PID"
# The miner WebSocket listener is bound only after the active model is downloaded,
# which can take several minutes on the first run.
wait_for_port "$MINER_WS_PORT" 600 "$NODE_PID"

# -----------------------------------------------------------------------------
# xeno-miner
# -----------------------------------------------------------------------------
MINER_EXTRA_ARGS=()
if [[ "$TRAINER" == "dnabert2" || "$TRAINER" == "mgm1" || "$TRAINER" == "gpu" || "$TRAINER" == "cuda" || "$TRAINER" == "metal" || "$TRAINER" == "rocm" ]]; then
    MINER_EXTRA_ARGS+=(
        --gpus "$GPUS"
        --micro-batch-size "$MICRO_BATCH_SIZE"
        --gradient-accumulation "$GRADIENT_ACCUMULATION"
        --gradient-top-k-ratio "$GRADIENT_TOP_K_RATIO"
        --zero "$ZERO"
        --max-seq-len "$MAX_SEQ_LEN"
    )
    [[ "$FP16" == "1" ]] && MINER_EXTRA_ARGS+=(--fp16)
    [[ "$GRADIENT_CHECKPOINTING" == "1" ]] && MINER_EXTRA_ARGS+=(--gradient-checkpointing)
    if [[ "$LORA" == "1" ]]; then
        MINER_EXTRA_ARGS+=(--lora)
        MINER_EXTRA_ARGS+=(--lora-rank "$LORA_RANK")
        MINER_EXTRA_ARGS+=(--lora-alpha "$LORA_ALPHA")
        MINER_EXTRA_ARGS+=(--lora-dropout "$LORA_DROPOUT")
        [[ -n "$LORA_TARGET_MODULES" ]] && MINER_EXTRA_ARGS+=(--lora-target-modules "$LORA_TARGET_MODULES")
    fi
fi

qlog "Starting xeno-miner (trainer=$TRAINER, gpus=$GPUS, micro_batch=$MICRO_BATCH_SIZE, acc=$GRADIENT_ACCUMULATION, max_seq_len=$MAX_SEQ_LEN, lora=$LORA, fp16=$FP16)..."
XENO_WALLET_PASSWORD="${XENO_WALLET_PASSWORD:-devnet-password}" \
RUST_LOG="${RUST_LOG:-info}" \
    "$BIN_PREFIX/xenom-miner" \
    --rpc-url "ws://127.0.0.1:$MINER_WS_PORT" \
    --model-id "${XENO_MINER_MODEL_ID:-xeno/mgm-1}" \
    --threads "${XENO_MINER_THREADS:-4}" \
    --trainer "$TRAINER" \
    --network devnet \
    --data-dir "$MINER_DATA_DIR" \
    --password "${XENO_WALLET_PASSWORD:-devnet-password}" \
    "${MINER_EXTRA_ARGS[@]}" \
    > "$LOG_DIR/xeno-miner.log" 2>&1 &
MINER_PID=$!
PIDS+=("$MINER_PID")
qlog "xeno-miner started (pid $MINER_PID)"

qok "Native devnet running. Logs in $LOG_DIR. Press Ctrl+C to stop."

# Wait for any background process. The trap will clean up on Ctrl+C.
wait "$NODE_PID" "$MINER_PID" 2>/dev/null || true
