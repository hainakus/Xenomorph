#!/usr/bin/env bash
# Run the Xenomorph miner on a separate rig, mining against a remote devnet node.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Run the xenom-miner on a remote GPU rig against a Xenomorph devnet node.

Required:
  --node <host>                   Hostname or IP of the devnet node
                                  (also accepts \$XENO_NODE_HOST)

Options:
  -b, --build                     Build release binary before starting (default)
  --no-build                      Skip cargo build
  -t, --trainer <name>            Miner trainer: mock, cpu, dnabert2, mgm1, gpu, cuda, metal, rocm, lora
                                  (default: \$XENO_MINER_TRAINER or mgm1)
  --features <features>           Extra cargo features for xenom-miner (e.g. cuda, metal).
                                  Overrides auto-detection. Also accepts \$XENO_MINER_FEATURES.
  --gpus <ids>                    Comma-separated GPU ordinals for multi-GPU training
                                  (default: \$XENO_MINER_GPUS or 0)
  --micro-batch-size <n>          Micro-batch size per GPU per accumulation step
                                  (default: \$XENO_MINER_MICRO_BATCH_SIZE or 1)
  --gradient-accumulation <n>     Number of gradient-accumulation steps
                                  (default: \$XENO_MINER_GRADIENT_ACCUMULATION or 2)
  --fp16                          Enable FP16 mixed precision
  --gradient-checkpointing        Enable gradient checkpointing (stub)
  --zero <n>                      ZeRO optimization level (stub, default 0)
  --max-seq-len <n>               Cap sequence length to save VRAM (default: 512)
  --lora                          Enable LoRA (default: on; use --no-lora or XENO_LORA=0 to disable)
  --no-lora                       Disable LoRA (required for from-scratch training)
  --lora-rank <n>                 LoRA rank (default: 8)
  --lora-alpha <n>                LoRA alpha (default: 16)
  --lora-dropout <f>              LoRA dropout (default: 0)
  --lora-target-modules <list>    Comma-separated LoRA target modules
  --gradient-top-k-ratio <f>      Gradient compression ratio for uploads (default: 1.0)
  -d, --data-dir <dir>            Base data directory
                                  (default: \$XENO_DATA_DIR or ./miner-data)
  --miner-ws-port <port>          Miner websocket port on the remote node
                                  (default: \$XENO_MINER_WS_PORT or 17110)
  -q, --quiet                     Minimal output
  -v, --verbose                   Debug output
  -h, --help                      Show this help and exit
"

BUILD=1
TRAINER="${XENO_MINER_TRAINER:-mgm1}"
FEATURES="${XENO_MINER_FEATURES:-}"
GPUS="${XENO_MINER_GPUS:-0}"
MICRO_BATCH_SIZE="${XENO_MINER_MICRO_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${XENO_MINER_GRADIENT_ACCUMULATION:-2}"
FP16=0
GRADIENT_CHECKPOINTING=0
ZERO=0
MAX_SEQ_LEN=512
LORA=1
if [[ -n "${XENO_LORA:-}" ]]; then
    case "$XENO_LORA" in
        0|false|no|off|FALSE) LORA=0 ;;
        1|true|yes|on|TRUE) LORA=1 ;;
    esac
fi
LORA_RANK="${XENO_LORA_RANK:-8}"
LORA_ALPHA="${XENO_LORA_ALPHA:-16}"
LORA_DROPOUT="${XENO_LORA_DROPOUT:-0}"
LORA_TARGET_MODULES=""
GRADIENT_TOP_K_RATIO="${XENO_GRADIENT_TOP_K_RATIO:-1.0}"
DATA_DIR="${XENO_DATA_DIR:-$SCRIPT_DIR/../miner-data}"
MINER_WS_PORT="${XENO_MINER_WS_PORT:-17110}"
NODE_HOST="${XENO_NODE_HOST:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--build) BUILD=1; shift ;;
        --no-build) BUILD=0; shift ;;
        -t|--trainer) TRAINER="$2"; shift 2 ;;
        --features) FEATURES="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --micro-batch-size) MICRO_BATCH_SIZE="$2"; shift 2 ;;
        --gradient-accumulation) GRADIENT_ACCUMULATION="$2"; shift 2 ;;
        --fp16) FP16=1; shift ;;
        --gradient-checkpointing) GRADIENT_CHECKPOINTING=1; shift ;;
        --zero) ZERO="$2"; shift 2 ;;
        --max-seq-len) MAX_SEQ_LEN="$2"; shift 2 ;;
        --lora) LORA=1; shift ;;
        --no-lora) LORA=0; shift ;;
        --lora-rank) LORA_RANK="$2"; shift 2 ;;
        --lora-alpha) LORA_ALPHA="$2"; shift 2 ;;
        --lora-dropout) LORA_DROPOUT="$2"; shift 2 ;;
        --lora-target-modules) LORA_TARGET_MODULES="$2"; shift 2 ;;
        --gradient-top-k-ratio) GRADIENT_TOP_K_RATIO="$2"; shift 2 ;;
        -d|--data-dir) DATA_DIR="$2"; shift 2 ;;
        --miner-ws-port) MINER_WS_PORT="$2"; shift 2 ;;
        --node) NODE_HOST="$2"; shift 2 ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

if [[ -z "$NODE_HOST" ]]; then
    err "--node (or \$XENO_NODE_HOST) is required"
    print_help_and_exit "$USAGE" 1
fi

if [[ "$TRAINER" != "mock" && "$TRAINER" != "cpu" && "$TRAINER" != "dnabert2" && \
      "$TRAINER" != "mgm1" && "$TRAINER" != "gpu" && "$TRAINER" != "cuda" && \
      "$TRAINER" != "metal" && "$TRAINER" != "rocm" && "$TRAINER" != "lora" ]]; then
    err "Unknown trainer: $TRAINER. Use mock, cpu, dnabert2, mgm1, gpu, cuda, metal, rocm, or lora."
    exit 1
fi

load_env

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DATA_DIR="$(mkdir -p "$DATA_DIR" && cd "$DATA_DIR" && pwd)"
LOG_DIR="$DATA_DIR/logs"
mkdir -p "$LOG_DIR"

BIN_PREFIX="${XENO_BIN_PREFIX:-$REPO_ROOT/target/release}"

# -----------------------------------------------------------------------------
# Build
# -----------------------------------------------------------------------------
if [[ "$BUILD" == "1" ]]; then
    if [[ -z "$FEATURES" ]]; then
        if [[ "$TRAINER" == "cuda" || "$TRAINER" == "gpu" || "$TRAINER" == "dnabert2" ]]; then
            if has_command nvidia-smi && has_command nvcc; then
                FEATURES="cuda"
                qlog "NVIDIA GPUs + nvcc detected; building with --features cuda"
            else
                warn "GPU trainer selected but nvidia-smi or nvcc not found; building CPU miner."
            fi
        elif [[ "$TRAINER" == "metal" ]]; then
            FEATURES="metal"
            qlog "Building with --features metal"
        elif [[ "$TRAINER" == "lora" ]]; then
            if has_command nvidia-smi && has_command nvcc; then
                FEATURES="cuda"
                qlog "NVIDIA GPUs + nvcc detected; building with --features cuda"
            elif [[ "$(uname -s)" == "Darwin" ]]; then
                FEATURES="metal"
                qlog "macOS detected; building with --features metal"
            else
                warn "No GPU backend detected for LoRA trainer; building CPU miner."
            fi
        fi
    else
        qlog "Building xenom-miner with explicit features: $FEATURES"
    fi

    qlog "Building xenom-miner..."
    cargo clean -p xenom-miner
    if [[ -n "$FEATURES" ]]; then
        cargo build --release -p xenom-miner --features "$FEATURES"
    else
        cargo build --release -p xenom-miner
    fi
fi

require_command "$BIN_PREFIX/xenom-miner"

# -----------------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------------
RPC_URL="ws://${NODE_HOST}:${MINER_WS_PORT}"

MINER_EXTRA_ARGS=()
if [[ "$TRAINER" == "dnabert2" || "$TRAINER" == "mgm1" || "$TRAINER" == "gpu" || "$TRAINER" == "cuda" || "$TRAINER" == "metal" || "$TRAINER" == "rocm" || "$TRAINER" == "lora" ]]; then
    MINER_EXTRA_ARGS+=(
        --gpus "$GPUS"
        --micro-batch-size "$MICRO_BATCH_SIZE"
        --gradient-accumulation "$GRADIENT_ACCUMULATION"
        --zero "$ZERO"
        --max-seq-len "$MAX_SEQ_LEN"
        --gradient-top-k-ratio "$GRADIENT_TOP_K_RATIO"
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

qlog "Starting xenom-miner against $RPC_URL (trainer=$TRAINER, gpus=$GPUS, micro_batch=$MICRO_BATCH_SIZE, acc=$GRADIENT_ACCUMULATION, max_seq_len=$MAX_SEQ_LEN, lora=$LORA, fp16=$FP16)..."
XENO_LORA_VALUE="$([ "$LORA" == "1" ] && echo true || echo false)"
XENO_WALLET_PASSWORD="${XENO_WALLET_PASSWORD:-devnet-password}" \
RUST_LOG="${RUST_LOG:-info}" \
XENO_LORA="$XENO_LORA_VALUE" \
    "$BIN_PREFIX/xenom-miner" \
    --rpc-url "$RPC_URL" \
    --model-id "${XENO_MODEL_ID:-xeno/mgm-1}" \
    --threads "${XENO_MINER_THREADS:-4}" \
    --trainer "$TRAINER" \
    --network devnet \
    --data-dir "$DATA_DIR" \
    --password "${XENO_WALLET_PASSWORD:-devnet-password}" \
    "${MINER_EXTRA_ARGS[@]}" \
    > "$LOG_DIR/xeno-miner.log" 2>&1
