#!/usr/bin/env bash

set -euo pipefail

BIN_DIR="/opt/xenom/bin"
LOG_DIR="${LOG_DIR:-/var/log/xenom}"
DATA_DIR="${DATA_DIR:-/var/lib/xenom}"
AIO_IMAGE_TAG="${AIO_IMAGE_TAG:-unknown}"
DB_PATH="${DB_PATH:-$DATA_DIR/genetics-l2.db}"
WORK_ROOT="${WORK_ROOT:-$DATA_DIR/l2-work}"
COORDINATOR_PORT="${COORDINATOR_PORT:-8091}"
COORDINATOR_HOST="${COORDINATOR_HOST:-0.0.0.0}"
COORDINATOR_URL="${COORDINATOR_URL:-http://127.0.0.1:${COORDINATOR_PORT}}"
FETCHER_POLL_SECS="${FETCHER_POLL_SECS:-300}"
WORKER_POLL_MS="${WORKER_POLL_MS:-5000}"
VALIDATOR_POLL_MS="${VALIDATOR_POLL_MS:-10000}"
VALIDATOR_SCORE_TOLERANCE="${VALIDATOR_SCORE_TOLERANCE:-0.05}"
SETTLEMENT_POLL_MS="${SETTLEMENT_POLL_MS:-15000}"
FETCHER_FLAGS="${FETCHER_FLAGS:---sra --igsr --gnomad --gdc --clinvar}"
ENABLE_WORKER="${ENABLE_WORKER:-0}" # 1=run local worker, 0=disable
EVM_RPC_ADDR="${EVM_RPC_ADDR:-127.0.0.1:8545}"
EVM_BLOCK_TIME_MS="${EVM_BLOCK_TIME_MS:-2000}"
EVM_STATE_DIR="${EVM_STATE_DIR:-$DATA_DIR/evm-state}"
EVM_DEVNET="${EVM_DEVNET:-0}" # 1=pass --devnet, 0=mainnet/testnet mode
EVM_L1_NODE="${EVM_L1_NODE:-}"
SETTLEMENT_MODE="${SETTLEMENT_MODE:-dry-run}" # dry-run | submit
SETTLEMENT_NODE="${SETTLEMENT_NODE:-127.0.0.1:36669}"
SETTLEMENT_NETWORK="${SETTLEMENT_NETWORK:-mainnet}" # mainnet | devnet | testnet
SETTLEMENT_EVM_NODE="${SETTLEMENT_EVM_NODE:-http://127.0.0.1:8545}"
SETTLEMENT_FEE_SOMPI="${SETTLEMENT_FEE_SOMPI:-}"
SETTLEMENT_QUORUM="${SETTLEMENT_QUORUM:-1}"
SETTLEMENT_SCORE_TOLERANCE="${SETTLEMENT_SCORE_TOLERANCE:-0.05}"
ENABLE_ANCHOR_COMMITTER="${ENABLE_ANCHOR_COMMITTER:-1}" # 1=run anchor committer, 0=disable
ANCHOR_MODE="${ANCHOR_MODE:-dry-run}" # dry-run | submit
ANCHOR_NODE="${ANCHOR_NODE:-127.0.0.1:36669}"
ANCHOR_EVM_NODE="${ANCHOR_EVM_NODE:-http://127.0.0.1:8545}"
ANCHOR_POLL_MS="${ANCHOR_POLL_MS:-10000}"
ANCHOR_STATE_DIR="${ANCHOR_STATE_DIR:-$DATA_DIR/anchor-state}"
ANCHOR_NETWORK="${ANCHOR_NETWORK:-mainnet}" # mainnet | devnet | testnet

mkdir -p "$LOG_DIR" "$DATA_DIR" "$WORK_ROOT" "$EVM_STATE_DIR" "$ANCHOR_STATE_DIR"
if [[ "$ENABLE_WORKER" == "1" && -z "${WORKER_PRIVKEY:-}" ]]; then
  echo "ERROR: WORKER_PRIVKEY is required when ENABLE_WORKER=1"
  exit 1
fi

if [[ -z "${VALIDATOR_PRIVKEY:-}" ]]; then
  echo "ERROR: VALIDATOR_PRIVKEY is required"
  exit 1
fi

if [[ "$ENABLE_WORKER" != "0" && "$ENABLE_WORKER" != "1" ]]; then
  echo "ERROR: ENABLE_WORKER must be 0 or 1"
  exit 1
fi

if [[ "$EVM_DEVNET" != "0" && "$EVM_DEVNET" != "1" ]]; then
  echo "ERROR: EVM_DEVNET must be 0 or 1"
  exit 1
fi

if [[ "$EVM_DEVNET" == "0" && -z "$EVM_L1_NODE" ]]; then
  echo "ERROR: EVM_L1_NODE is required when EVM_DEVNET=0 (mainnet/testnet L1-tied mode)"
  exit 1
fi

if [[ "$SETTLEMENT_MODE" != "dry-run" && "$SETTLEMENT_MODE" != "submit" ]]; then
  echo "ERROR: SETTLEMENT_MODE must be 'dry-run' or 'submit'"
  exit 1
fi

if [[ "$SETTLEMENT_NETWORK" != "mainnet" && "$SETTLEMENT_NETWORK" != "devnet" && "$SETTLEMENT_NETWORK" != "testnet" ]]; then
  echo "ERROR: SETTLEMENT_NETWORK must be 'mainnet', 'devnet', or 'testnet'"
  exit 1
fi
if [[ "$ENABLE_ANCHOR_COMMITTER" != "0" && "$ENABLE_ANCHOR_COMMITTER" != "1" ]]; then
  echo "ERROR: ENABLE_ANCHOR_COMMITTER must be 0 or 1"
  exit 1
fi
if [[ "$ANCHOR_MODE" != "dry-run" && "$ANCHOR_MODE" != "submit" ]]; then
  echo "ERROR: ANCHOR_MODE must be 'dry-run' or 'submit'"
  exit 1
fi
if [[ "$ANCHOR_NETWORK" != "mainnet" && "$ANCHOR_NETWORK" != "devnet" && "$ANCHOR_NETWORK" != "testnet" ]]; then
  echo "ERROR: ANCHOR_NETWORK must be 'mainnet', 'devnet', or 'testnet'"
  exit 1
fi

if ! [[ "$SETTLEMENT_QUORUM" =~ ^[0-9]+$ ]] || [[ "$SETTLEMENT_QUORUM" -lt 1 ]]; then
  echo "ERROR: SETTLEMENT_QUORUM must be an integer >= 1"
  exit 1
fi

if [[ "$SETTLEMENT_MODE" == "submit" && -z "${SETTLEMENT_PRIVKEY:-}" ]]; then
  echo "ERROR: SETTLEMENT_PRIVKEY is required when SETTLEMENT_MODE=submit"
  exit 1
fi
if [[ "$ENABLE_ANCHOR_COMMITTER" == "1" && "$ANCHOR_MODE" == "submit" && -z "${COMMITTER_PRIVKEY:-}" ]]; then
  echo "ERROR: COMMITTER_PRIVKEY is required when ENABLE_ANCHOR_COMMITTER=1 and ANCHOR_MODE=submit"
  exit 1
fi

PIDS=()

start_bg() {
  local name="$1"
  shift
  echo "[l2] starting ${name}..."
  "$@" > "${LOG_DIR}/${name}.log" 2>&1 &
  PIDS+=("$!")
}

wait_for_coordinator() {
  local tries=60
  local url="${COORDINATOR_URL}/health"
  echo "[l2] waiting for coordinator health: ${url}"
  for _ in $(seq 1 "$tries"); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[l2] coordinator is healthy"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: coordinator did not become healthy in time"
  return 1
}

wait_for_key_file() {
  local key_file="${DB_PATH}.key"
  local tries=60
  echo "[l2] waiting for coordinator key file: ${key_file}"
  for _ in $(seq 1 "$tries"); do
    if [[ -s "$key_file" ]]; then
      echo "[l2] coordinator key file is ready"
      return 0
    fi
    sleep 1
  done
  echo "ERROR: coordinator key file did not appear in time"
  return 1
}

cleanup() {
  echo "[l2] shutting down..."
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
}
trap cleanup EXIT INT TERM

evm_cmd=(
  "${BIN_DIR}/xenom-evm-node"
  --rpc-addr "$EVM_RPC_ADDR"
  --state-dir "$EVM_STATE_DIR"
)
if [[ "$EVM_DEVNET" == "1" ]]; then
  evm_cmd+=(--devnet)
fi
if [[ -n "$EVM_L1_NODE" ]]; then
  evm_cmd+=(--l1-node "$EVM_L1_NODE")
else
  evm_cmd+=(--block-time "$EVM_BLOCK_TIME_MS")
fi
start_bg evm-node "${evm_cmd[@]}"

start_bg coordinator \
  "${BIN_DIR}/genetics-l2-coordinator" \
  --db-path "$DB_PATH" \
  --listen "${COORDINATOR_HOST}:${COORDINATOR_PORT}" \
  --scripts-dir /opt/xenom/scripts \
  --datasets-dir "${DATA_DIR}/datasets"

wait_for_coordinator
wait_for_key_file

start_bg fetcher \
  "${BIN_DIR}/genetics-l2-fetcher" \
  --coordinator "$COORDINATOR_URL" \
  --poll-secs "$FETCHER_POLL_SECS" \
  ${FETCHER_FLAGS}

if [[ "$ENABLE_WORKER" == "1" ]]; then
  start_bg worker \
    "${BIN_DIR}/genetics-l2-worker" \
    --coordinator "$COORDINATOR_URL" \
    --work-root "$WORK_ROOT" \
    --poll-ms "$WORKER_POLL_MS"
else
  echo "[l2] local worker disabled (ENABLE_WORKER=0)"
fi

start_bg validator \
  "${BIN_DIR}/genetics-l2-validator" \
  --coordinator "$COORDINATOR_URL" \
  --coordinator-key-file "${DB_PATH}.key" \
  --poll-ms "$VALIDATOR_POLL_MS" \
  --score-tolerance "$VALIDATOR_SCORE_TOLERANCE"

settlement_cmd=(
  "${BIN_DIR}/genetics-l2-settlement"
  --coordinator "$COORDINATOR_URL"
  --node "$SETTLEMENT_NODE"
  --poll-ms "$SETTLEMENT_POLL_MS"
  --quorum "$SETTLEMENT_QUORUM"
  --score-tolerance "$SETTLEMENT_SCORE_TOLERANCE"
)
if [[ "$SETTLEMENT_NETWORK" == "devnet" ]]; then
  settlement_cmd+=(--devnet)
elif [[ "$SETTLEMENT_NETWORK" == "testnet" ]]; then
  settlement_cmd+=(--testnet)
fi
if [[ -n "$SETTLEMENT_EVM_NODE" ]]; then
  settlement_cmd+=(--evm-node "$SETTLEMENT_EVM_NODE")
fi
if [[ -n "$SETTLEMENT_FEE_SOMPI" ]]; then
  settlement_cmd+=(--fee-sompi "$SETTLEMENT_FEE_SOMPI")
fi
if [[ "$SETTLEMENT_MODE" == "submit" ]]; then
  settlement_cmd+=(--submit)
fi
start_bg settlement "${settlement_cmd[@]}"

if [[ "$ENABLE_ANCHOR_COMMITTER" == "1" ]]; then
  anchor_cmd=(
    "${BIN_DIR}/xenom-anchor-committer"
    --evm-node "$ANCHOR_EVM_NODE"
    --node "$ANCHOR_NODE"
    --state-dir "$ANCHOR_STATE_DIR"
    --poll-ms "$ANCHOR_POLL_MS"
  )
  if [[ "$ANCHOR_NETWORK" == "devnet" ]]; then
    anchor_cmd+=(--devnet)
  elif [[ "$ANCHOR_NETWORK" == "testnet" ]]; then
    anchor_cmd+=(--testnet)
  fi
  if [[ "$ANCHOR_MODE" == "submit" ]]; then
    anchor_cmd+=(--submit)
  fi
  start_bg anchor-committer "${anchor_cmd[@]}"
else
  echo "[l2] anchor-committer disabled (ENABLE_ANCHOR_COMMITTER=0)"
fi

echo "[l2] all services started"
echo "[l2] logs: ${LOG_DIR}/*.log"
if [[ -n "$EVM_L1_NODE" ]]; then
  EVM_MODE="l1-tied"
else
  EVM_MODE="timed"
fi
echo "[l2] image: ${AIO_IMAGE_TAG}"
echo "[l2] evm: mode=${EVM_MODE} rpc=http://${EVM_RPC_ADDR} l1=${EVM_L1_NODE:-none}"
echo "[l2] coordinator: ${COORDINATOR_URL}"
echo "[l2] worker: enabled=${ENABLE_WORKER}"
echo "[l2] settlement: mode=${SETTLEMENT_MODE} node=${SETTLEMENT_NODE} network=${SETTLEMENT_NETWORK} evm=${SETTLEMENT_EVM_NODE}"
echo "[l2] anchor-committer: enabled=${ENABLE_ANCHOR_COMMITTER} mode=${ANCHOR_MODE} node=${ANCHOR_NODE} network=${ANCHOR_NETWORK} evm=${ANCHOR_EVM_NODE}"

wait -n "${PIDS[@]}"
echo "[l2] one process exited, stopping container"
