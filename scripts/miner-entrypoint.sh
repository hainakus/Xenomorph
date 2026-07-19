#!/usr/bin/env bash
# Entrypoint wrapper for the xeno-miner container.

set -euo pipefail

ARGS=(
    "--rpc-url=${XENO_MINER_RPC_URL:-ws://xeno-seed:17110}"
    "--model-id=${XENO_MINER_MODEL_ID:-multimolecule/dnabert2}"
    "--threads=${XENO_MINER_THREADS:-4}"
    "--data-dir=${XENO_MINER_DATA_DIR:-/root/.xenom-miner}"
    "--password=${XENO_WALLET_PASSWORD:-devnet-password}"
)

ARGS+=("--trainer=${XENO_MINER_TRAINER:-mock}")

# Deprecated: --mock-mode is still accepted as a hidden alias for --trainer=mock.
if [[ "${XENO_MINER_MOCK_MODE:-false}" == "true" ]]; then
    ARGS+=("--mock-mode")
fi

if [[ -n "${XENO_MINER_WALLET:-}" ]]; then
    ARGS+=("--wallet=${XENO_MINER_WALLET}")
fi

if [[ -n "${XENO_MINER_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    IFS=' ' read -r -a EXTRA_ARGS <<< "$XENO_MINER_EXTRA_ARGS"
    ARGS+=("${EXTRA_ARGS[@]}")
fi

exec /usr/local/bin/xenom-miner "${ARGS[@]}" "$@"
