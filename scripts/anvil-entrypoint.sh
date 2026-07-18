#!/usr/bin/env bash
# Entrypoint wrapper for the anvil container.

set -euo pipefail

ARGS=(
    "--host=0.0.0.0"
    "--port=8545"
    "--block-time=${XENO_BLOCK_TIME:-12}"
)

if [[ -n "${XENO_FORK_URL:-}" ]]; then
    ARGS+=("--fork-url=$XENO_FORK_URL")
fi

if [[ -n "${XENO_FOUNDRY_MNEMONIC:-}" ]]; then
    ARGS+=("--mnemonic=$XENO_FOUNDRY_MNEMONIC")
fi

exec anvil "${ARGS[@]}" "$@"
