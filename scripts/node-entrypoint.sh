#!/usr/bin/env bash
# Entrypoint wrapper for the xeno-node container.

set -euo pipefail

RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"

ARGS=(
    "--rpclisten=0.0.0.0:${RPC_PORT}"
    "--listen=0.0.0.0:${P2P_PORT}"
)

if [[ -n "${XENO_NODE_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    IFS=' ' read -r -a EXTRA_ARGS <<< "$XENO_NODE_EXTRA_ARGS"
    ARGS+=("${EXTRA_ARGS[@]}")
fi

if [[ -n "${XENO_NODE_ADDPEERS:-}" ]]; then
    for peer in $XENO_NODE_ADDPEERS; do
        ARGS+=("--addpeer=$peer")
    done
fi

exec /usr/local/bin/xenom "${ARGS[@]}" "$@"
