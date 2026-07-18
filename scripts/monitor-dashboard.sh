#!/usr/bin/env bash
# Real-time monitoring dashboard for the Xenomorph AI devnet.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --export <file>    Save snapshot to file and exit
  --interval <secs>  Refresh interval (default: 5)
  --once             Single snapshot, do not loop
  -q, --quiet        Minimal output
  -v, --verbose      Debug output
  -h, --help         Show this help and exit
"

EXPORT=""
INTERVAL=5
ONCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --export) EXPORT="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --once) ONCE=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

load_env
require_command jq
require_docker

RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
SEED_PORT="${XENO_SEED_GRPC_PORT:-50051}"

# -----------------------------------------------------------------------------
# Snapshot
# -----------------------------------------------------------------------------
snapshot() {
    local height peers blocks miner_status seed_status node_status cpu mem

    height="$(get_block_height 127.0.0.1 "$RPC_PORT")"
    height="${height:-N/A}"
    peers="$(get_peer_count 127.0.0.1 "$RPC_PORT")"
    peers="${peers:-0}"

    blocks="$(docker_compose logs --no-log-prefix --tail=200 xeno-miner 2>/dev/null | grep -c 'Submitted block' || echo 0)"

    node_status="unknown"
    seed_status="unknown"
    miner_status="unknown"

    if docker_compose ps -q xeno-node &>/dev/null; then
        node_status="$(docker inspect -f '{{.State.Status}}' "$(docker_compose ps -q xeno-node)" 2>/dev/null || echo unknown)"
    fi
    if docker_compose ps -q xeno-seed &>/dev/null; then
        seed_status="$(docker inspect -f '{{.State.Status}}' "$(docker_compose ps -q xeno-seed)" 2>/dev/null || echo unknown)"
    fi
    if docker_compose ps -q xeno-miner &>/dev/null; then
        miner_status="$(docker inspect -f '{{.State.Status}}' "$(docker_compose ps -q xeno-miner)" 2>/dev/null || echo unknown)"
    fi

    # Docker stats for CPU/Mem
    cpu="N/A"; mem="N/A"
    local stats
    stats="$(docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' xeno-node xeno-seed xeno-miner 2>/dev/null || true)"
    if [[ -n "$stats" ]]; then
        cpu="$(echo "$stats" | awk -F',' '{gsub(/%/,"",$1); sum+=$1} END {printf "%.1f%%", sum}')"
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S') | Node: ${node_status} | Seed: ${seed_status} | Miner: ${miner_status}"
    echo "  Height: $height | Peers: $peers | Blocks submitted: $blocks | CPU: $cpu | Mem: $mem"
}

# -----------------------------------------------------------------------------
# Export snapshot
# -----------------------------------------------------------------------------
if [[ -n "$EXPORT" ]]; then
    snapshot > "$EXPORT"
    qok "Snapshot saved to $EXPORT"
    exit 0
fi

# -----------------------------------------------------------------------------
# Loop
# -----------------------------------------------------------------------------
if [[ "$ONCE" == "1" ]]; then
    snapshot
    exit 0
fi

qlog "Monitoring devnet (press Ctrl-C to stop)..."
while true; do
    clear || printf '\033[H\033[J'
    echo "=== Xenomorph AI Devnet Monitor ==="
    snapshot
    sleep "$INTERVAL"
done
