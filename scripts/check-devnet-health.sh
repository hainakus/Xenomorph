#!/usr/bin/env bash
# Check the health of the Xenomorph AI devnet services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --service <name>    Check a specific service (node, seed, miner, anvil)
  --json              Output JSON for CI/CD
  -q, --quiet         Minimal output
  -v, --verbose       Debug output
  -h, --help          Show this help and exit
"

SERVICE=""
JSON=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --service) SERVICE="$2"; shift 2 ;;
        --json) JSON=1; shift ;;
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
ANVIL_PORT="${XENO_ANVIL_PORT:-8545}"

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
declare -A STATUS
declare -A MESSAGE
declare -A PEERS
declare -A BLOCKS
OVERALL=0

set_status() {
    local svc="$1" st="$2" msg="$3"
    STATUS["$svc"]="$st"
    MESSAGE["$svc"]="$msg"
    if [[ "$st" == "critical" ]]; then
        OVERALL=1
    fi
}

# -----------------------------------------------------------------------------
# Container status
# -----------------------------------------------------------------------------
check_container() {
    local svc="$1"
    local state
    state="$(docker_compose ps -q "$svc" 2>/dev/null | xargs -I {} docker inspect -f '{{.State.Status}}' {} 2>/dev/null || echo "missing")"
    debug "$svc container state: $state"
    case "$state" in
        running)  set_status "$svc" healthy "running" ;;
        restarting) set_status "$svc" warning "restarting" ;;
        exited|dead) set_status "$svc" critical "stopped ($state)" ;;
        *) set_status "$svc" critical "not found" ;;
    esac
}

# -----------------------------------------------------------------------------
# TCP port checks
# -----------------------------------------------------------------------------
check_tcp() {
    local host="$1" port="$2"
    timeout 2 bash -c "cat < /dev/null > /dev/tcp/$host/$port" 2>/dev/null
}

# -----------------------------------------------------------------------------
# Service checks
# -----------------------------------------------------------------------------
check_node() {
    check_container xeno-node
    if [[ "${STATUS[xeno-node]}" != "critical" ]]; then
        if check_tcp 127.0.0.1 "$RPC_PORT"; then
            local height
            height="$(get_block_height 127.0.0.1 "$RPC_PORT")"
            if [[ -n "$height" ]]; then
                set_status xeno-node healthy "RPC reachable, blue score: $height"
                BLOCKS["xeno-node"]="$height"
            else
                set_status xeno-node warning "RPC port open but no response"
            fi
        else
            set_status xeno-node critical "RPC port $RPC_PORT not reachable"
        fi
    fi
    PEERS["xeno-node"]="$(get_peer_count 127.0.0.1 "$P2P_PORT")"
}

check_seed() {
    check_container xeno-seed
    if [[ "${STATUS[xeno-seed]}" != "critical" ]]; then
        if check_tcp 127.0.0.1 "$SEED_PORT"; then
            set_status xeno-seed healthy "gRPC port $SEED_PORT reachable"
        else
            set_status xeno-seed warning "gRPC port $SEED_PORT not reachable"
        fi
    fi
}

check_miner() {
    check_container xeno-miner
    if [[ "${STATUS[xeno-miner]}" != "critical" ]]; then
        local blocks
        blocks="$(docker_compose logs --no-log-prefix --tail=50 xeno-miner 2>/dev/null | grep -c 'Submitted block' || true)"
        if [[ "$blocks" -gt 0 ]]; then
            set_status xeno-miner healthy "$blocks blocks submitted"
        else
            set_status xeno-miner warning "running, no blocks submitted yet"
        fi
    fi
}

check_anvil() {
    if docker_compose ps -q anvil &>/dev/null; then
        check_container anvil
        if [[ "${STATUS[anvil]}" != "critical" ]]; then
            if check_tcp 127.0.0.1 "$ANVIL_PORT"; then
                local block
                block="$(timeout 5 curl -s -X POST "http://127.0.0.1:$ANVIL_PORT" \
                    -H 'Content-Type: application/json' \
                    -d '{"jsonrpc":"2.0","method":"eth_blockNumber","params":[],"id":1}' 2>/dev/null | jq -r '.result // empty' 2>/dev/null || true)"
                if [[ -n "$block" ]]; then
                    set_status anvil healthy "RPC reachable, block: $block"
                else
                    set_status anvil warning "RPC port open but no block number"
                fi
            else
                set_status anvil critical "RPC port $ANVIL_PORT not reachable"
            fi
        fi
    else
        set_status anvil healthy "not running (profile evm disabled)"
    fi
}

# -----------------------------------------------------------------------------
# Run checks
# -----------------------------------------------------------------------------
SERVICES=("xeno-node" "xeno-seed" "xeno-miner" "anvil")
if [[ -n "$SERVICE" ]]; then
    case "$SERVICE" in
        node) check_node ;;
        seed) check_seed ;;
        miner) check_miner ;;
        anvil) check_anvil ;;
        *) err "Unknown service: $SERVICE"; exit 1 ;;
    esac
else
    check_node
    check_seed
    check_miner
    check_anvil
fi

# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------
if [[ "$JSON" == "1" ]]; then
    # Build JSON manually to avoid jq dependency in constrained environments
    json_out="{"
    first=1
    for svc in "${SERVICES[@]}"; do
        if [[ -n "${STATUS[$svc]:-}" ]]; then
            [[ "$first" == "1" ]] || json_out+=","
            json_out+="\"$svc\":{\"status\":\"${STATUS[$svc]}\",\"message\":\"${MESSAGE[$svc]}\""
            if [[ -n "${PEERS[$svc]:-}" ]]; then
                json_out+=",\"peers\":${PEERS[$svc]}"
            fi
            if [[ -n "${BLOCKS[$svc]:-}" ]]; then
                json_out+=",\"block_height\":\"${BLOCKS[$svc]}\""
            fi
            json_out+="}"
            first=0
        fi
    done
    json_out+="}"
    echo "$json_out"
else
    for svc in "${SERVICES[@]}"; do
        if [[ -n "${STATUS[$svc]:-}" ]]; then
            case "${STATUS[$svc]}" in
                healthy)  qok "$svc: ${MESSAGE[$svc]}" ;;
                warning) warn "$svc: ${MESSAGE[$svc]}" ;;
                critical) err "$svc: ${MESSAGE[$svc]}" ;;
            esac
        fi
    done
fi

exit $OVERALL
