#!/usr/bin/env bash
# shellcheck source=/dev/null
#
# Shared helpers for Xenomorph AI devnet scripts.
# Intended to be sourced, not executed directly.

set -uo pipefail

# -----------------------------------------------------------------------------
# Colors
# -----------------------------------------------------------------------------
if [[ -t 1 && "${NO_COLOR:-0}" != "1" ]]; then
    readonly C_RED='\033[0;31m'
    readonly C_GREEN='\033[0;32m'
    readonly C_YELLOW='\033[1;33m'
    readonly C_BLUE='\033[0;34m'
    readonly C_NC='\033[0m'
else
    readonly C_RED=''
    readonly C_GREEN=''
    readonly C_YELLOW=''
    readonly C_BLUE=''
    readonly C_NC=''
fi

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
log()   { echo -e "${C_BLUE}[INFO]${C_NC}  $*"; }
warn()  { echo -e "${C_YELLOW}[WARN]${C_NC}  $*" >&2; }
err()   { echo -e "${C_RED}[ERROR]${C_NC} $*" >&2; }
ok()    { echo -e "${C_GREEN}[OK]${C_NC}   $*"; }

# -----------------------------------------------------------------------------
# Verbosity
# -----------------------------------------------------------------------------
debug() {
    if [[ "${XENO_VERBOSE:-0}" == "1" ]]; then
        echo -e "${C_BLUE}[DEBUG]${C_NC} $*" >&2
    fi
}

# -----------------------------------------------------------------------------
# Quiet mode
# -----------------------------------------------------------------------------
is_quiet() {
    [[ "${XENO_QUIET:-0}" == "1" ]]
}

qlog() { is_quiet || log "$@"; }
qwarn() { is_quiet || warn "$@"; }
qerr() { err "$@"; }
qok() { is_quiet || ok "$@"; }

# -----------------------------------------------------------------------------
# Load .env
# -----------------------------------------------------------------------------
load_env() {
    local env_file="${1:-.env}"
    if [[ -f "$env_file" ]]; then
        # Export all non-comment, non-empty lines
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" =~ ^# ]] && continue
            # Trim whitespace and quotes
            key="$(echo "$key" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            value="$(echo "$value" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//;s/^"//;s/"$//')"
            [[ -z "$key" ]] && continue
            export "$key=$value"
        done < "$env_file"
        debug "Loaded environment from $env_file"
    else
        warn "$env_file not found, using defaults"
    fi
}

# -----------------------------------------------------------------------------
# Docker / Docker Compose detection
# -----------------------------------------------------------------------------
detect_compose() {
    if docker compose version &>/dev/null; then
        echo "docker compose"
    elif docker-compose version &>/dev/null; then
        echo "docker-compose"
    else
        echo ""
    fi
}

COMPOSE_CMD="${COMPOSE_CMD:-$(detect_compose)}"

require_docker() {
    if ! command -v docker &>/dev/null; then
        err "docker is not installed or not in PATH"
        exit 1
    fi
    if [[ -z "$COMPOSE_CMD" ]]; then
        err "docker compose plugin or docker-compose is required"
        exit 1
    fi
}

compose_file() {
    echo "${XENO_COMPOSE_FILE:-docker-compose.devnet.yml}"
}

docker_compose() {
    $COMPOSE_CMD -f "$(compose_file)" "$@"
}

# -----------------------------------------------------------------------------
# Random value generation
# -----------------------------------------------------------------------------
generate_secret() {
    openssl rand -hex 32 2>/dev/null || head -c 64 /dev/urandom | xxd -p | tr -d '\n'
}

# -----------------------------------------------------------------------------
# Port availability
# -----------------------------------------------------------------------------
is_port_free() {
    local port="$1"
    if command -v nc &>/dev/null; then
        ! nc -z 127.0.0.1 "$port" 2>/dev/null
    elif command -v ss &>/dev/null; then
        ! ss -tln 2>/dev/null | grep -q ":$port "
    elif command -v netstat &>/dev/null; then
        ! netstat -tln 2>/dev/null | grep -q ":$port "
    else
        true
    fi
}

check_ports() {
    local ports=("$@")
    local failed=0
    for port in "${ports[@]}"; do
        if ! is_port_free "$port"; then
            err "Port $port is already in use"
            failed=1
        fi
    done
    return $failed
}

# -----------------------------------------------------------------------------
# Node RPC helpers
# -----------------------------------------------------------------------------
get_block_height() {
    local host="${1:-127.0.0.1}"
    local port="${2:-16110}"
    local height
    height="$(timeout 5 curl -s -X POST "http://$host:$port" \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","method":"getBlockDagInfo","params":{},"id":1}' 2>/dev/null | jq -r '.result.virtualSelectedParentBlueScore // empty' 2>/dev/null || true)"
    if [[ -z "$height" ]]; then
        height="$(timeout 5 curl -s -X POST "http://$host:$port" \
            -H 'Content-Type: application/json' \
            -d '{"jsonrpc":"2.0","method":"getInfo","params":{},"id":1}' 2>/dev/null | jq -r '.result // empty' 2>/dev/null || true)"
    fi
    echo "$height"
}

get_peer_count() {
    local host="${1:-127.0.0.1}"
    local port="${2:-16110}"
    local peers
    peers="$(timeout 5 curl -s -X POST "http://$host:$port" \
        -H 'Content-Type: application/json' \
        -d '{"jsonrpc":"2.0","method":"getConnectedPeerInfo","params":{},"id":1}' 2>/dev/null | jq '.result | length' 2>/dev/null || echo 0)"
    echo "$peers"
}

# -----------------------------------------------------------------------------
# OS detection
# -----------------------------------------------------------------------------
detect_os() {
    local os
    os="$(uname -s | tr '[:upper:]' '[:lower:]')"
    case "$os" in
        linux*)  echo "linux" ;;
        darwin*) echo "macos" ;;
        msys*|cygwin*|mingw*) echo "windows" ;;
        *)       echo "$os" ;;
    esac
}

# -----------------------------------------------------------------------------
# Command helpers
# -----------------------------------------------------------------------------
has_command() {
    command -v "$1" &>/dev/null
}

require_command() {
    local cmd="$1"
    if ! has_command "$cmd"; then
        err "$cmd is required but not installed"
        exit 1
    fi
}

# -----------------------------------------------------------------------------
# Trap and cleanup helpers
# -----------------------------------------------------------------------------
cleanup_pids=()
register_cleanup() {
    cleanup_pids+=("$1")
}

on_exit() {
    for pid in ${cleanup_pids[@]+"${cleanup_pids[@]}"}; do
        kill "$pid" 2>/dev/null || true
    done
}

trap on_exit EXIT

# -----------------------------------------------------------------------------
# Usage helpers
# -----------------------------------------------------------------------------
print_help_and_exit() {
    local usage="$1"
    local exit_code="${2:-0}"
    echo "$usage"
    exit "$exit_code"
}
