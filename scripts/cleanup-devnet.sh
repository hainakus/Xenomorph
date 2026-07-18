#!/usr/bin/env bash
# Clean up the Xenomorph AI devnet deployment.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --volumes        Remove volumes as well
  --images         Remove built images
  --purge-all      Remove wallet files and all data (use with care)
  --force          Skip confirmation
  -q, --quiet      Minimal output
  -v, --verbose    Debug output
  -h, --help       Show this help and exit
"

REMOVE_VOLUMES=0
REMOVE_IMAGES=0
PURGE_ALL=0
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --volumes) REMOVE_VOLUMES=1; shift ;;
        --images) REMOVE_IMAGES=1; shift ;;
        --purge-all) PURGE_ALL=1; shift ;;
        --force) FORCE=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

load_env
require_docker

# -----------------------------------------------------------------------------
# Confirmation
# -----------------------------------------------------------------------------
if [[ "$FORCE" == "0" ]]; then
    read -r -p "Stop and remove devnet containers? [y/N] " confirm
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        qlog "Cleanup cancelled"
        exit 0
    fi
    if [[ "$PURGE_ALL" == "1" ]]; then
        read -r -p "PURGE-ALL will delete wallet files and all data. Confirm? [y/N] " confirm2
        if [[ ! "$confirm2" =~ ^[Yy]$ ]]; then
            qlog "Cleanup cancelled"
            exit 0
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Disk before
# -----------------------------------------------------------------------------
DISK_BEFORE=0
if command -v df &>/dev/null; then
    DISK_BEFORE="$(df -k . | awk 'NR==2 {print $4}')"
fi

# -----------------------------------------------------------------------------
# Stop and remove containers
# -----------------------------------------------------------------------------
qlog "Stopping devnet..."
DOWN_ARGS=(--remove-orphans)
if [[ "$REMOVE_VOLUMES" == "1" ]]; then
    DOWN_ARGS+=(--volumes)
fi
if ! docker_compose down "${DOWN_ARGS[@]}"; then
    err "docker compose down failed"
    exit 1
fi

# -----------------------------------------------------------------------------
# Purge data
# -----------------------------------------------------------------------------
DATA_DIR="${XENO_DATA_DIR:-./devnet-data}"
if [[ "$PURGE_ALL" == "1" ]]; then
    qlog "Removing data directory: $DATA_DIR"
    rm -rf "$DATA_DIR"
elif [[ "$REMOVE_VOLUMES" == "1" ]]; then
    rm -rf "$DATA_DIR"/*
fi

# -----------------------------------------------------------------------------
# Remove images
# -----------------------------------------------------------------------------
if [[ "$REMOVE_IMAGES" == "1" ]]; then
    qlog "Removing images..."
    VERSION="${XENO_VERSION:-latest}"
    for img in xeno-node xeno-seed xeno-miner; do
        docker rmi "${img}:${VERSION}" "${img}:latest" 2>/dev/null || true
    done
fi

# -----------------------------------------------------------------------------
# Disk freed
# -----------------------------------------------------------------------------
DISK_AFTER=0
if command -v df &>/dev/null; then
    DISK_AFTER="$(df -k . | awk 'NR==2 {print $4}')"
fi
FREED_KB=$((DISK_AFTER - DISK_BEFORE))
if [[ "$FREED_KB" -lt 0 ]]; then
    FREED_KB=0
fi
FREED_MB="$((FREED_KB / 1024))"

qok "Cleanup complete. Disk space freed: ~${FREED_MB}MB"
exit 0
