#!/usr/bin/env bash
# ── createmanifest.sh ─────────────────────────────────────────────────────────
# Generates hive-manifest.conf in the same directory as this script.
# Called automatically by build.sh, but can also be run standalone.
#
# HiveOS reads hive-manifest.conf when the miner package is installed to
# identify the miner name, version, supported algorithms and API port.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Read version from workspace Cargo.toml
MINER_VER=$(grep -m1 '^version' "${REPO_ROOT}/Cargo.toml" | sed 's/.*= *"\(.*\)"/\1/')
MINER_NAME="genome-miner"

cat > "${SCRIPT_DIR}/h-manifest.conf" << EOF
MINER_NAME="${MINER_NAME}"
MINER_VER="${MINER_VER}"
MINER_ALGO="genome-pow kheavyhash"
MINER_FORK="xenom"
MINER_LOG_BASENAME="${MINER_NAME}"
MINER_API_PORT="4000"
MINER_BIN="${MINER_NAME}"
EOF

echo "Created: ${SCRIPT_DIR}/h-manifest.conf  (v${MINER_VER})"
