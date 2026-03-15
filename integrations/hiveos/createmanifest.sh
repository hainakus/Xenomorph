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
CUSTOM_NAME="${MINER_NAME}"
CUSTOM_VERSION="${MINER_VER}"
CUSTOM_CONFIG_FILENAME="/hive/miners/custom/${MINER_NAME}/${MINER_NAME}.conf"
CUSTOM_LOG_BASENAME="/var/log/miner/${MINER_NAME}/${MINER_NAME}"
EOF

echo "Created: ${SCRIPT_DIR}/h-manifest.conf  (v${MINER_VER})"
