#!/usr/bin/env bash
# ── install.sh ────────────────────────────────────────────────────────────────
# Standalone one-shot installer for genome-miner on HiveOS (or any Linux rig).
#
# What it does:
#   1. Creates required directories
#   2. Downloads grch38.xenom from GitHub Releases (with resume support)
#   3. Copies miner files to /hive/miners/genome-miner/
#   4. Sets correct permissions
#
# Usage (run as root or with sudo on the HiveOS rig):
#   # From the packed release archive already extracted:
#   bash /hive/miners/genome-miner/install.sh
#
#   # Or directly from the internet:
#   curl -fsSL https://raw.githubusercontent.com/hainakus/Xenomorph/master/integrations/hiveos/install.sh | bash
#
# The genome dataset (~780 MB) is stored at:
#   ~/.rusty-xenom/grch38.xenom   (user home, default auto-detect path)
# A symlink is also created inside the miner dir for convenience.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MINER_NAME="genome-miner"
MINER_DIR="/hive/miners/${MINER_NAME}"
GENOME_URL="https://github.com/hainakus/Xenomorph/releases/download/genome-grch38-v0/grch38.xenom"
GENOME_DIR="${HOME}/.rusty-xenom"
GENOME_PATH="${GENOME_DIR}/grch38.xenom"
GENOME_MIN_BYTES=780000000

# ── Helpers ───────────────────────────────────────────────────────────────────
info()    { echo "[install]  $*"; }
success() { echo "[install] ✔ $*"; }
warn()    { echo "[install] ⚠ $*"; }
error()   { echo "[install] ✖ $*" >&2; exit 1; }

file_size() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1" 2>/dev/null || echo 0
}

genome_ok() {
    [[ -f "$1" ]] && [[ $(file_size "$1") -ge ${GENOME_MIN_BYTES} ]]
}

# ── 1. Directory setup ────────────────────────────────────────────────────────
info "Creating directories..."
mkdir -p "${GENOME_DIR}"
mkdir -p "${MINER_DIR}"
success "Directories ready."

# ── 2. Genome dataset ─────────────────────────────────────────────────────────
if genome_ok "${GENOME_PATH}"; then
    success "Genome dataset already present: ${GENOME_PATH}  ($(file_size "${GENOME_PATH}") bytes)"
else
    info "Genome dataset not found (or incomplete) — downloading GRCh38 (~780 MB)..."
    info "URL : ${GENOME_URL}"
    info "Dest: ${GENOME_PATH}"
    info "Download supports resume — safe to Ctrl-C and re-run."
    echo ""

    if command -v wget &>/dev/null; then
        wget -c --show-progress "${GENOME_URL}" -O "${GENOME_PATH}"
    elif command -v curl &>/dev/null; then
        curl -L -C - --progress-bar "${GENOME_URL}" -o "${GENOME_PATH}"
    else
        error "Neither wget nor curl is installed.\n  Run: apt-get install -y wget\n  Then re-run this script."
    fi

    echo ""
    if genome_ok "${GENOME_PATH}"; then
        success "Download complete: ${GENOME_PATH}  ($(file_size "${GENOME_PATH}") bytes)"
    else
        ACTUAL=$(file_size "${GENOME_PATH}")
        warn "File may be incomplete: ${ACTUAL} bytes (expected >= ${GENOME_MIN_BYTES})."
        warn "Re-run this script to resume the download."
        # Keep partial file so wget/curl can resume; do not exit — partial is usable for devnet
    fi
fi

# ── 3. Symlink inside miner dir ───────────────────────────────────────────────
GENOME_LINK="${MINER_DIR}/grch38.xenom"
if [[ ! -e "${GENOME_LINK}" ]]; then
    ln -sf "${GENOME_PATH}" "${GENOME_LINK}"
    success "Symlink: ${GENOME_LINK} → ${GENOME_PATH}"
fi

# ── 4. Miner binary + scripts ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Copy scripts if running from the integration dir (not piped from curl)
if [[ -f "${SCRIPT_DIR}/${MINER_NAME}" ]]; then
    info "Installing miner binary..."
    cp -f "${SCRIPT_DIR}/${MINER_NAME}" "${MINER_DIR}/${MINER_NAME}"
    success "Binary installed: ${MINER_DIR}/${MINER_NAME}"
fi

for SCRIPT in h-config.sh h-run.sh h-stats.sh; do
    if [[ -f "${SCRIPT_DIR}/${SCRIPT}" ]]; then
        cp -f "${SCRIPT_DIR}/${SCRIPT}" "${MINER_DIR}/${SCRIPT}"
    fi
done

# ── 5. Permissions ────────────────────────────────────────────────────────────
info "Setting permissions..."
[[ -f "${MINER_DIR}/${MINER_NAME}" ]] && chmod 755 "${MINER_DIR}/${MINER_NAME}"
for SCRIPT in h-config.sh h-run.sh h-stats.sh; do
    [[ -f "${MINER_DIR}/${SCRIPT}" ]] && chmod 755 "${MINER_DIR}/${SCRIPT}"
done
success "Permissions set."

# ── 6. Manifest ───────────────────────────────────────────────────────────────
if [[ -f "${SCRIPT_DIR}/h-manifest.conf" ]]; then
    cp -f "${SCRIPT_DIR}/h-manifest.conf" "${MINER_DIR}/h-manifest.conf"
    success "Manifest installed."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════"
echo "  genome-miner installed successfully."
echo ""
echo "  Genome dataset : ${GENOME_PATH}"
echo "  Miner dir      : ${MINER_DIR}"
echo ""
echo "  HiveOS flight-sheet (Custom Miner):"
echo "    Miner name : genome-miner"
echo "    Wallet     : <your xenom: reward address>"
echo "    Pool URL   : <your node IP>:36669"
echo "    Extra args : --mainnet   (or --testnet)"
echo "════════════════════════════════════════════════════════"
