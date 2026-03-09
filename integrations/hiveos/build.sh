#!/usr/bin/env bash
# ── build.sh ──────────────────────────────────────────────────────────────────
# Builds genome-miner for HiveOS deployment.
#
# Usage (from repo root):
#   bash integrations/hiveos/build.sh [--package]
#
# Options:
#   --package   After building, create the distributable tar.gz archive
#               (genome-miner-<version>-linux.tar.gz) ready for HiveOS upload.
#
# Targets tried (in order):
#   1. x86_64-unknown-linux-musl  (fully static, best for HiveOS portability)
#   2. x86_64-unknown-linux-gnu   (dynamic, fallback)
#
# The resulting binary is placed at:
#   integrations/hiveos/genome-miner
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

MINER_NAME="genome-miner"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT_DIR="${REPO_ROOT}/integrations/hiveos"
PACKAGE=false

for arg in "$@"; do
    case "$arg" in
        --package) PACKAGE=true ;;
    esac
done

# ── Read version from workspace Cargo.toml ────────────────────────────────────
MINER_VER=$(grep -m1 '^version' "${REPO_ROOT}/Cargo.toml" | sed 's/.*= *"\(.*\)"/\1/')
echo "Building ${MINER_NAME} v${MINER_VER} for HiveOS ..."

cd "${REPO_ROOT}"

# ── Choose build target ───────────────────────────────────────────────────────
MUSL_TARGET="x86_64-unknown-linux-musl"
GNU_TARGET="x86_64-unknown-linux-gnu"

if rustup target list --installed 2>/dev/null | grep -q "${MUSL_TARGET}"; then
    TARGET="${MUSL_TARGET}"
    echo "Using static musl target: ${TARGET}"
else
    echo "musl target not installed — falling back to ${GNU_TARGET}."
    echo "For a portable static binary run:"
    echo "  rustup target add ${MUSL_TARGET}"
    TARGET="${GNU_TARGET}"
fi

# ── Build ─────────────────────────────────────────────────────────────────────
if [ "${TARGET}" = "${MUSL_TARGET}" ]; then
    RUSTFLAGS="-C target-feature=+crt-static" \
    cargo build --release -p genome-miner --target "${TARGET}"
    BUILT_BIN="${REPO_ROOT}/target/${TARGET}/release/${MINER_NAME}"
else
    cargo build --release -p genome-miner --target "${TARGET}"
    BUILT_BIN="${REPO_ROOT}/target/${TARGET}/release/${MINER_NAME}"
fi

# ── Copy to integration dir ───────────────────────────────────────────────────
cp "${BUILT_BIN}" "${OUT_DIR}/${MINER_NAME}"
chmod +x "${OUT_DIR}/${MINER_NAME}"
echo "Binary: ${OUT_DIR}/${MINER_NAME}  ($(du -sh "${OUT_DIR}/${MINER_NAME}" | cut -f1))"

# ── Generate manifest ─────────────────────────────────────────────────────────
bash "${OUT_DIR}/createmanifest.sh"

# ── Package for HiveOS upload (optional) ─────────────────────────────────────
if [ "${PACKAGE}" = true ]; then
    ARCHIVE="${REPO_ROOT}/${MINER_NAME}-${MINER_VER}-linux.tar.gz"
    echo "Creating package: ${ARCHIVE}"
    tar -czf "${ARCHIVE}" \
        -C "${OUT_DIR}" \
        "${MINER_NAME}" \
        h-config.sh \
        h-run.sh \
        h-stats.sh \
        hive-manifest.conf
    echo "Package ready: ${ARCHIVE}  ($(du -sh "${ARCHIVE}" | cut -f1))"
    echo ""
    echo "Install on HiveOS rig:"
    echo "  scp ${ARCHIVE} user@rig:/tmp/"
    echo "  ssh user@rig 'cd /tmp && tar -xzf ${MINER_NAME}-${MINER_VER}-linux.tar.gz"
    echo "                -C /hive/miners/${MINER_NAME}/ --strip-components=0'"
fi

echo "Done."
