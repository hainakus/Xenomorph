#!/usr/bin/env bash
# Build Docker images for the Xenomorph AI devnet on macOS.
#
# This is a thin wrapper around build-devnet.sh that:
#   * Forces building inside Docker (cross-compiling Rust for Linux from macOS is
#     only practical with a cross-linker such as cargo-zigbuild).
#   * Sets DOCKER_DEFAULT_PLATFORM to linux/arm64 on Apple Silicon or
#     linux/amd64 on Intel Macs, so Docker Desktop builds the native image
#     architecture instead of emulating x86_64 on ARM via Rosetta/QEMU.
#
# Usage: ./scripts/build-devnet-macos.sh [OPTIONS]
# All options are forwarded to build-devnet.sh.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Thin wrapper for build-devnet.sh on macOS. All options are forwarded.

Common options:
  --no-cache        Build without cache
  --registry <url>  Prefix image names with registry
  --parallel        Build images in parallel
  --skip-anvil      Do not pull the anvil image
  -q, --quiet       Minimal output
  -v, --verbose     Debug output
  -h, --help        Show this help and exit

Examples:
  $(basename "$0")
  $(basename "$0") --no-cache --parallel

For a local (non-Docker) Rust build on macOS, install cargo-zigbuild and use
build-devnet.sh directly with --local --target <aarch64|x86_64>-unknown-linux-gnu.
EOF
}

# Parse our own --help; pass everything else through.
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        *) break ;;
    esac
done

require_docker

OS="$(detect_os)"
ARCH="$(uname -m)"

case "$ARCH" in
    arm64|aarch64)
        DOCKER_PLATFORM="linux/arm64"
        ;;
    x86_64|amd64)
        DOCKER_PLATFORM="linux/amd64"
        ;;
    *)
        warn "Unknown macOS architecture: $ARCH. Leaving DOCKER_DEFAULT_PLATFORM unset."
        DOCKER_PLATFORM=""
        ;;
esac

if [[ -n "$DOCKER_PLATFORM" ]]; then
    export DOCKER_DEFAULT_PLATFORM="$DOCKER_PLATFORM"
    qlog "macOS $ARCH detected; building for $DOCKER_PLATFORM"
fi

qlog "Invoking build-devnet.sh --docker-build $@"
exec "$SCRIPT_DIR/build-devnet.sh" --docker-build "$@"
