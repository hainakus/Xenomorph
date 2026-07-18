#!/usr/bin/env bash
# Build Docker images for the Xenomorph AI devnet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --no-cache            Build without cache
  --registry <url>        Prefix image names with registry (e.g. ghcr.io/xenom/)
  --parallel            Build images in parallel
  --local               Use pre-built local binaries instead of building
  --target <target>     Rust target for local build (default: native)
  --skip-anvil          Do not pull the anvil image
  -q, --quiet           Minimal output
  -v, --verbose         Debug output
  -h, --help            Show this help and exit
"

NO_CACHE=""
REGISTRY=""
PARALLEL=0
LOCAL=1
RUST_TARGET=""
SKIP_ANVIL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-cache) NO_CACHE="--no-cache"; shift ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --parallel) PARALLEL=1; shift ;;
        --local) LOCAL=1; shift ;;
        --target) RUST_TARGET="$2"; shift 2 ;;
        --skip-anvil) SKIP_ANVIL=1; shift ;;
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

REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VERSION="${XENO_VERSION:-0.1.0}"
TAGS=()

build_image() {
    local name="$1"
    local dockerfile="$2"
    local binary="$3"
    local image_name
    if [[ -n "$REGISTRY" ]]; then
        image_name="${REGISTRY%/}/$name"
    else
        image_name="$name"
    fi

    local full_tag="$image_name:$VERSION"
    local latest_tag="$name:latest"
    if [[ -n "$REGISTRY" ]]; then
        latest_tag="${REGISTRY%/}/$name:latest"
    fi

    qlog "Building $full_tag ..."
    local start_time
    start_time="$(date +%s)"

    local build_args=()
    if [[ "$LOCAL" == "1" ]]; then
        if [[ ! -f "$binary" ]]; then
            err "Local binary not found: $binary"
            return 1
        fi
        build_args+=("--build-arg" "BINARY_PATH=$binary")
    fi

    if ! docker build -f "$dockerfile" -t "$full_tag" -t "$latest_tag" $NO_CACHE "${build_args[@]}" .; then
        err "Failed to build $name"
        return 1
    fi

    local elapsed
    elapsed="$(($(date +%s) - start_time))"
    qok "Built $full_tag in ${elapsed}s"
    TAGS+=("$full_tag" "$latest_tag")
}

# -----------------------------------------------------------------------------
# Local Rust build
# -----------------------------------------------------------------------------
if [[ "$LOCAL" == "1" ]]; then
    require_command cargo
    OS="$(detect_os)"
    if [[ "$OS" == "macos" && -z "$RUST_TARGET" ]]; then
        if rustup target list --installed 2>/dev/null | grep -q "x86_64-unknown-linux-gnu"; then
            RUST_TARGET="x86_64-unknown-linux-gnu"
            warn "macOS detected; using target $RUST_TARGET for Linux containers"
        else
            warn "macOS detected without x86_64-unknown-linux-gnu target; Docker image may not run in Linux container"
            warn "Install with: rustup target add x86_64-unknown-linux-gnu"
            warn "Or use a Linux/WSL host. Continuing with native target at your own risk."
        fi
    fi

    CARGO_ARGS=()
    if [[ -n "$RUST_TARGET" ]]; then
        CARGO_ARGS+=("--target" "$RUST_TARGET")
    fi

    qlog "Building Rust binaries..."
    cargo build --release -p xenom "${CARGO_ARGS[@]}"
    cargo build --release -p seed-node "${CARGO_ARGS[@]}"
    cargo build --release -p xenom-miner "${CARGO_ARGS[@]}"

    BIN_PREFIX="target/release"
    if [[ -n "$RUST_TARGET" ]]; then
        BIN_PREFIX="target/$RUST_TARGET/release"
    fi
else
    BIN_PREFIX="target/release"
fi

# -----------------------------------------------------------------------------
# Docker build
# -----------------------------------------------------------------------------
IMAGES=("xeno-node:Dockerfile.xeno-node:$BIN_PREFIX/xenom" "xeno-seed:Dockerfile.xeno-seed:$BIN_PREFIX/seed-node" "xeno-miner:Dockerfile.xeno-miner:$BIN_PREFIX/xenom-miner")

if [[ "$PARALLEL" == "1" ]]; then
    pids=()
    for spec in "${IMAGES[@]}"; do
        IFS=':' read -r name dockerfile binary <<< "$spec"
        build_image "$name" "$dockerfile" "$binary" &
        pids+=("$!")
    done
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            failed=1
        fi
    done
    if [[ "$failed" == "1" ]]; then
        err "One or more image builds failed"
        exit 2
    fi
else
    for spec in "${IMAGES[@]}"; do
        IFS=':' read -r name dockerfile binary <<< "$spec"
        build_image "$name" "$dockerfile" "$binary"
    done
fi

# -----------------------------------------------------------------------------
# Pull anvil image if requested
# -----------------------------------------------------------------------------
if [[ "$SKIP_ANVIL" == "0" ]]; then
    ANVIL_IMAGE="${XENO_ANVIL_IMAGE:-ghcr.io/foundry-rs/foundry:stable}"
    qlog "Pulling anvil image: $ANVIL_IMAGE"
    if ! docker pull "$ANVIL_IMAGE"; then
        warn "Failed to pull $ANVIL_IMAGE; anvil service may not start"
    fi
fi

# -----------------------------------------------------------------------------
# Verify images
# -----------------------------------------------------------------------------
qlog "Verifying images..."
for tag in "${TAGS[@]}"; do
    if ! docker inspect --type=image "$tag" &>/dev/null; then
        err "Image not found: $tag"
        exit 2
    fi
    qok "Image exists: $tag"
done

qok "All images built successfully"
exit 0
