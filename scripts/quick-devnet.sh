#!/usr/bin/env bash
# One-command setup for the Xenomorph AI devnet.
#
# Exit codes:
#   0 success
#   1 missing prerequisite
#   2 build failed
#   3 start failed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  -b, --branch <branch>     Git branch to checkout (default: current branch)
  -e, --env <file>            Environment file (default: .env)
  -f, --compose-file <file>   Compose file (default: docker-compose.devnet.yml)
  --no-build                  Skip image build
  --no-anvil                  Skip anvil/EVM services
  -q, --quiet                 Minimal output
  -v, --verbose               Debug output
  -h, --help                  Show this help and exit
"

# Defaults
BRANCH=""
ENV_FILE=".env"
COMPOSE_FILE=""
NO_BUILD=0
NO_ANVIL=0

# -----------------------------------------------------------------------------
# Parse arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--branch) BRANCH="$2"; shift 2 ;;
        -e|--env) ENV_FILE="$2"; shift 2 ;;
        -f|--compose-file) COMPOSE_FILE="$2"; shift 2 ;;
        --no-build) NO_BUILD=1; shift ;;
        --no-anvil) NO_ANVIL=1; shift ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

# -----------------------------------------------------------------------------
# Load env
# -----------------------------------------------------------------------------
load_env "$ENV_FILE"
if [[ -n "$COMPOSE_FILE" ]]; then
    export XENO_COMPOSE_FILE="$COMPOSE_FILE"
fi

# -----------------------------------------------------------------------------
# OS & prerequisites
# -----------------------------------------------------------------------------
OS="$(detect_os)"
qlog "Detected OS: $OS"

require_docker
require_command git

if [[ "$NO_ANVIL" == "0" ]]; then
    # cast is used by governance/payment test scripts
    if ! has_command cast; then
        warn "cast (Foundry) not found; EVM test scripts will not work. Install with: brew install foundry"
    fi
fi

# -----------------------------------------------------------------------------
# Check ports
# -----------------------------------------------------------------------------
RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"
P2P_PORT="${XENO_NODE_P2P_PORT:-16111}"
SEED_PORT="${XENO_SEED_GRPC_PORT:-50051}"
ANVIL_PORT="${XENO_ANVIL_PORT:-8545}"

PORTS=("$RPC_PORT" "$P2P_PORT" "$SEED_PORT")
if [[ "$NO_ANVIL" == "0" ]]; then
    PORTS+=("$ANVIL_PORT")
fi

qlog "Checking ports: ${PORTS[*]}"
if ! check_ports "${PORTS[@]}"; then
    err "One or more required ports are unavailable"
    exit 1
fi
qok "All required ports are free"

# -----------------------------------------------------------------------------
# Ensure repo
# -----------------------------------------------------------------------------
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [[ -n "$BRANCH" ]]; then
    qlog "Checking out branch: $BRANCH"
    if ! git checkout "$BRANCH"; then
        err "Failed to checkout branch $BRANCH"
        exit 1
    fi
fi

# -----------------------------------------------------------------------------
# Generate .env if missing
# -----------------------------------------------------------------------------
if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f ".env.example" ]]; then
        qlog "Generating $ENV_FILE from .env.example"
        cp .env.example "$ENV_FILE"
        # Inject random secrets for devnet
        SECRET="$(generate_secret)"
        sed -i.bak "s/XENO_WALLET_PASSWORD=.*/XENO_WALLET_PASSWORD=${SECRET}/" "$ENV_FILE" && rm -f "$ENV_FILE.bak"
    else
        warn ".env.example not found; creating minimal .env"
        cat > "$ENV_FILE" <<'EOF'
XENO_VERSION=0.1.0
XENO_NODE_RPC_PORT=16110
XENO_NODE_P2P_PORT=16111
XENO_SEED_GRPC_PORT=50051
XENO_ANVIL_PORT=8545
XENO_MINER_TRAINER=mock
XENO_MINER_MODEL_ID=dnabert2
XENO_DATA_DIR=./devnet-data
XENO_COMPOSE_FILE=./docker-compose.devnet.yml
EOF
    fi
    load_env "$ENV_FILE"
fi

# -----------------------------------------------------------------------------
# Build images
# -----------------------------------------------------------------------------
if [[ "$NO_BUILD" == "0" ]]; then
    qlog "Building devnet images..."
    BUILD_ARGS=()
    [[ "$NO_ANVIL" == "1" ]] && BUILD_ARGS+=("--skip-anvil")
    if ! "$SCRIPT_DIR/build-devnet.sh" "${BUILD_ARGS[@]}"; then
        err "Image build failed"
        exit 2
    fi
fi

# -----------------------------------------------------------------------------
# Start stack
# -----------------------------------------------------------------------------
qlog "Starting devnet stack..."
PROFILES=()
if [[ "$NO_ANVIL" == "0" ]]; then
    PROFILES+=("--profile" "evm")
fi

if ! docker_compose up -d --remove-orphans "${PROFILES[@]}"; then
    err "Failed to start devnet stack"
    exit 3
fi

# -----------------------------------------------------------------------------
# Wait for services
# -----------------------------------------------------------------------------
qlog "Waiting for services to become healthy..."
RETRIES=30
for i in $(seq 1 $RETRIES); do
    if docker_compose ps xeno-node | grep -q healthy; then
        break
    fi
    if [[ "$i" == "$RETRIES" ]]; then
        err "xeno-node did not become healthy in time"
        docker_compose logs --tail=50 xeno-node
        exit 3
    fi
    sleep 2
done

# -----------------------------------------------------------------------------
# Final status
# -----------------------------------------------------------------------------
qok "Devnet stack is up"
qlog "Services:"
docker_compose ps

qlog ""
qlog "Next steps:"
qlog "  ./scripts/check-devnet-health.sh"
qlog "  ./scripts/test-governance-flow.sh --proposal add-model --model-id test-v1"
qlog "  ./scripts/test-payment-flow.sh --amount 0.1"
qlog "  ./scripts/monitor-dashboard.sh"

exit 0
