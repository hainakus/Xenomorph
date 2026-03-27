#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ─────────────────────────────────────────────────────────────
IMAGE_REPO="${IMAGE_REPO:-iotapi322/xenom-node}"
IMAGE_TAG="${IMAGE_TAG:-v2}"
DOCKERFILE="${DOCKERFILE:-Dockerfile.node}"
CONTEXT_DIR="${CONTEXT_DIR:-$(dirname "$(realpath "$0")")}"

FULL_IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"

# ─── Parse flags ───────────────────────────────────────────────────────────────
PUSH=false
NO_CACHE=false

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Build the xenom-node Docker image from source.

Options:
  --push          Push the image to Docker Hub after building
  --no-cache      Build without Docker layer cache
  --tag TAG       Override image tag        (default: ${IMAGE_TAG})
  --repo REPO     Override image repository (default: ${IMAGE_REPO})
  -h, --help      Show this help message

Environment variables (override defaults):
  IMAGE_REPO      Repository name  (default: ${IMAGE_REPO})
  IMAGE_TAG       Image tag        (default: ${IMAGE_TAG})
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --push)      PUSH=true ;;
        --no-cache)  NO_CACHE=true ;;
        --tag)       IMAGE_TAG="$2"; FULL_IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"; shift ;;
        --repo)      IMAGE_REPO="$2"; FULL_IMAGE="${IMAGE_REPO}:${IMAGE_TAG}"; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Unknown option: $1"; usage; exit 1 ;;
    esac
    shift
done

# ─── Build ─────────────────────────────────────────────────────────────────────
echo "==> Building ${FULL_IMAGE}"
echo "    Dockerfile : ${DOCKERFILE}"
echo "    Context    : ${CONTEXT_DIR}"

BUILD_ARGS=(
    build
    --pull
    --file "${CONTEXT_DIR}/${DOCKERFILE}"
    --tag  "${FULL_IMAGE}"
)

[[ "${NO_CACHE}" == "true" ]] && BUILD_ARGS+=(--no-cache)

docker "${BUILD_ARGS[@]}" "${CONTEXT_DIR}"

echo ""
echo "==> Build complete: ${FULL_IMAGE}"

# ─── Push ──────────────────────────────────────────────────────────────────────
if [[ "${PUSH}" == "true" ]]; then
    echo "==> Pushing ${FULL_IMAGE}"
    docker push "${FULL_IMAGE}"
    echo "==> Push complete"
fi

# ─── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "To run locally (matches xenom-node.service):"
echo "  docker run -d \\"
echo "    --name xenom-node \\"
echo "    -p 26666:26666 \\"
echo "    -p 16668:16668 \\"
echo "    -p 17110:17110 \\"
echo "    -v xenom-data:/root/.rusty-xenom \\"
echo "    ${FULL_IMAGE} \\"
echo "      --utxoindex \\"
echo "      --rpclisten=0.0.0.0:16668 \\"
echo "      --listen=0.0.0.0:26666 \\"
echo "      --rpclisten-borsh=0.0.0.0:17110 \\"
echo "      --addpeer=89.155.26.12:26666 \\"
echo "      --addpeer=84.247.131.3:26666 \\"
echo "      --addpeer=194.233.66.230:26666 \\"
echo "      --addpeer=213.199.56.32:26666 \\"
echo "      --addpeer=45.90.123.219:26666 \\"
echo "      --addpeer=45.90.123.88:26666"
