#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-iotapi322/xenom-l2-aio:v2}"
DOCKERFILE_PATH="l2-all-in-one.Dockerfile"
PUSH=1

cd "$REPO_ROOT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --push) PUSH=1; shift ;;
    --no-push) PUSH=0; shift ;;
    *) echo "Usage: $0 [--push|--no-push]"; exit 1 ;;
  esac
done

if [[ "$IMAGE_TAG" =~ [A-Z] ]]; then
  echo "ERROR: IMAGE_TAG must be lowercase: $IMAGE_TAG"
  exit 1
fi

echo "[l2-docker] Building ${IMAGE_TAG} using ${DOCKERFILE_PATH}..."

if [[ "$PUSH" -eq 1 ]]; then
  docker buildx build -f "$DOCKERFILE_PATH" -t "$IMAGE_TAG" --push .
else
  docker buildx build -f "$DOCKERFILE_PATH" -t "$IMAGE_TAG" --load .
fi

echo "[l2-docker] Done."

