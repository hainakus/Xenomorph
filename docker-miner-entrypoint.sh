#!/bin/bash
set -e

GENOME_FILE="${GENOME_DIR}/grch38.xenom"

if [ -z "${WALLET}" ]; then
  echo "ERROR: WALLET environment variable is required."
  echo "  docker run ... -e WALLET=xenom:your_address_here ..."
  exit 1
fi

# Download genome file if not already present (e.g. via volume mount)
if [ ! -f "${GENOME_FILE}" ]; then
  echo "Genome file not found at ${GENOME_FILE}. Downloading (~739 MB)..."
  wget -q --show-progress \
    "${GENOME_URL}" \
    -O "${GENOME_FILE}"
  echo "Genome file downloaded."
else
  echo "Genome file found at ${GENOME_FILE}."
fi

WORKER_NAME="${WORKER:-$(hostname)}"
# Strip leading "xenom:" prefix from wallet if already present to avoid doubling it
WALLET_ADDR="${WALLET#xenom:}"
STRATUM_WORKER="xenom:${WALLET_ADDR}.${WORKER_NAME}"

# Start a virtual X display so libGLX_nvidia.so can enumerate GPUs headlessly
Xvfb :99 -screen 0 1x1x24 &
export DISPLAY=:99

echo "Starting genome-miner..."
echo "  Stratum : ${STRATUM}"
echo "  Worker  : ${STRATUM_WORKER}"
echo "  GPU     : ${GPU}"

exec genome-miner gpu \
  --gpu "${GPU}" \
  --mainnet \
  --no-tui \
  --genome-file "${GENOME_FILE}" \
  --stratum "${STRATUM}" \
  --stratum-worker "${STRATUM_WORKER}" \
  "$@"
