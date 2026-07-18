#!/usr/bin/env bash
# Collect logs from the Xenomorph AI devnet services.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --since <time>      Collect logs since this duration, e.g. 1h, 24h (default: 24h)
  --service <name>    Collect only from this service
  --anonymize         Remove private keys and IPs
  --output <dir>      Output directory (default: ./logs)
  -q, --quiet         Minimal output
  -v, --verbose       Debug output
  -h, --help          Show this help and exit
"

SINCE="24h"
SERVICE=""
ANONYMIZE=0
OUTPUT_DIR="./logs"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --since) SINCE="$2"; shift 2 ;;
        --service) SERVICE="$2"; shift 2 ;;
        --anonymize) ANONYMIZE=1; shift ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
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

mkdir -p "$OUTPUT_DIR"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
ARCHIVE="$OUTPUT_DIR/xeno-devnet-logs-$TIMESTAMP.tar.gz"
TMP_DIR="$(mktemp -d)"

cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Collect logs
# -----------------------------------------------------------------------------
LOG_ARGS=(--no-color --timestamps --since "$SINCE")
if [[ -n "$SERVICE" ]]; then
    LOG_ARGS+=("$SERVICE")
    OUT_FILE="$TMP_DIR/${SERVICE}.log"
else
    OUT_FILE="$TMP_DIR/all-services.log"
fi

qlog "Collecting logs (since $SINCE)..."
if ! docker_compose logs "${LOG_ARGS[@]}" > "$OUT_FILE" 2>&1; then
    err "Failed to collect logs"
    exit 1
fi

# -----------------------------------------------------------------------------
# Anonymize
# -----------------------------------------------------------------------------
if [[ "$ANONYMIZE" == "1" ]]; then
    qlog "Anonymizing sensitive data..."
    sed -i.bak \
        -e 's/0x[a-fA-F0-9]\{64\}/<PRIVATE_KEY>/g' \
        -e 's/\b[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\.[0-9]\{1,3\}\b/<IP>/g' \
        "$OUT_FILE" && rm -f "$OUT_FILE.bak"
fi

# -----------------------------------------------------------------------------
# Add metadata
# -----------------------------------------------------------------------------
cat > "$TMP_DIR/metadata.txt" <<EOF
Collection timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Since: $SINCE
Service: ${SERVICE:-all}
Anonymized: $ANONYMIZE
Compose file: $(compose_file)
EOF

# -----------------------------------------------------------------------------
# Create archive
# -----------------------------------------------------------------------------
find "$TMP_DIR" -type f -print0 | tar -czf "$ARCHIVE" --null -T -

qok "Logs collected: $ARCHIVE"
echo "$ARCHIVE"
exit 0
