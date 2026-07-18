#!/usr/bin/env bash
# Stress test the Xenomorph AI devnet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --miners <count>     Number of miners to scale (default: 5)
  --duration <time>    Test duration, e.g. 1h, 30m, 600s (default: 30m)
  --rps <number>       RPC requests per second (default: 10)
  --report-dir <dir>   Directory for report (default: ./reports)
  --report-name <name> Report filename (default: stress-report-<timestamp>.json)
  -q, --quiet          Minimal output
  -v, --verbose        Debug output
  -h, --help           Show this help and exit
"

MINERS=5
DURATION="30m"
RPS=10
REPORT_DIR="./reports"
REPORT_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --miners) MINERS="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --rps) RPS="$2"; shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        --report-name) REPORT_NAME="$2"; shift 2 ;;
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
require_command jq

RPC_PORT="${XENO_NODE_RPC_PORT:-16110}"

if ! [[ "$MINERS" =~ ^[0-9]+$ ]]; then
    err "--miners must be a non-negative integer"
    exit 1
fi
if ! [[ "$RPS" =~ ^[0-9]+$ ]]; then
    err "--rps must be a non-negative integer"
    exit 1
fi

# Convert duration to seconds
DURATION_SEC="$(echo "$DURATION" | awk '
    /[0-9]+s?$/ { gsub(/s$/,""); print $1 }
    /[0-9]+m$/ { gsub(/m$/,""); print $1 * 60 }
    /[0-9]+h$/ { gsub(/h$/,""); print $1 * 3600 }
    /[0-9]+d$/ { gsub(/d$/,""); print $1 * 86400 }
')"
if [[ -z "$DURATION_SEC" || ! "$DURATION_SEC" =~ ^[0-9]+$ ]]; then
    err "Invalid duration: $DURATION"
    exit 1
fi

REPORT_NAME="${REPORT_NAME:-stress-report-$(date +%Y%m%d-%H%M%S).json}"
mkdir -p "$REPORT_DIR"
REPORT_FILE="$REPORT_DIR/$REPORT_NAME"

# -----------------------------------------------------------------------------
# Scale miners
# -----------------------------------------------------------------------------
qlog "Scaling miners to $MINERS..."
if ! docker_compose up -d --scale xeno-miner="$MINERS" --no-recreate xeno-miner; then
    err "Failed to scale miners"
    exit 1
fi

# -----------------------------------------------------------------------------
# Collect start state
# -----------------------------------------------------------------------------
START_TIME="$(date +%s)"
START_HEIGHT="$(get_block_height 127.0.0.1 "$RPC_PORT")"
START_HEIGHT="${START_HEIGHT:-0}"

# -----------------------------------------------------------------------------
# Background RPC load generator
# -----------------------------------------------------------------------------
RPC_SUCCESS=0
RPC_FAILED=0
PIDS=()

rpc_load() {
    local end_time="$(($(date +%s) + DURATION_SEC))"
    local interval
    if [[ "$RPS" -gt 0 ]]; then
        interval="$(awk "BEGIN {printf \"%.4f\", 1.0/$RPS}")"
    else
        interval=1
    fi
    while [[ "$(date +%s)" -lt "$end_time" ]]; do
        if curl -s -X POST "http://127.0.0.1:$RPC_PORT" \
            -H 'Content-Type: application/json' \
            -d '{"jsonrpc":"2.0","method":"getBlockDagInfo","params":{},"id":1}' >/dev/null 2>&1; then
            RPC_SUCCESS=$((RPC_SUCCESS + 1))
        else
            RPC_FAILED=$((RPC_FAILED + 1))
        fi
        sleep "$interval"
    done
}

if [[ "$RPS" -gt 0 ]]; then
    qlog "Starting RPC load generator at $RPS rps for $DURATION..."
    for _ in $(seq 1 4); do
        rpc_load &
        PIDS+=("$!")
    done
fi

# -----------------------------------------------------------------------------
# Monitor
# -----------------------------------------------------------------------------
qlog "Monitoring for $DURATION..."
END_TIME="$(($(date +%s) + DURATION_SEC))"
MINER_LOGS_BEFORE="$(docker_compose logs --no-log-prefix --tail=1000 xeno-miner 2>/dev/null | grep -c 'Submitted block' || true)"

while [[ "$(date +%s)" -lt "$END_TIME" ]]; do
    sleep 5
    if ! is_quiet; then
        HEIGHT="$(get_block_height 127.0.0.1 "$RPC_PORT")"
        HEIGHT="${HEIGHT:-0}"
        printf "\rHeight: %-10s | RPC OK: %-5s | RPC FAIL: %-5s" "$HEIGHT" "$RPC_SUCCESS" "$RPC_FAILED"
    fi
done
printf "\n"

# -----------------------------------------------------------------------------
# Stop background load
# -----------------------------------------------------------------------------
for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
done

# -----------------------------------------------------------------------------
# Collect end state
# -----------------------------------------------------------------------------
END_TIME_VAL="$(date +%s)"
END_HEIGHT="$(get_block_height 127.0.0.1 "$RPC_PORT")"
END_HEIGHT="${END_HEIGHT:-0}"
MINER_LOGS_AFTER="$(docker_compose logs --no-log-prefix --tail=1000 xeno-miner 2>/dev/null | grep -c 'Submitted block' || true)"
BLOCKS_SUBMITTED=$((MINER_LOGS_AFTER - MINER_LOGS_BEFORE))
ELAPSED=$((END_TIME_VAL - START_TIME))
HEIGHT_GAIN=$((END_HEIGHT - START_HEIGHT))

# -----------------------------------------------------------------------------
# System stats
# -----------------------------------------------------------------------------
CPU_USAGE="N/A"
MEM_USAGE="N/A"
if command -v docker &>/dev/null; then
    STATS="$(docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' xeno-node xeno-seed xeno-miner 2>/dev/null || true)"
    if [[ -n "$STATS" ]]; then
        CPU_USAGE="$(echo "$STATS" | awk -F',' '{sum+=substr($1,1,length($1)-1)} END {printf "%.1f%%", sum}')"
        MEM_USAGE="$(echo "$STATS" | awk -F',' '{sum+=$2} END {print sum " / total"}')"
    fi
fi

# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------
cat > "$REPORT_FILE" <<EOF
{
  "duration_seconds": $ELAPSED,
  "miners": $MINERS,
  "rps_target": $RPS,
  "start_height": $START_HEIGHT,
  "end_height": $END_HEIGHT,
  "height_gain": $HEIGHT_GAIN,
  "blocks_submitted": $BLOCKS_SUBMITTED,
  "rpc_success": $RPC_SUCCESS,
  "rpc_failed": $RPC_FAILED,
  "cpu_usage": "$CPU_USAGE",
  "memory_usage": "$MEM_USAGE",
  "report_file": "$REPORT_FILE"
}
EOF

qok "Stress test complete. Report saved to $REPORT_FILE"
cat "$REPORT_FILE"

# -----------------------------------------------------------------------------
# Scale back
# -----------------------------------------------------------------------------
qlog "Scaling miners back to 1..."
docker_compose up -d --scale xeno-miner=1 --no-recreate xeno-miner || warn "Failed to scale miners back"

exit 0
