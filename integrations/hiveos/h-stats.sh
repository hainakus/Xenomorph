#!/usr/bin/env bash
# ── h-stats.sh ────────────────────────────────────────────────────────────────
# Called by HiveOS every ~10 s to collect miner stats.
#
# genome-miner exposes a HiveOS-compatible JSON stats API on port 4000
# (implemented in api.rs).  This script fetches it and outputs the JSON
# directly — HiveOS parses it to update the dashboard.
#
# Expected JSON fields (all returned by api.rs):
#   hs        – array of hashrates per GPU in H/s
#   hs_units  – "hs"
#   temp      – °C per GPU
#   fan       – % per GPU
#   power     – W per GPU
#   accepted  – total accepted shares
#   rejected  – total rejected shares
#   algo      – "genome-pow" or "kheavyhash"
#   ver       – miner version
#   uptime    – seconds since start
#   ar        – [accepted, rejected]
# ─────────────────────────────────────────────────────────────────────────────

MINER_NAME="genome-miner"
API_PORT="${MINER_API_PORT:-4000}"
API_URL="http://127.0.0.1:${API_PORT}/"
CONNECT_TIMEOUT=3
MAX_RETRIES=3

# ── Check miner process ───────────────────────────────────────────────────────
if ! pgrep -x "${MINER_NAME}" > /dev/null 2>&1; then
    echo "[h-stats] ${MINER_NAME} is not running"
    # Output a minimal JSON so HiveOS shows zeros instead of an error
    echo '{"hs":[],"hs_units":"hs","temp":[],"fan":[],"power":[],"accepted":0,"rejected":0,"algo":"genome-pow","ver":"0","uptime":0,"ar":[0,0]}'
    exit 0
fi

# ── Fetch stats from API ──────────────────────────────────────────────────────
for attempt in $(seq 1 ${MAX_RETRIES}); do
    RESPONSE=$(curl -sf \
        --connect-timeout ${CONNECT_TIMEOUT} \
        --max-time $((CONNECT_TIMEOUT + 2)) \
        "${API_URL}" 2>/dev/null)

    if [[ -n "${RESPONSE}" ]]; then
        break
    fi

    if [[ "${attempt}" -lt "${MAX_RETRIES}" ]]; then
        sleep 1
    fi
done

# ── Validate and output ───────────────────────────────────────────────────────
if [[ -z "${RESPONSE}" ]]; then
    echo "[h-stats] No response from ${API_URL} after ${MAX_RETRIES} attempts"
    echo '{"hs":[],"hs_units":"hs","temp":[],"fan":[],"power":[],"accepted":0,"rejected":0,"algo":"genome-pow","ver":"0","uptime":0,"ar":[0,0]}'
    exit 0
fi

# Validate it is JSON (basic check)
if ! echo "${RESPONSE}" | grep -q '"hs"'; then
    echo "[h-stats] Unexpected API response: ${RESPONSE}"
    exit 1
fi

# Output the JSON — HiveOS reads this from stdout
echo "${RESPONSE}"
