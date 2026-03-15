#!/usr/bin/env bash
# ── h-stats.sh ────────────────────────────────────────────────────────────────
# Sourced by the HiveOS agent every ~10 s to collect miner stats.
# MUST set two variables:
#   $khs   — total hashrate in kH/s (numeric)
#   $stats — JSON object with per-GPU arrays and share counts
# ─────────────────────────────────────────────────────────────────────────────

MINER_NAME="genome-miner"
API_PORT="${MINER_API_PORT:-4000}"
API_URL="http://127.0.0.1:${API_PORT}/"

# ── Defaults (miner not running or API unreachable) ───────────────────────────
khs=0
stats='{"hs":[],"hs_units":"hs","temp":[],"fan":[],"power":[],"accepted":0,"rejected":0,"algo":"genome-pow","ver":"0","uptime":0,"ar":[0,0]}'

# ── Check miner process ───────────────────────────────────────────────────────
if ! pgrep -x "${MINER_NAME}" > /dev/null 2>&1; then
    return 0 2>/dev/null || true
fi

# ── Fetch stats from miner API ────────────────────────────────────────────────
RESPONSE=$(curl -sf --connect-timeout 3 --max-time 5 "${API_URL}" 2>/dev/null)

if [[ -z "${RESPONSE}" ]] || ! echo "${RESPONSE}" | grep -q '"hs"'; then
    return 0 2>/dev/null || true
fi

# ── Parse khs from hs array (values are in H/s → divide by 1000) ─────────────
# Sum all GPU hashrates and convert H/s → kH/s
khs=$(echo "${RESPONSE}" | \
    grep -oP '"hs"\s*:\s*\[\K[^\]]*' | \
    tr ',' '\n' | \
    awk 'BEGIN{s=0} /^[0-9]/{s+=($1/1000)} END{printf "%.2f", s}')

khs="${khs:-0}"

# ── Export full stats JSON ────────────────────────────────────────────────────
stats="${RESPONSE}"
