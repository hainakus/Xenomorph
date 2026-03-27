#!/bin/bash
set -euo pipefail

export WORKER_PRIVKEY="de29392403d841224062e48382801726a34aad6bc41d469a892807a9f3e3b41c"

DEFAULT_WORK_ROOT="/var/lib/xenom/l2-work"
FALLBACK_WORK_ROOT="/tmp/xenom/l2-work"
WORK_ROOT="${L2_WORK_ROOT:-$DEFAULT_WORK_ROOT}"

# If the configured work root isn't writable by the current user,
# fall back to a per-user writable directory to avoid os error 13.
if ! mkdir -p "$WORK_ROOT" 2>/dev/null || ! test -w "$WORK_ROOT"; then
  mkdir -p "$FALLBACK_WORK_ROOT"
  echo "[run_l2_worker] WARN: '$WORK_ROOT' is not writable by $(id -un); using '$FALLBACK_WORK_ROOT' instead." >&2
  WORK_ROOT="$FALLBACK_WORK_ROOT"
fi

./target/release/genetics-l2-worker \
  --coordinator http://10.0.0.240:8091 \
  --work-root "$WORK_ROOT" \
  --poll-ms 5000
