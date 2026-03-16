#!/usr/bin/env bash
# Download Kaggle datasets to cache before starting the pool.
# Run this ONCE before ./script.sh — subsequent runs skip already-downloaded files.
#
# Usage:
#   chmod +x scripts/download-datasets.sh
#   ./scripts/download-datasets.sh
#
# Cache location: /tmp/kaggle-datasets/_cache/{competition}/
# The pool coordinator symlinks each job_id to the relevant cache dir.

set -euo pipefail

CACHE_BASE="/tmp/kaggle-datasets/_cache"

# ── Colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERR ]${NC} $*"; }

# ── Check kaggle CLI ──────────────────────────────────────────────────────────
if ! command -v kaggle &>/dev/null; then
  error "kaggle CLI not found. Install with: pip install kaggle"
  error "Then set up ~/.kaggle/kaggle.json with your API key."
  exit 1
fi

if [ ! -f "$HOME/.kaggle/kaggle.json" ]; then
  error "~/.kaggle/kaggle.json not found."
  error "Download it from https://www.kaggle.com/settings (API section)."
  exit 1
fi

chmod 600 "$HOME/.kaggle/kaggle.json"

# ── Download a competition ────────────────────────────────────────────────────
download_competition() {
  local SLUG="$1"
  local CACHE_DIR="$CACHE_BASE/$SLUG"
  local READY_MARKER="$CACHE_DIR/.ready"
  local LOCK_FILE="$CACHE_BASE/_${SLUG}.lock"

  mkdir -p "$CACHE_DIR"

  # Count existing audio files (recursive)
  local EXISTING
  EXISTING=$(find "$CACHE_DIR" \( -name '*.ogg' -o -name '*.wav' -o -name '*.mp3' -o -name '*.flac' \) 2>/dev/null | wc -l | tr -d ' ')

  if [ -f "$READY_MARKER" ]; then
    info "$SLUG: fully cached ($EXISTING audio files) — skipping"
    return 0
  fi

  if [ "$EXISTING" -gt "0" ]; then
    info "$SLUG: $EXISTING audio files found in cache — downloading only missing files"
  else
    info "$SLUG: starting fresh download → $CACHE_DIR"
  fi

  # Prevent parallel download of same slug
  if [ -f "$LOCK_FILE" ]; then
    warn "$SLUG: lock file exists ($LOCK_FILE) — another download may be running"
    warn "Remove it with: rm -f $LOCK_FILE"
    return 1
  fi

  touch "$LOCK_FILE"

  # Download (kaggle skips files that already exist with same size)
  info "$SLUG: running kaggle competitions download..."
  if kaggle competitions download -c "$SLUG" --path "$CACHE_DIR/" 2>&1; then
    # Extract any zip files (skip existing files with -n)
    local FOUND_ZIP=0
    for z in "$CACHE_DIR"/*.zip; do
      if [ -f "$z" ]; then
        FOUND_ZIP=1
        info "Extracting $(basename $z)..."
        unzip -n -q "$z" -d "$CACHE_DIR/" && rm -f "$z"
      fi
    done

    local FINAL_COUNT
    FINAL_COUNT=$(find "$CACHE_DIR" \( -name '*.ogg' -o -name '*.wav' -o -name '*.mp3' -o -name '*.flac' \) 2>/dev/null | wc -l | tr -d ' ')

    if [ "$FINAL_COUNT" -gt "0" ]; then
      touch "$READY_MARKER"
      info "$SLUG: READY — $FINAL_COUNT audio files in $CACHE_DIR"
    else
      warn "$SLUG: download completed but no audio files found (check zip contents)"
    fi
  else
    error "$SLUG: kaggle download failed"
    rm -f "$LOCK_FILE"
    return 1
  fi

  rm -f "$LOCK_FILE"
}

# ── Main ──────────────────────────────────────────────────────────────────────
mkdir -p "$CACHE_BASE"

info "Cache base: $CACHE_BASE"
echo ""

# BirdCLEF 2026
download_competition "birdclef-2026"

echo ""
info "Done. Cache status:"
for d in "$CACHE_BASE"/*/; do
  [ -d "$d" ] || continue
  SLUG=$(basename "$d")
  COUNT=$(find "$d" \( -name '*.ogg' -o -name '*.wav' -o -name '*.mp3' \) 2>/dev/null | wc -l | tr -d ' ')
  READY=""
  [ -f "$d/.ready" ] && READY=" [READY]"
  echo "  $SLUG: $COUNT audio files$READY"
done
