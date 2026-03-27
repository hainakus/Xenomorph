#!/usr/bin/env bash
set -euo pipefail

# Downloads and installs Xenom node release, starts the node, then prints a keypair.
# Usage:
#   bash install_xenom_node_and_keys.sh
# Optional env overrides:
#   XENOM_URL
#   INSTALL_DIR
#   DATA_DIR
#   RUN_NODE=1|0

XENOM_URL="${XENOM_URL:-https://github.com/hainakus/Xenomorph/releases/download/v0.15.5/xenom-linux-x86_64.tar.gz}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/.local/xenom}"
BIN_DIR="$INSTALL_DIR/bin"
DATA_DIR="${DATA_DIR:-$HOME/.xenom-node}"
RUN_NODE="${RUN_NODE:-1}"

mkdir -p "$BIN_DIR" "$DATA_DIR"

TARBALL="$(mktemp /tmp/xenom-XXXXXX.tar.gz)"
EXTRACT_DIR="$(mktemp -d /tmp/xenom-extract-XXXXXX)"

cleanup() {
  rm -f "$TARBALL"
  rm -rf "$EXTRACT_DIR"
}
trap cleanup EXIT

echo "[1/5] Downloading release..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 3 --connect-timeout 15 -o "$TARBALL" "$XENOM_URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$TARBALL" "$XENOM_URL"
else
  echo "ERROR: neither curl nor wget is installed."
  exit 1
fi

echo "[2/5] Extracting archive..."
tar -xzf "$TARBALL" -C "$EXTRACT_DIR"

# Copy executables into install bin dir
found_any=0
while IFS= read -r -d '' f; do
  found_any=1
  cp -f "$f" "$BIN_DIR/$(basename "$f")"
  chmod +x "$BIN_DIR/$(basename "$f")"
done < <(find "$EXTRACT_DIR" -type f -perm -u+x -print0)

if [[ "$found_any" -eq 0 ]]; then
  echo "ERROR: no executables found in release archive."
  exit 1
fi

echo "[3/5] Installed binaries to: $BIN_DIR"
ls -1 "$BIN_DIR" | sed 's/^/  - /'

# Ensure current shell can use binaries immediately
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
  export PATH="$BIN_DIR:$PATH"
fi

# Persist PATH update if not already present
if ! grep -qs "# xenom-bin-path" "$HOME/.bashrc"; then
  {
    echo ""
    echo "# xenom-bin-path"
    echo "export PATH=\"$BIN_DIR:\$PATH\""
  } >> "$HOME/.bashrc"
fi

# Pick node binary
NODE_BIN=""
for candidate in xenom kaspad; do
  if [[ -x "$BIN_DIR/$candidate" ]]; then
    NODE_BIN="$BIN_DIR/$candidate"
    break
  fi
done

if [[ -z "$NODE_BIN" ]]; then
  echo "WARNING: could not find node binary named 'xenom' or 'kaspad' in $BIN_DIR"
else
  echo "Node binary: $NODE_BIN"
fi

if [[ "$RUN_NODE" == "1" && -n "$NODE_BIN" ]]; then
  echo "[4/5] Starting node in background..."
  NODE_LOG="$DATA_DIR/node.log"
  NODE_PIDFILE="$DATA_DIR/node.pid"

  if [[ -f "$NODE_PIDFILE" ]] && kill -0 "$(cat "$NODE_PIDFILE")" 2>/dev/null; then
    echo "Node already running with PID $(cat "$NODE_PIDFILE")."
  else
    nohup "$NODE_BIN" --utxoindex >"$NODE_LOG" 2>&1 &
    echo $! > "$NODE_PIDFILE"
    sleep 2
    if kill -0 "$(cat "$NODE_PIDFILE")" 2>/dev/null; then
      echo "Node started. PID=$(cat "$NODE_PIDFILE")"
      echo "Log: $NODE_LOG"
    else
      echo "WARNING: node may have exited. Check: $NODE_LOG"
    fi
  fi
fi

echo "[5/5] Generating keypair..."

# Prefer dedicated keygen if available
if [[ -x "$BIN_DIR/xenom-stratum-bridge" ]]; then
  out="$($BIN_DIR/xenom-stratum-bridge --keygen 2>&1 || true)"
  echo "$out"
  priv="$(echo "$out" | sed -n 's/.*Private key[[:space:]]*:[[:space:]]*\([0-9a-fA-F]\{64\}\).*/\1/p' | head -n1)"
  pub="$(echo "$out" | sed -n 's/.*Pool address[[:space:]]*:[[:space:]]*\(xenom:[^[:space:]]*\).*/\1/p' | head -n1)"
  if [[ -n "$priv" && -n "$pub" ]]; then
    echo ""
    echo "PRIVATE_KEY=$priv"
    echo "PUBLIC_ADDRESS=$pub"
    exit 0
  fi
fi

# Fallback to rothschild one-shot key output (if present)
if [[ -x "$BIN_DIR/rothschild" ]]; then
  out="$($BIN_DIR/rothschild 2>&1 || true)"
  echo "$out"
  priv="$(echo "$out" | sed -n 's/.*Generated private key \([0-9a-fA-F]\+\) and address .*/\1/p' | head -n1)"
  pub="$(echo "$out" | sed -n 's/.*Generated private key [0-9a-fA-F]\+ and address \([^ .]*\).*/\1/p' | head -n1)"
  if [[ -n "$priv" && -n "$pub" ]]; then
    echo ""
    echo "PRIVATE_KEY=$priv"
    echo "PUBLIC_ADDRESS=$pub"
    exit 0
  fi
fi

cat <<'MSG'
Could not auto-generate keys from bundled binaries.
Try one of these manually (if available):
  xenom-stratum-bridge --keygen
  rothschild

If your release only contains the node daemon, use a wallet/bridge tool release to create a keypair.
MSG
