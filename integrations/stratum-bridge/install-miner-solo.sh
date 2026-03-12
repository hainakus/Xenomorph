#!/bin/bash

set -e

NODE="xenom.space:36669"
GENOME_DIR="$HOME/.rusty-xenom"
GENOME_FILE="$GENOME_DIR/grch38.xenom"
MINER_BIN="$HOME/genome-miner"
MINER_URL="https://github.com/hainakus/Xenomorph/releases/download/genome-miner-v0.4.0/genome-miner-linux-x86_64.tar.gz"
GENOME_URL="https://github.com/hainakus/Xenomorph/releases/download/genome-miner-v0.4.0/grch38.xenom"

echo "==== Xenom Genome Miner Installer (Solo / Public Node) ===="

if [ -z "$1" ]; then
    echo "Usage: $0 <xenom-wallet-address>"
    exit 1
fi

WALLET="$1"

echo "Wallet:  $WALLET"
echo "Node:    $NODE"

# install dependencies
echo "Installing dependencies..."
sudo apt-get update -qq
sudo apt-get install -y wget curl tar

# create genome directory
mkdir -p "$GENOME_DIR"

# download genome file
if [ ! -f "$GENOME_FILE" ]; then
    echo "Downloading genome file..."
    wget -q --show-progress -O "$GENOME_FILE" \
        "$GENOME_URL"
else
    echo "Genome file already exists: $GENOME_FILE"
fi

# download miner
if [ ! -f "$MINER_BIN" ]; then
    echo "Downloading genome-miner..."
    TMP_TAR="$(mktemp /tmp/genome-miner-XXXXXX.tar.gz)"
    wget -q --show-progress -O "$TMP_TAR" "$MINER_URL"
    tar -xzf "$TMP_TAR" -C "$HOME" --wildcards --no-anchored 'genome-miner' --strip-components=1 2>/dev/null \
        || tar -xzf "$TMP_TAR" -C "$HOME"
    chmod +x "$MINER_BIN"
    rm -f "$TMP_TAR"
else
    echo "Miner already exists: $MINER_BIN"
fi

echo ""
echo "Starting solo mining to node $NODE ..."
echo ""

exec "$MINER_BIN" gpu \
    --gpu all \
    --mainnet \
    --genome-file "$GENOME_FILE" \
    --rpcserver "$NODE" \
    --mining-address "$WALLET"
