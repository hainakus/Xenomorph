#!/usr/bin/env bash
# Test USDT payment flow for AI inference on the Xenomorph AI devnet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --amount <number>      Payment amount in USDT (default: 0.1)
  --recipient <addr>     Recipient address (default: anvil account 2)
  --model-id <id>        Model ID for the query (default: dnabert2)
  --query-id <id>        Query ID (default: random)
  --stress <n>           Run n sequential payments
  --payments <addr>      Payments contract address (auto-detected if empty)
  --usdt <addr>          USDT contract address (auto-detected if empty)
  -q, --quiet            Minimal output
  -v, --verbose          Debug output
  -h, --help             Show this help and exit
"

AMOUNT="0.1"
RECIPIENT=""
MODEL_ID="dnabert2"
QUERY_ID=""
STRESS=0
PAYMENTS_ADDRESS=""
USDT_ADDRESS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --amount) AMOUNT="$2"; shift 2 ;;
        --recipient) RECIPIENT="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --query-id) QUERY_ID="$2"; shift 2 ;;
        --stress) STRESS="$2"; shift 2 ;;
        --payments) PAYMENTS_ADDRESS="$2"; shift 2 ;;
        --usdt) USDT_ADDRESS="$2"; shift 2 ;;
        -q|--quiet) XENO_QUIET=1; shift ;;
        -v|--verbose) XENO_VERBOSE=1; shift ;;
        -h|--help) print_help_and_exit "$USAGE" 0 ;;
        *) err "Unknown option: $1"; print_help_and_exit "$USAGE" 1 ;;
    esac
done

export XENO_QUIET="${XENO_QUIET:-0}"
export XENO_VERBOSE="${XENO_VERBOSE:-0}"

load_env
require_command cast
require_command jq

ANVIL_PORT="${XENO_ANVIL_PORT:-8545}"
RPC_URL="http://127.0.0.1:$ANVIL_PORT"
MNEMONIC="${XENO_FOUNDRY_MNEMONIC:-test test test test test test test test test test test junk}"
PAYMENTS_ADDRESS="${PAYMENTS_ADDRESS:-${XENO_PAYMENTS_ADDRESS:-}}"
USDT_ADDRESS="${USDT_ADDRESS:-${XENO_USDT_ADDRESS:-}}"
USER_INDEX="${XENO_TEST_ACCOUNT_INDEX:-1}"

if ! [[ "$STRESS" =~ ^[0-9]+$ ]]; then
    err "--stress must be a non-negative integer"
    exit 1
fi

if [[ -z "$QUERY_ID" ]]; then
    QUERY_ID="query-$(date +%s)-$(openssl rand -hex 4 2>/dev/null || head -c 8 /dev/urandom | xxd -p)"
fi

# Default recipient to anvil account 2 if not set
if [[ -z "$RECIPIENT" ]]; then
    RECIPIENT="$(cast wallet address --mnemonic "$MNEMONIC" --mnemonic-index 2)"
fi

# -----------------------------------------------------------------------------
# Deploy contracts if needed
# -----------------------------------------------------------------------------
deploy_if_needed() {
    if [[ -z "$USDT_ADDRESS" ]]; then
        qlog "Deploying MockUSDT..."
        USDT_BIN="tests/fixtures/contracts/MockUSDT.bin"
        if [[ ! -f "$USDT_BIN" ]]; then
            err "USDT bytecode not found"
            exit 1
        fi
        # Constructor takes initial supply (uint256). 1 billion USDT with 6 decimals.
        INITIAL_SUPPLY="1000000000000000"
        USDT_ADDRESS="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index 0 --create "$(cat "$USDT_BIN")" "$INITIAL_SUPPLY" --json 2>/dev/null | jq -r '.contractAddress // empty' || true)"
        if [[ -z "$USDT_ADDRESS" || "$USDT_ADDRESS" == "null" ]]; then
            err "Failed to deploy MockUSDT"
            exit 1
        fi
        qok "Deployed MockUSDT at $USDT_ADDRESS"
    fi

    if [[ -z "$PAYMENTS_ADDRESS" ]]; then
        qlog "Deploying InferencePayments..."
        PAYMENTS_BIN="tests/fixtures/contracts/InferencePayments.bin"
        if [[ ! -f "$PAYMENTS_BIN" ]]; then
            err "Payments bytecode not found"
            exit 1
        fi
        TREASURY="$(cast wallet address --mnemonic "$MNEMONIC" --mnemonic-index 0)"
        SEED="$(cast wallet address --mnemonic "$MNEMONIC" --mnemonic-index 3)"
        TRAINING="$(cast wallet address --mnemonic "$MNEMONIC" --mnemonic-index 4)"
        PAYMENTS_ADDRESS="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index 0 --create "$(cat "$PAYMENTS_BIN")" "$USDT_ADDRESS" "$TREASURY" "$SEED" "$TRAINING" --json 2>/dev/null | jq -r '.contractAddress // empty' || true)"
        if [[ -z "$PAYMENTS_ADDRESS" || "$PAYMENTS_ADDRESS" == "null" ]]; then
            err "Failed to deploy InferencePayments"
            exit 1
        fi
        qok "Deployed InferencePayments at $PAYMENTS_ADDRESS"
    fi

    if [[ -f ".env" ]]; then
        for var in XENO_USDT_ADDRESS XENO_PAYMENTS_ADDRESS; do
            val=""
            case "$var" in
                XENO_USDT_ADDRESS) val="$USDT_ADDRESS" ;;
                XENO_PAYMENTS_ADDRESS) val="$PAYMENTS_ADDRESS" ;;
            esac
            if grep -q "^${var}=" .env; then
                sed -i.bak "s/^${var}=.*/${var}=${val}/" .env && rm -f .env.bak
            else
                echo "${var}=${val}" >> .env
            fi
        done
    fi
}

deploy_if_needed

USDT_ABI='[{"constant":false,"inputs":[{"name":"_spender","type":"address"},{"name":"_value","type":"uint256"}],"name":"approve","outputs":[{"name":"","type":"bool"}],"type":"function"},{"constant":true,"inputs":[{"name":"_owner","type":"address"}],"name":"balanceOf","outputs":[{"name":"","type":"uint256"}],"type":"function"}]'
PAYMENTS_ABI="tests/fixtures/contracts/InferencePayments.abi"
if [[ ! -f "$PAYMENTS_ABI" ]]; then
    err "Payments ABI not found at $PAYMENTS_ABI"
    exit 1
fi

# -----------------------------------------------------------------------------
# Convert amount to USDT units (6 decimals)
# -----------------------------------------------------------------------------
AMOUNT_WEI="$(cast to-wei "$AMOUNT" 6 | tr -d '"')"
USER="$(cast wallet address --mnemonic "$MNEMONIC" --mnemonic-index "$USER_INDEX")"

# Mint USDT to user if using MockUSDT
qlog "Minting USDT to $USER..."
cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index 0 \
    "$USDT_ADDRESS" "mint(address,uint256)" "$USER" "$AMOUNT_WEI" >/dev/null

# -----------------------------------------------------------------------------
# Single or stress payment
# -----------------------------------------------------------------------------
run_payment() {
    local qid="$1"
    local start end gas tx status
    start="$(date +%s.%N)"

    # Approve
    cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$USER_INDEX" \
        "$USDT_ADDRESS" "approve(address,uint256)" "$PAYMENTS_ADDRESS" "$AMOUNT_WEI" >/dev/null

    # Initiate payment
    qid_hash="$(cast keccak "$qid")"

    BALANCE_BEFORE="$(cast call --rpc-url "$RPC_URL" "$USDT_ADDRESS" 'balanceOf(address)(uint256)' "$USER" | cast to-dec)"

    TX_RESULT="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$USER_INDEX" \
        "$PAYMENTS_ADDRESS" "initiatePayment(bytes32,string,uint256)" "$qid_hash" "$MODEL_ID" "$AMOUNT_WEI" --json 2>/dev/null)"
    TX_HASH="$(echo "$TX_RESULT" | jq -r '.transactionHash // empty')"
    gas="$(echo "$TX_RESULT" | jq -r '.gasUsed // empty')"
    if [[ -z "$TX_HASH" || "$TX_HASH" == "null" ]]; then
        err "Payment initiation failed"
        return 1
    fi

    # Complete payment (admin)
    cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index 0 \
        "$PAYMENTS_ADDRESS" "completePayment(bytes32)" "$qid_hash" >/dev/null

    end="$(date +%s.%N)"
    elapsed="$(awk "BEGIN {printf \"%.3f\", $end - $start}")"
    BALANCE_AFTER="$(cast call --rpc-url "$RPC_URL" "$USDT_ADDRESS" 'balanceOf(address)(uint256)' "$USER" | cast to-dec)"

    cat <<EOF
Payment Result:
  Query ID:      $qid
  TX Hash:       $TX_HASH
  Gas Used:      ${gas:-unknown}
  Amount:        $AMOUNT USDT
  Confirm Time:  ${elapsed}s
  Balance Before: $BALANCE_BEFORE
  Balance After:  $BALANCE_AFTER
EOF
}

if [[ "$STRESS" -gt "1" ]]; then
    qlog "Running $STRESS sequential payments..."
    for i in $(seq 1 "$STRESS"); do
        qid="$QUERY_ID-$i"
        run_payment "$qid"
    done
else
    run_payment "$QUERY_ID"
fi

exit 0
