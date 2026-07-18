#!/usr/bin/env bash
# Test the complete governance lifecycle on the Xenomorph AI devnet.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

readonly USAGE="Usage: $(basename "$0") [OPTIONS]

Options:
  --proposal <type>      Proposal type: add-model, remove-model, update-params (default: add-model)
  --model-id <id>        Model ID for the proposal (default: test-v1)
  --model-repo <repo>    HuggingFace repo (default: xenom/test-v1)
  --cleanup              Remove the test proposal after execution
  --governance <addr>    Governance contract address (auto-detected from .env if empty)
  --account-index <n>    Anvil account index to use as proposer (default: 1)
  -q, --quiet            Minimal output
  -v, --verbose          Debug output
  -h, --help             Show this help and exit
"

PROPOSAL_TYPE="add-model"
MODEL_ID="test-v1"
MODEL_REPO="xenom/test-v1"
CLEANUP=0
GOV_ADDRESS=""
ACCOUNT_INDEX=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --proposal) PROPOSAL_TYPE="$2"; shift 2 ;;
        --model-id) MODEL_ID="$2"; shift 2 ;;
        --model-repo) MODEL_REPO="$2"; shift 2 ;;
        --cleanup) CLEANUP=1; shift ;;
        --governance) GOV_ADDRESS="$2"; shift 2 ;;
        --account-index) ACCOUNT_INDEX="$2"; shift 2 ;;
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
require_docker

ANVIL_PORT="${XENO_ANVIL_PORT:-8545}"
RPC_URL="http://127.0.0.1:$ANVIL_PORT"
MNEMONIC="${XENO_FOUNDRY_MNEMONIC:-test test test test test test test test test test test junk}"
GOV_ADDRESS="${GOV_ADDRESS:-${XENO_GOVERNANCE_ADDRESS:-}}"

# -----------------------------------------------------------------------------
# Contract artifacts
# -----------------------------------------------------------------------------
GOV_ABI="tests/fixtures/contracts/ModelGovernance.abi"
if [[ ! -f "$GOV_ABI" ]]; then
    err "Governance ABI not found at $GOV_ABI"
    exit 1
fi

# -----------------------------------------------------------------------------
# Deploy if address not set
# -----------------------------------------------------------------------------
if [[ -z "$GOV_ADDRESS" ]]; then
    qlog "No governance address configured; deploying ModelGovernance..."
    GOV_BIN="tests/fixtures/contracts/ModelGovernance.bin"
    if [[ ! -f "$GOV_BIN" ]]; then
        err "Governance bytecode not found at $GOV_BIN"
        exit 1
    fi
    GOV_ADDRESS="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index 0 --create "$(cat "$GOV_BIN")" --json 2>/dev/null | jq -r '.contractAddress // empty' || true)"
    if [[ -z "$GOV_ADDRESS" || "$GOV_ADDRESS" == "null" ]]; then
        err "Failed to deploy ModelGovernance"
        exit 1
    fi
    qok "Deployed ModelGovernance at $GOV_ADDRESS"
    # Persist to .env for subsequent runs
    if [[ -f ".env" ]]; then
        if grep -q "^XENO_GOVERNANCE_ADDRESS=" .env; then
            sed -i.bak "s/^XENO_GOVERNANCE_ADDRESS=.*/XENO_GOVERNANCE_ADDRESS=$GOV_ADDRESS/" .env && rm -f .env.bak
        else
            echo "XENO_GOVERNANCE_ADDRESS=$GOV_ADDRESS" >> .env
        fi
    fi
fi

# -----------------------------------------------------------------------------
# Encode proposal based on type
# -----------------------------------------------------------------------------
case "$PROPOSAL_TYPE" in
    add-model)
        # proposeModel(string modelId, string hfRepo, string hfRevision, bytes32 genesisCheckpoint, uint256 vram, uint256 reward, uint256 minStake)
        HF_REVISION="main"
        GENESIS="0x0000000000000000000000000000000000000000000000000000000000000000"
        VRAM=8
        REWARD=10000000000
        MIN_STAKE=10000
        DATA="$(cast calldata \
            'proposeModel(string,string,string,bytes32,uint256,uint256,uint256)' \
            "$MODEL_ID" "$MODEL_REPO" "$HF_REVISION" "$GENESIS" "$VRAM" "$REWARD" "$MIN_STAKE")"
        ;;
    remove-model)
        DATA="$(cast calldata 'proposeRemoveModel(string)' "$MODEL_ID")"
        ;;
    update-params)
        DATA="$(cast calldata 'proposeUpdateParams(string,uint256)' "$MODEL_ID" "20000000000")"
        ;;
    *)
        err "Unknown proposal type: $PROPOSAL_TYPE"
        exit 1
        ;;
esac

# -----------------------------------------------------------------------------
# Create proposal
# -----------------------------------------------------------------------------
qlog "Creating $PROPOSAL_TYPE proposal for model $MODEL_ID..."
PROPOSE_TX="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$ACCOUNT_INDEX" \
    "$GOV_ADDRESS" "$DATA" --json 2>/dev/null)"
PROPOSE_HASH="$(echo "$PROPOSE_TX" | jq -r '.transactionHash // empty')"
if [[ -z "$PROPOSE_HASH" || "$PROPOSE_HASH" == "null" ]]; then
    err "Failed to create proposal"
    exit 1
fi
qok "Proposal transaction: $PROPOSE_HASH"

# Fetch proposal id (proposalCount before create, or event)
PROPOSAL_ID="$(cast call --rpc-url "$RPC_URL" "$GOV_ADDRESS" \
    'proposalCount()(uint256)' 2>/dev/null | cast to-dec || true)"
if [[ -z "$PROPOSAL_ID" || "$PROPOSAL_ID" == "0" ]]; then
    PROPOSAL_ID="1"
fi
qlog "Proposal ID: $PROPOSAL_ID"

# -----------------------------------------------------------------------------
# Vote
# -----------------------------------------------------------------------------
qlog "Voting YES with test accounts..."
for idx in 1 2 3; do
    cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$idx" \
        "$GOV_ADDRESS" "vote(uint256,bool)" "$PROPOSAL_ID" true --async >/dev/null
    qlog "  Account $idx voted"
done

# -----------------------------------------------------------------------------
# Advance time & execute
# -----------------------------------------------------------------------------
qlog "Advancing chain time..."
cast rpc --rpc-url "$RPC_URL" anvil_increaseTime 691200 >/dev/null
cast rpc --rpc-url "$RPC_URL" anvil_mine >/dev/null

qlog "Executing proposal..."
EXEC_TX="$(cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$ACCOUNT_INDEX" \
    "$GOV_ADDRESS" "executeProposal(uint256)" "$PROPOSAL_ID" --json 2>/dev/null)"
EXEC_HASH="$(echo "$EXEC_TX" | jq -r '.transactionHash // empty')"
if [[ -z "$EXEC_HASH" || "$EXEC_HASH" == "null" ]]; then
    err "Failed to execute proposal"
    exit 1
fi
qok "Execution transaction: $EXEC_HASH"

# -----------------------------------------------------------------------------
# Verify
# -----------------------------------------------------------------------------
qlog "Verifying final state..."
IS_ACTIVE="$(cast call --rpc-url "$RPC_URL" "$GOV_ADDRESS" \
    'isModelActive(string,uint256)(bool)' "$MODEL_ID" "$(cast rpc --rpc-url "$RPC_URL" eth_blockNumber | cast to-dec)" 2>/dev/null | cast to-dec || echo 0)"

if [[ "$IS_ACTIVE" == "1" ]]; then
    qok "Model $MODEL_ID is active on-chain"
else
    err "Model $MODEL_ID is not active after execution"
    exit 1
fi

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
if [[ "$CLEANUP" == "1" ]]; then
    qlog "Cleaning up test proposal..."
    cast send --rpc-url "$RPC_URL" --mnemonic "$MNEMONIC" --mnemonic-index "$ACCOUNT_INDEX" \
        "$GOV_ADDRESS" "deprecateModel(string)" "$MODEL_ID" >/dev/null || warn "deprecateModel not available or failed"
fi

cat <<EOF
Governance Flow Result:
  Proposal type: $PROPOSAL_TYPE
  Model ID:      $MODEL_ID
  Proposal ID:   $PROPOSAL_ID
  Propose TX:    $PROPOSE_HASH
  Execute TX:    $EXEC_HASH
  Active:        yes
EOF

exit 0
