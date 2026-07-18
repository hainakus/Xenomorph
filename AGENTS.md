# Agent Notes: Xenomorph Integration Tests

## Integration Test Suite

- Location: `tests/`
- Crate: `xenom-integration-tests`
- Entry point: `tests/integration/main.rs`

## How to run

```bash
# Sequential (recommended to avoid port clashes)
cargo test --test integration -- --nocapture --test-threads=1

# Parallel harness (serial_test still serialises the suite)
cargo test --test integration -- --nocapture --test-threads=4
```

## Requirements

- `anvil` from Foundry must be on `PATH` for EVM/governance/payment tests.
  - Install via Homebrew: `brew install foundry`
- All 11 integration tests must pass.

## Suite overview

- `test_miner_node.rs`: miner WebSocket RPC connection, batch training, block submission, reward ledger.
- `test_node_contract.rs`: governance proposal execution and active model registry reading.
- `test_governance_flow.rs`: full lifecycle proposal → vote → execute → mine.
- `test_payment_flow.rs`: USDT payment for inference via `InferencePayments` contract.
- `test_fault_tolerance.rs`: miner reconnect, invalid proof rejection, double-spend protection.

## Verification

```bash
cargo fmt --all -- --check
cargo clippy -p xenom-integration-tests -- -D warnings
cargo test --test integration
```

## Notes

- Tests spawn isolated Anvil devnets and mock WebSocket seed nodes; each test sets up and tears down its own environment.
- `hex_to_bytes32` in `tests/integration/fixtures.rs` converts arbitrary strings to `bytes32`: valid hex is decoded, short ASCII strings are left-padded, and long inputs are hashed with Keccak-256.
