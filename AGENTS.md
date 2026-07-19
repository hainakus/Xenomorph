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

## GenomePoW (`seed-node/src/genome/`)

- `.xenom` archives are produced by `genome-freeze` from GRCh38 FASTA files and use the `XENOGEN1` header layout (64-byte header + 2-bit packed DNA).
- `GenomeArchive` parses `.xenom` archives (2-bit DNA encoding: A=00, C=01, G=10, T=11; four bases per byte, MSB-first).
- `GenomeBatchGenerator` produces deterministic `GenomeTrainingBatch` slices from a 32-byte seed.
- `GenomeStorage` caches archives locally and downloads missing ones via HTTP or an IPFS gateway. The default fallback is the canonical GitHub Release `grch38.xenom` (override with `XENO_GENOME_URL`).
- The seed-node WebSocket server handles `RpcRequest::GetGenomeTrainingBatch` and replies with `RpcResponse::GenomeTrainingBatch` (extracted DNA sequences).
- New RPC enum variants are appended at the end to keep binary compatibility with older miners.

### Build & test

```bash
cargo build -p seed-node -p xenom-miner
cargo test -p seed-node -p xenom-miner
```

## Miner (`xenom-miner`)

### Build & test

```bash
cargo build -p xenom-miner
cargo test -p xenom-miner -p seed-node
```

### CLI trainer selection

- `--trainer mock` — fast CPU-less mock trainer (default for devnet).
- `--trainer cpu` — legacy Candle MLP trainer.
- `--trainer dnabert2` — real DNABERT-2 training; requires a seed-node that serves `GetModelCheckpoint`.
- `--mock-mode` / `--mock` are hidden aliases for `--trainer=mock`.

Examples:

```bash
# Devnet (mock)
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer mock --dry-run

# Real DNABERT-2 training
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer dnabert2

# DNABERT-2 training on a genome archive served by the seed-node
# --network derives the canonical genome merkle root from consensus Params.
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer dnabert2 --network mainnet

# Override the genome merkle root explicitly
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer dnabert2 --genome-merkle <64-hex-chars>
```

### End-to-end DNABERT-2 devnet test

- Script: `scripts/test-dnabert2-devnet.sh`
- HITL: requires ~8 GB RAM and several CPU cores to download the 110M model and run real MLM training.
- Validates: model download, real `loss_before`/`loss_after` values, and block submission.
- Run: `./scripts/test-dnabert2-devnet.sh --duration 300`

## Devnet Deployment Scripts

- Location: `scripts/`
- Compose: `docker-compose.devnet.yml`
- Config template: `.env.example`

### Quick start

```bash
cp .env.example .env
./scripts/quick-devnet.sh          # build images, start stack
./scripts/check-devnet-health.sh   # verify services
./scripts/monitor-dashboard.sh     # live dashboard
./scripts/cleanup-devnet.sh --volumes
```

### Makefile targets

```bash
make setup  # quick-devnet.sh
make test   # health + governance + payment tests
make clean  # cleanup-devnet.sh
```

### Notes

- `build-devnet.sh` compiles `xenom`, `seed-node`, and `xenom-miner` binaries locally on Linux, or inside Docker on macOS via `--docker-build`.
- `build-devnet-macos.sh` is a macOS wrapper around `build-devnet.sh` that forces Docker builds and sets `DOCKER_DEFAULT_PLATFORM` to `linux/arm64` (Apple Silicon) or `linux/amd64` (Intel) so images are built for the native host architecture.
- `seed-node/src/main.rs` reads `XENO_NODE_RPC`, `XENO_GRPC_ADDR`, `XENO_MODELS_DIR`, `XENO_MINER_WS_ADDR`, and `XENO_DEFAULT_MODEL_ID` from the environment so it can reach the node in Docker networking.
- The seed-node downloads the default Hugging Face model (`multimolecule/dnabert2`) into `XENO_MODELS_DIR` on startup if it is not already present.
