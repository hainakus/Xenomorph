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
- `--trainer dnabert2` — real DNABERT-2 training; auto-selects GPU (CUDA/Metal) with CPU fallback.
- `--trainer gpu` — same as `dnabert2` (GPU auto-select).
- `--trainer cuda` — force NVIDIA CUDA backend.
- `--trainer metal` — force Apple Metal backend.
- `--trainer rocm` — AMD ROCm/HIP (not yet implemented, returns a clear error).
- `--mock-mode` / `--mock` are hidden aliases for `--trainer=mock`.
- `--gpu-device <N>` — GPU device ordinal (default 0).
- `--fp16` — enable FP16 mixed precision on CUDA/Metal when available.

Build with GPU support:

```bash
# NVIDIA CUDA
cargo build -p xenom-miner --release --features cuda

# Apple Metal
cargo build -p xenom-miner --release --features metal
```

Examples:

```bash
# Devnet (mock)
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer mock --dry-run

# Real DNABERT-2 training (auto GPU)
./target/debug/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer dnabert2

# Force CUDA with FP16 on a specific GPU
./target/release/xenom-miner --rpc-url ws://xeno-seed:17110 --trainer cuda --gpu-device 0 --fp16

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
- Native (no Docker): `scripts/run-native-devnet.sh` — builds/starts `xenom`, `seed-node` and `xenom-miner` directly from `target/release`.
  - GPU auto-detection: when `XENO_MINER_TRAINER` is `dnabert2`, `gpu`, or `cuda` and both `nvidia-smi` and `nvcc` are present, the script compiles `xenom-miner` with `--features cuda`.
  - Override with `--features <features>` or `XENO_MINER_FEATURES` (e.g. `XENO_MINER_FEATURES=cuda ./scripts/run-native-devnet.sh --trainer cuda`).

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

## Wallet addresses and network prefixes

- `xenom-miner` derives the mining address from the BIP39 mnemonic using the network prefix selected via `--network` (`mainnet`/`testnet`/`devnet`/`simnet`).
- Valid prefixes are `xenom` (mainnet), `xenomtest` (testnet), `xenomdev` (devnet) and `xenomsim` (simnet).
- If `--wallet` is supplied, it is validated against the selected network. An address with the wrong prefix is rejected before training/submission starts.
- The same mnemonic produces a different address string for each network; only the prefix changes.

## Seed-node -> Xenom node block forwarding

- `seed-node/src/rpc/server.rs` forwards `SubmitBlock` training proofs to the Xenomorph full node over the Borsh `XenomorphRpcClient` (`XENO_NODE_RPC`, default `xeno-node:16112`).
- The `xenom` full node now implements `TrainingBlockService` (`xenom/src/training_block_service.rs`), a dedicated Borsh listener that:
  1. Validates the miner address and prefix against the running network.
  2. Deserializes the `TrainingProof` from the seed-node.
  3. Validates the proof against the active model: model id, base checkpoint, and loss improvement (`DifficultyTarget`).
  4. Builds a compact `CoinbaseExtraData` payload and requests a block template from the local consensus.
  5. Solves the block PoW with `kaspa_pow::State`.
  6. Submits the solved block via `submit_block_call` and returns `accepted` + `block_hash` to the seed-node.
- The listener is enabled with `--training-rpc-listen=<IP:PORT>` and is wired into `daemon.rs` as an `AsyncService`.
- The active model defaults to `multimolecule/dnabert2` and its weights hash defaults to the network's `genome_merkle_root` (because the devnet miner uses the genome merkle root as the base checkpoint). Override with `--active-model-id` and `--active-model-weights-hash`.
- Native devnet and Docker `.env.example` point `XENO_NODE_RPC` to the training RPC port (`16112` by default) and expose/forward that port.
- If `--training-rpc-listen` is not provided, the listener is disabled and the seed-node will fail to connect as before.
- This is full-node-side training proof validation (the block is rejected before being built/submitted if the proof is invalid). Consensus-level validation in `Header`/`UsefulPoW` is still dead code and not yet wired into the block pipeline.
- The `model_id` field is now included in `TrainingBlock` so the seed-node can include it in the forwarded request.
