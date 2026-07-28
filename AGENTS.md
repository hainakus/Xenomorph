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
- The unified `xenom` WebSocket server (`xenom/src/training/websocket_server.rs`) handles `RpcRequest::GetGenomeTrainingBatch` and replies with `RpcResponse::GenomeTrainingBatch` (extracted DNA sequences). The standalone `seed-node` provides the same handler for legacy deployments.
- New RPC enum variants are appended at the end to keep binary compatibility with older miners.

### Build & test

```bash
cargo build -p xenom -p seed-node -p xenom-miner
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
- `--trainer mgm1` — Mini Genome Model (MGM-1) training; small genome-focused Transformer.
- `--trainer gpu` — same as `dnabert2` (GPU auto-select).
- `--trainer cuda` — force NVIDIA CUDA backend.
- `--trainer metal` — force Apple Metal backend.
- `--trainer rocm` — AMD ROCm/HIP (not yet implemented, returns a clear error).
- `--mock-mode` / `--mock` are hidden aliases for `--trainer=mock`.
- `--gpus <0,1,...>` — GPU device ordinals for multi-GPU training (default `0`).
- `--micro-batch-size <N>` — micro-batch size per GPU per accumulation step (default `1`).
- `--gradient-accumulation <N>` — number of gradient-accumulation steps (default `1`).
- `--fp16` — enable FP16 mixed precision on CUDA/Metal when available.
- `--gradient-checkpointing` — enable gradient checkpointing (stub).
- `--zero <N>` — ZeRO optimization level (stub, default `0`).

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
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 --trainer mock --dry-run

# MGM-1 training (default model, auto device)
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 --trainer mgm1

# Real DNABERT-2 training (auto GPU)
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 --trainer dnabert2

# Force CUDA with FP16 on a specific GPU
./target/release/xenom-miner --rpc-url ws://xeno-node:17110 --trainer cuda --gpus 0 --fp16 --micro-batch-size 1

# Multi-GPU CUDA with FP16 and gradient accumulation
./target/release/xenom-miner --rpc-url ws://xeno-node:17110 \
  --trainer cuda --gpus 0,1 --micro-batch-size 1 --gradient-accumulation 2 --fp16

# DNA training on a genome archive served by the unified xeno-node.
# --network selects the wallet address prefix; the real human GRCh38 genome
# merkle root is always used for DNA trainers (dnabert2, mgm1, gpu, cuda, rocm, metal).
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 --trainer mgm1 --network devnet

# Override the genome merkle root explicitly
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 --trainer mgm1 --genome-merkle <64-hex-chars>
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
- Native (no Docker): `scripts/run-native-devnet.sh` — builds/starts the unified `xenom` node and `xenom-miner` directly from `target/release`.
  - GPU auto-detection: when `XENO_MINER_TRAINER` is `dnabert2`, `gpu`, or `cuda` and both `nvidia-smi` and `nvcc` are present, the script compiles `xenom-miner` with `--features cuda`.
  - Override with `--features <features>` or `XENO_MINER_FEATURES` (e.g. `XENO_MINER_FEATURES=cuda ./scripts/run-native-devnet.sh --trainer cuda`).
  - Multi-GPU options are forwarded to the miner: `--gpus`, `--micro-batch-size`, `--gradient-accumulation`, `--fp16`, `--gradient-checkpointing`, `--zero`.

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

- `build-devnet.sh` compiles `xenom` and `xenom-miner` binaries locally on Linux, or inside Docker on macOS via `--docker-build`. `seed-node` is still built for standalone/legacy deployments but is no longer required in the unified devnet.
- `build-devnet-macos.sh` is a macOS wrapper around `build-devnet.sh` that forces Docker builds and sets `DOCKER_DEFAULT_PLATFORM` to `linux/arm64` (Apple Silicon) or `linux/amd64` (Intel) so images are built for the native host architecture.
- The unified `xenom` node reads `--models-dir`, `--miner-ws-listen`, and `--inference-grpc-listen` (or the corresponding `XENO_*` environment variables in scripts) and downloads or generates the default model (`xeno/mgm-1`) into `XENO_MODELS_DIR` on startup if it is not already present.
- `seed-node/src/main.rs` still reads `XENO_NODE_RPC`, `XENO_GRPC_ADDR`, `XENO_MODELS_DIR`, `XENO_MINER_WS_ADDR`, and `XENO_DEFAULT_MODEL_ID` for standalone/legacy deployments, but is not needed when `xenom` is run with `--miner-ws-listen` and `--inference-grpc-listen`.

## Wallet addresses and network prefixes

- `xenom-miner` derives the mining address from the BIP39 mnemonic using the network prefix selected via `--network` (`mainnet`/`testnet`/`devnet`/`simnet`).
- Valid prefixes are `xenom` (mainnet), `xenomtest` (testnet), `xenomdev` (devnet) and `xenomsim` (simnet).
- If `--wallet` is supplied, it is validated against the selected network. An address with the wrong prefix is rejected before training/submission starts.
- The same mnemonic produces a different address string for each network; only the prefix changes.

## Training block submission

There are two equivalent paths for a miner to submit a `TrainingBlock`:

1. **Unified `xenom` node (preferred):** `xenom/src/training/websocket_server.rs` accepts `SubmitBlock` over the miner WebSocket, and `xenom/src/training/coordinator.rs` validates it and submits the mined block through the local `RpcCoreService`.
2. **Legacy `seed-node` forwarding:** `seed-node/src/rpc/server.rs` forwards `SubmitBlock` training proofs to the Xenomorph full node over the Borsh `XenomorphRpcClient` (`XENO_NODE_RPC`, default `xeno-node:16112`). The `xenom` full node implements `TrainingBlockService` (`xenom/src/training_block_service.rs`) for this path.

In both cases the `xenom` full node:
  1. Validates the miner address and prefix against the running network.
  2. Deserializes the `TrainingProof`.
  3. Validates the proof against the active model: model id, base checkpoint, and loss improvement (`DifficultyTarget`).
  4. Builds a compact `CoinbaseExtraData` payload and requests a block template from the local consensus.
  5. Solves the block PoW with `kaspa_pow::State`.
  6. Submits the solved block via `submit_block_call` and returns `accepted` + `block_hash`.

- `--training-rpc-listen=<IP:PORT>` enables the legacy Borsh listener for standalone `seed-node` deployments.
- The active model defaults to `xeno/mgm-1` and its weights hash defaults to the network's `genome_merkle_root` (because the devnet miner uses the genome merkle root as the base checkpoint). Override with `--active-model-id` and `--active-model-weights-hash`.
- This is full-node-side training proof validation (the block is rejected before being built/submitted if the proof is invalid). Consensus-level validation in `Header`/`UsefulPoW` is still dead code and not yet wired into the block pipeline.
- `xenom-miner` `DnaBert2Trainer` clamps the batch `learning_rate` to `1e-5` for AdamW.
- `TrainingBlockService` devnet `DifficultyTarget` allows the loss to increase by up to `1.0` per batch; with synthetic/random devnet batches a pre-trained model may not improve in a single step.
- `TrainingBlockService` now mines the correct PoW for the active network: legacy KHeavyHash before `genome_pow_activation_daa_score`, and Genome PoW (with synthetic fragments) after it. This fixes the `block has invalid proof-of-work` rejections on devnet.

## MGM-1 multi-GPU and FedAvg

- `xenom-miner` supports data-parallel multi-GPU training for `xeno/mgm-1` via `Mgm1MultiGpuTrainer` (`xenom-miner/src/trainer/mgm1_multi_gpu.rs`). It is selected when `--gpus` lists more than one device and the model id contains `mgm-1`.
- The multi-GPU path uses one `Mgm1Trainer` replica per CUDA/Metal device and does not fall back to CPU; if no GPU is available it returns an error.
- Each replica computes gradients on its micro-batch, gradients are gathered/averaged on the master device, the master is updated, and replicas are restored to the shared base checkpoint before the next batch.
- MGM-1 trainers extract `GradientUpdate` payloads for seed-node FedAvg aggregation.
- `seed-node` (`seed-node/src/model/manager.rs`) aggregates MGM-1 gradients by loading the current model files, adding the FedAvg-averaged **weight-space delta** directly (`apply_mgm1_gradients_sync` with scale `1.0`), and storing the new safetensors checkpoint.  It does **not** re-optimize with a separate SGD step.
- MGM-1 FedAvg currently supports active-base gradients only; stale-but-related base gradients are rejected.
- `Mgm1Trainer` clamps the effective learning rate to `1e-3` and uses the shared `ManualAdamW` optimizer (also used by DNABERT-2) so named gradients can be applied directly.  Miners take `MGM1_LOCAL_STEPS = 16` local AdamW steps per genome batch so the small transformer can learn local context and avoid predicting only the majority base.
- `xenom` (`xenom/src/training/gradient_validator.rs`) verifies MGM-1 `GradientUpdate` payloads by loading the base checkpoint, preparing the exact batch, recomputing `loss_before`, decrypting the payload, verifying the gradient commitment, applying the weight-space delta, and comparing the resulting `loss_after`.  This avoids the cross-device numerical drift that made full CPU re-execution fail against Metal/CUDA miners.

```bash
# Multi-GPU MGM-1 training (CUDA example)
./target/release/xenom-miner --rpc-url ws://xeno-node:17110 \
  --trainer cuda --gpus 0,1 --micro-batch-size 1 --gradient-accumulation 1

# MGM-1 genome training on a served archive
./target/debug/xenom-miner --rpc-url ws://xeno-node:17110 \
  --trainer cuda --gpus 0,1 --network mainnet
```
