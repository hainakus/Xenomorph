# Xenomorph AI / Devnet Architecture

This document describes the architecture of the Xenomorph AI devnet branch as currently implemented. It focuses on the components that enable **Useful Proof-of-Work (UsefulPoW)** through AI model training and the supporting devnet tooling.

## 1. High-level overview

The devnet is composed of several Rust binaries and a set of deployment scripts. The core idea is that miners perform AI model training as proof-of-work instead of pure hash grinding, and submit a `TrainingProof` attached to a block. A seed node coordinates miners, serves model inference via gRPC, and downloads models from Hugging Face. An API gateway provides HTTP access to model inference and governance operations.

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Host / Docker                                    │
│  ┌─────────────┐    gRPC     ┌─────────────────┐    WebSocket   ┌──────────┐ │
│  │  xenom-node │◄───────────►│   seed-node     │◄──────────────►│xenom-miner│ │
│  │  :16110/11  │   TCP       │ :50051 / :17110 │   Borsh        │           │ │
│  └─────────────┘             │  + HF download  │                └───────────┘ │
│                              └────────┬────────┘                              │
│                                       │ HTTP gRPC :50051                      │
│                                       ▼                                       │
│                              ┌─────────────────┐                              │
│                              │  api-gateway    │                              │
│                              │    :3000        │                              │
│                              └────────┬────────┘                              │
│                                       │ HTTP :8545 (profile `evm`)             │
│                                       ▼                                       │
│                              ┌─────────────────┐                              │
│                              │     anvil       │                              │
│                              │  EVM devnet     │                              │
│                              └─────────────────┘                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Architecture constraints

The devnet follows these ownership rules:

- **Only `seed-node` downloads models from Hugging Face.**
- **Only `seed-node` stores model weights**, encrypted with AES-256-GCM under `XENO_MODELS_DIR`.
- **`xenom-miner` does not download from Hugging Face and does not persist model weights.** It may, however, request the checkpoint from `seed-node` and hold it in memory while training.
- **Only `xenom-miner` trains.**
- **`xenom-node` validates proofs and reaches consensus;** it does not hold or train models.

## 2. Core binaries and crates

### `xenom` — full node

- **Entry point:** `xenom/src/main.rs`
- **Role:** Kaspa-derived full node with UsefulPoW support.
- **Notable features:** `heap`, `devnet-prealloc`, `semaphore-trace`.
- **Exposes:** RPC on `XENO_NODE_RPC_PORT` (default 16110) and P2P on `XENO_NODE_P2P_PORT` (default 16111).
- **Key integration:** block header can carry a `TrainingProof` (`consensus/core/src/pow/training_proof.rs`).

### `seed-node` — model serving & miner coordination

- **Entry point:** `seed-node/src/main.rs`
- **Role:** Sits between the blockchain node and miners. It downloads AI models from Hugging Face, exposes them through a gRPC inference service, and distributes training batches to miners over WebSocket.
- **Runtime flow:**
  1. Reads `XENO_NODE_RPC`, `XENO_GRPC_ADDR`, `XENO_MINER_WS_ADDR`, `XENO_MODELS_DIR`, `XENO_DEFAULT_MODEL_ID` from the environment.
  2. Builds a `ModelManager` for encrypted local model storage.
  3. Spawns a background task to download the default model (`multimolecule/dnabert2`) from Hugging Face.
  4. Starts a gRPC inference service (`seed-node/src/serving/inference.rs`) on `XENO_GRPC_ADDR`.
  5. Starts a WebSocket miner server (`seed-node/src/rpc/server.rs`) on `XENO_MINER_WS_ADDR`.
- **Key modules:**
  - `seed-node/src/model/manager.rs` — loads/stores/caches models.
  - `seed-node/src/model/storage.rs` — AES-256-GCM encrypted model storage.
  - `seed-node/src/model/downloader.rs` — downloads `model.safetensors` / `pytorch_model.bin` from Hugging Face.
  - `seed-node/src/rpc/client.rs` — TCP/Borsh client to the `xenom` node.
  - `seed-node/src/rpc/server.rs` — WebSocket/Borsh server for miners.
  - `seed-node/src/serving/inference.rs` — gRPC `Predict`, `Embed`, `GetModelInfo`, `ListModels`, `HealthCheck`.
  - `seed-node/src/serving/proof.rs` — proof-of-service generator.

### `xenom-miner` — UsefulPoW miner

- **Entry point:** `xenom-miner/src/main.rs`
- **Role:** Connects to the seed node, fetches `TrainingBatch`es, trains a model, generates a ZK proof commitment, builds/sings a block, and submits it.
- **Runtime flow:**
  1. Loads or creates a BIP39 wallet (`xenom-miner/src/wallet/manager.rs`).
  2. Connects to `seed-node` via WebSocket (`xenom-miner/src/rpc/client.rs`).
  3. In a loop: fetch batch → train (in `tokio::task::spawn_blocking`) → generate proof → build and sign block → submit.
- **Trainer backends:**
  - `MockTrainer` (`xenom-miner/src/trainer/mock_trainer.rs`) — deterministic fake training, default in devnet.
  - `CpuTrainer` (`xenom-miner/src/trainer/cpu_trainer.rs`) — small Candle MLP on synthetic data.
- **Other key modules:**
  - `xenom-miner/src/prover/zk_prover.rs` — placeholder ZK proof (hash based).
  - `xenom-miner/src/block/builder.rs` — builds `TrainingBlock`, finds nonce, signs.
  - `xenom-miner/src/wallet/manager.rs` — secp256k1 signing, `xnom:` address derivation.
  - `xenom-miner/src/governance/voter.rs` — `ModelGovernance.sol` voter (ethers-rs).

### `api-gateway` — HTTP API

- **Entry point:** `api-gateway/src/main.rs`
- **Role:** Public HTTP layer for model inference and governance.
- **Routes:**
  - `GET /models`, `GET /models/{id}`
  - `POST /predict/{model_id}`, `GET /queries/{id}`, `POST /webhook/payment`
  - `GET/POST /governance/proposals`, `GET /governance/proposals/{id}`, `POST /governance/proposals/{id}/vote`, `POST /governance/proposals/{id}/execute`
  - `GET /governance/models`, `GET /governance/models/{id}`
  - `GET /health`
- **Key modules:**
  - `api-gateway/src/payments/verifier.rs` — USDT payment verification.
  - `api-gateway/src/governance/` — governance contract client and route handlers.

### `genome-miner` and `genome-freeze`

- `genome-miner` is a separate CPU/GPU PoW miner that operates on a packed GRCh38 genome dataset, distinct from the AI UsefulPoW miner.
- `genome-freeze` converts GRCh38 FASTA files into a 2-bit packed `.xenom` file and computes its Merkle root.
- These are part of the broader Xenomorph PoW design but are not on the AI training path.

### `mining` crate

- **Location:** `mining/src/`
- Contains Kaspa block template building and mempool management.
- The `mining/src/training/` submodule has:
  - `model_trainer.rs` — PyTorch-based trainer (`tch`, feature-gated `ai-training`).
  - `miner.rs` — `TrainingMiner` that produces a `TrainingProof`.
  - `job_manager.rs` — coordinates multiple training jobs and picks the best proof.
- It is currently **not wired** to `xenom-miner`; the `xenom-miner` uses its own trainers.

### `tests` / integration tests

- **Location:** `tests/integration/main.rs`
- Exercises miner connection, governance flow, payment flow, and fault tolerance.
- Depends on `xenom-miner`, `seed-node`, and `api-gateway` as library crates.

## 3. Data flow

### Miner ↔ Seed node (WebSocket, Borsh)

- **Port:** 17110
- **Messages:** `xenom-miner/src/rpc/messages.rs` and `seed-node/src/rpc/messages.rs`.
- **Requests from miner:**
  - `GetTrainingBatch { model_id }`
  - `SubmitBlock(TrainingBlock)`
  - `GetBalance { address }`
  - `GetDifficulty`
  - `Heartbeat`
- **Responses from seed node:**
  - `TrainingBatch(Option<TrainingBatch>)`
  - `BlockHash([u8; 32])`
  - `Balance(u64)`
  - `Difficulty([u8; 32])`
  - `Pong`
  - `Error(String)`

`TrainingBatch` shape:

```rust
pub struct TrainingBatch {
    pub batch_id: u64,
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub data_indices: Vec<u64>,
    pub target_improvement: f64,
    pub learning_rate: f32,
}
```

### Seed node ↔ Xenom node (TCP, Borsh)

- **Port:** 16110
- **Purpose:** seed node fetches the latest model checkpoint and submits training blocks to the full node.
- `seed-node/src/rpc/client.rs` defines `RpcMessage` variants such as `GetModelCheckpoint`, `SubmitTrainingBlock`, and `Ping`.

### API Gateway ↔ Seed node (gRPC / protobuf)

- **Port:** 50051
- **Service:** `xenom.inference.Inference` (proto compiled in `seed-node/src/serving/inference.rs`).
- Used for inference requests that may be gated by USDT payments verified by the gateway.

## 4. Configuration and environment variables

All AI/devnet services are configured through environment variables. The `.env.example` file contains the canonical list. Key variables:

| Variable | Used by | Default | Purpose |
|----------|---------|---------|---------|
| `XENO_NODE_RPC` | seed-node | `xeno-node:16110` | Address of the Xenomorph full node |
| `XENO_GRPC_ADDR` | seed-node | `0.0.0.0:50051` | gRPC inference bind address |
| `XENO_MINER_WS_ADDR` | seed-node | `0.0.0.0:17110` | WebSocket miner server bind address |
| `XENO_MODELS_DIR` | seed-node | `/data/models` | Local encrypted model storage |
| `XENO_DEFAULT_MODEL_ID` | seed-node | `multimolecule/dnabert2` | Hugging Face model to download |
| `XENO_MINER_RPC_URL` | xenom-miner | `ws://xeno-seed:17110` | WebSocket URL of seed node |
| `XENO_MINER_MODEL_ID` | xenom-miner | `multimolecule/dnabert2` | Model id to train |
| `XENO_MINER_THREADS` | xenom-miner | `4` | CPU threads for Candle trainer |
| `XENO_MINER_MOCK_MODE` | xenom-miner | `true` | Use fast mock trainer |
| `XENO_WALLET_PASSWORD` | xenom-miner | `devnet-password` | Wallet encryption |
| `USDT_CONTRACT_ADDRESS` | api-gateway | (Mumbai) | USDT token for payment verification |
| `GOVERNANCE_CONTRACT_ADDRESS` | api-gateway | (Mumbai) | `ModelGovernance` contract |
| `GOVERNANCE_OPERATOR_KEY` | api-gateway | - | Operator private key |

## 5. Devnet deployment workflow

The `scripts/` directory automates the devnet:

1. `scripts/quick-devnet.sh` — one-command setup:
   - Checks ports.
   - Generates `.env` from `.env.example` if missing.
   - Calls `scripts/build-devnet.sh` unless `--no-build` is passed.
   - Runs `docker compose -f docker-compose.devnet.yml up -d`.
   - Waits for `xeno-node` to be healthy.
2. `scripts/build-devnet.sh` — builds Docker images for `xeno-node`, `xeno-seed`, and `xenom-miner`. On macOS it defaults to a Docker-based build to produce Linux binaries.
3. `scripts/check-devnet-health.sh` — checks RPC connectivity and peer counts.
4. `scripts/test-governance-flow.sh` and `scripts/test-payment-flow.sh` — Foundry/cast based integration tests.
5. `scripts/cleanup-devnet.sh` — tears down containers and optional volumes.

`docker-compose.devnet.yml` defines the services on the `xeno-devnet` bridge network. The `xeno-miner` service has a 4 CPU / 4 GB resource limit and depends on `xeno-seed`.

## 6. Training proof and consensus

The consensus layer defines the proof format in `consensus/core/src/pow/training_proof.rs`:

```rust
pub struct TrainingProof {
    pub model_id: ModelId,
    pub base_checkpoint: Hash,
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: Hash,
    pub zk_proof: ZKProof,
    pub batch_indices: Vec<u64>,
}
```

- `DifficultyTarget` requires `loss_before - loss_after >= 0.001` and `loss_after < 1.0`.
- `quality_score()` returns `1.0 + min(0.5, improvement * 10.0)`.
- `ZKProof` is currently a **placeholder**: `verify()` only checks that `proof_data` is non-empty and that `loss_after < loss_before`.

## 7. Gaps and placeholders

The following items are known gaps in the current branch:

- **Real model training in `xenom-miner`:** the `CpuTrainer` is a small synthetic MLP; it does not load or train the real `multimolecule/dnabert2` model.
- **ZK proof:** `consensus/core/src/pow/training_proof.rs:66` and `mining/src/training/miner.rs:204` contain placeholders. A real ZK-SNARK (e.g. EZKL) is not implemented.
- **Consensus reward integration:** `quality_score` exists but reward distribution and difficulty adjustment are not fully wired into consensus.
- **`mining` crate separation:** `mining/src/training/` is not used by `xenom-miner`; the two training paths are duplicated/independent.
- **Training data source:** batches are synthetic; no real dataset is loaded.
- **Model checkpoint distribution:** seed-node stores and encrypts models locally but does not push checkpoints back to `xenom-miner`; `base_checkpoint` is currently often zeros.
- **Governance and payment contracts:** the API gateway points to Mumbai/testnet contracts; the Solidity sources and ABIs are referenced but not part of this review.

## 8. Recommendations

1. Unify training logic so `xenom-miner` and `mining` crate share a single trainer implementation.
2. Replace the ZK placeholder with a concrete ZK system and circuit for the training step.
3. Define and implement difficulty adjustment based on real training times and hardware diversity.
4. Implement a real model loader (Candle/Hugging Face) in `xenom-miner` with a `--trainer=dnabert2` backend.
5. Document the governance/payment contract deployment and ABI generation steps.

## 9. References

- `seed-node/src/main.rs`
- `seed-node/src/rpc/server.rs`
- `seed-node/src/model/manager.rs`
- `xenom-miner/src/main.rs`
- `xenom-miner/src/trainer/mod.rs`
- `xenom-miner/src/rpc/messages.rs`
- `api-gateway/src/main.rs`
- `consensus/core/src/pow/training_proof.rs`
- `docker-compose.devnet.yml`
- `.env.example`
