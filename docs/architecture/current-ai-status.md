# Xenomorph AI Architecture — Current Implementation Status

This document is a factual audit of the current Xenomorph repository against the target architecture where the blockchain is the source of truth, model weights are stored encrypted, only Orchestrator Services decrypt and run inference, and miners train LoRA/adapters without receiving plaintext full weights.

Every conclusion below is backed by code found in the repository. If a feature is not present, it is explicitly marked as missing.

---

## 1. Blockchain / Model Registry

**Status:** 🟡 Partial

**Location:**
- `smart-contracts/ModelRegistry.sol`
- `smart-contracts/ModelGovernance.sol`
- `smart-contracts/InferencePayments.sol`
- `seed-node/src/consensus/dynamic_registry.rs`
- `xenom-miner/src/governance/voter.rs`
- `api-gateway/src/governance/`

**Description:**
- `ModelRegistry.sol` defines a `Model` struct with `name`, `description`, `version`, `modelHash` (`bytes32`), `submitter`, `blockHeight`, `active`/`deprecated` flags, `totalQueries`, and `totalEarnings`. It also maintains `modelVerificationHash` and `verifiedModels` mappings and authorized submitter/verifier lists.
- `ModelGovernance.sol` implements permissionless, stake-weighted governance for model proposals. An `ActiveModel` struct stores `modelId`, `hfRepo`, `hfRevision`, `genesisCheckpoint` (`bytes32`), `vramRequired`, `rewardPerBlock`, `minStakeToTrain`, activation/deprecation blocks, and active flag. Voting lasts 7 days with a 1-day execution delay, 66% approval threshold, and 1M XEN quorum.
- `InferencePayments.sol` handles USDT payments for inference, distributing revenue 70% to seed nodes, 20% to training, 10% to treasury.
- The seed-node and unified `xenom` node read the active model from the on-chain registry via `dynamic_registry.rs`.
- The API gateway can query the `ModelGovernance` contract to list proposals, cast votes, and execute proposals.

**Missing functionality:**
- CID / IPFS / content-addressed identifiers are not stored in the contracts. The `modelHash` field is a `bytes32` hash, not a CID.
- SHA-256 verification is declared as a field but the contract does not perform off-chain hash verification against delivered weights.
- Signature / public-key verification for model registration is not implemented in the contract (the `submitter` address is trusted from `msg.sender` only).
- Real staking integration is a placeholder; `ModelGovernance` uses a local `stakeBalance` mapping that can be set directly by the owner for tests.
- Rewards are conceptually linked to models through `rewardPerBlock` and `totalEarnings`, but reward distribution logic inside the Kaspa consensus is not audited in this scope.

---

## 2. Training Pipeline

**Status:** ✅ Implemented

**Location:**
- `xenom-miner/src/main.rs`
- `xenom-miner/src/trainer/` (`mod.rs`, `cpu_trainer.rs`, `dnabert2_trainer.rs`, `mgm1_trainer.rs`, `mgm1_multi_gpu.rs`, `multi_gpu.rs`, `gradient.rs`, `lora.rs`)
- `xenom/src/training/coordinator.rs`
- `xenom/src/training/gradient_validator.rs`
- `seed-node/src/model/manager.rs`

**Description:**
- The miner supports multiple trainer backends: `mock` (no compute), `cpu` (Candle MLP), `dnabert2` (real DNABERT-2 with CUDA/Metal/CPU fallback), `mgm1` (Mini Genome Model).
- DNABERT-2 training is implemented through `DnaBert2Trainer` and the optimized `MultiGpuTrainer`/`GpuTrainer` path, which handles CUDA-first allocation, FP16 mixed precision, multi-GPU data-parallel training, and gradient averaging.
- Masked language modelling uses span masking and optional reverse-complement augmentation for genome data.
- LoRA (Low-Rank Adaptation) adapters are supported through `lora.rs` and `LoraConfig`, with target modules, rank, alpha, and dropout.
- FedAvg gradient aggregation is implemented in `seed-node/src/model/manager.rs` and in the unified node, with bounded-staleness checkpoint caching (default 8 historical checkpoints).
- Gradient compression with top-k sparsification and blake3 gradient commitments is in `xenom-miner/src/trainer/gradient.rs`.
- Training proof validation (loss improvement, gradient commitment, and full re-execution for MGM-1) is in `xenom/src/training/gradient_validator.rs` and `coordinator.rs`.
- Checkpoint handling includes historical weights keyed by hash and block-height-specific inference.

**Missing functionality:**
- Full-model fine-tuning (not LoRA) on the miner is not implemented; the current path always loads the full base and may update LoRA matrices or the full model through gradient steps, but the intended "miners only train adapters" mode is not strictly enforced.
- Distributed training across multiple machines is not implemented.
- vLLM, TensorRT, llama.cpp, and ONNX training backends are missing.

---

## 3. Model Storage

**Status:** 🟡 Partial

**Location:**
- `seed-node/src/model/storage.rs`
- `seed-node/src/model/downloader.rs`
- `xenom/src/training/coordinator.rs`
- `xenom-miner/src/model_cache.rs`
- `xenom-miner/src/model_client.rs`

**Description:**
- Models are stored encrypted on the local filesystem under `XENO_MODELS_DIR`.
- `ModelStorage` writes `config.enc`, `tokenizer.enc`, and `weights.enc` using AES-256-GCM.
- A `keyhash` file is stored alongside for verification.
- Historical checkpoints are stored as `weights_<hash>.enc`.
- The downloader fetches from Hugging Face (`hfRepo`/`hfRevision`) and falls back to GitHub for genome archives.
- The miner caches the checkpoint locally in `ModelCache` and merges a fresh LoRA adapter into a cached base when the node returns an adapter-only payload.

**Missing functionality:**
- IPFS integration is a placeholder: `cid` in `Announcement` and related RPC messages is set to the `weights_hash` until IPFS/libp2p CIDs are wired.
- S3 storage backend is not found.
- Arweave storage backend is not found.
- Distributed or replicated storage across nodes is not implemented.
- The orchestrator does not pull models from decentralized storage on demand; it downloads from Hugging Face and encrypts to a local directory.

---

## 4. Model Encryption

**Status:** 🟡 Partial

**Location:**
- `crypto/model-crypto/src/lib.rs`
- `crypto/model-crypto/Cargo.toml`
- `seed-node/src/model/storage.rs`
- `xenom-miner/src/model_client.rs`

**Description:**
- AES-256-GCM encryption for model files is implemented in `crypto/model-crypto/src/lib.rs`.
- The format is `nonce || ciphertext` with a 12-byte random nonce.
- The key is derived from `XENO_MODEL_KEY` (64-char hex or SHA-256 of the string), with a hard-coded default `xenom-devnet-model-key` for devnet.
- `model_crypto::key_hash` is stored with files to detect key mismatches.
- The `chacha20poly1305` crate is listed in the workspace but only `aes_gcm` is used.

**Missing functionality:**
- ChaCha20 is not used.
- HKDF is not implemented.
- No KMS integration (AWS KMS, HashiCorp Vault, GCP Cloud KMS, etc.).
- No TPM, HSM, or hardware-backed key storage.
- No key rotation mechanism.
- No per-model or per-user encryption keys; the same `XENO_MODEL_KEY` is shared between node and miner.
- No memory zeroization of the decryption key after use in `model_crypto` (`zeroize` is only used in wallet modules).

---

## 5. Orchestrator Service

**Status:** 🟡 Partial

**Location:**
- `xenom/src/training/coordinator.rs`
- `xenom/src/training/inference_service.rs`
- `xenom/src/training/websocket_server.rs`
- `xenom/src/training/checkpoint_sync.rs`
- `seed-node/src/serving/inference.rs`
- `seed-node/src/serving/inference_engine.rs`
- `seed-node/src/model/manager.rs`

**Description:**
- A unified `xenom` full node combines Kaspa consensus, training coordination, gRPC inference, and a miner WebSocket server.
- `InferenceGrpcService` reuses `seed_node::serving::inference::InferenceService` to serve the `xenom.inference.Inference` gRPC API (`Predict`, `Embed`, `EvaluateMaskedLlm`, `GetModelInfo`).
- `ModelManager` loads encrypted model files, decrypts them on demand, and keeps a model cache in memory.
- The orchestrator validates training proofs before building and mining blocks.
- Historical checkpoint inference is supported: `GetModelInfo`/`Predict`/`EvaluateMaskedLlm` accept an optional `block_height` and load the weights active at that block.
- Gossip-based checkpoint sync (`xenom/src/training/checkpoint_sync.rs`, `crypto/model-crypto/src/gossip.rs`) announces new checkpoints over the Kaspa P2P layer with secp256k1 signatures.

**Missing functionality:**
- No request scheduler or batching for inference.
- No model cache eviction policy (LRU, size-based, time-based).
- No hot-swap mechanism for model version switches beyond reloading from disk.
- No versioning abstraction beyond hash-based checkpoint history.
- No load balancer or multi-inference-instance orchestration.
- No automatic decentralized model download in the orchestrator (it relies on Hugging Face / local files).
- No health monitoring or auto-recovery of crashed models.

---

## 6. AI Gateway

**Status:** 🟡 Partial

**Location:**
- `api-gateway/src/main.rs`
- `api-gateway/src/handlers/openai.rs`
- `api-gateway/src/handlers/predict.rs`
- `api-gateway/src/handlers/models.rs`
- `api-gateway/src/seed_client.rs`

**Description:**
- The API gateway is an Axum HTTP server.
- OpenAI-compatible routes exist:
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
- `chat_completions` sanitizes input to DNA bases and forwards to the seed-node gRPC `Predict` endpoint.
- `embeddings` forwards to the seed-node gRPC `Embed` endpoint.
- `list_models` and `get_model` proxy the gRPC `ListModels` / `GetModelInfo` calls.
- Custom endpoints: `GET/POST /models`, `POST /predict/{model_id}`, payment webhooks, governance proposals/voting.

**Missing functionality:**
- Streaming is explicitly **not** implemented: `chat_completions` returns `StatusCode::NOT_IMPLEMENTED` when `stream: true`.
- No request batching for inference.
- No routing logic (single seed-node backend).
- No authentication or authorization.
- No rate limiting.
- No request queuing or priority handling.
- CORS is enabled for all origins (`allow_origin(Any)` in `api-gateway/src/main.rs`).

---

## 7. Inference Backends

**Status:** 🟡 Partial

**Location:**
- `xenom-miner/src/dnabert2.rs`
- `xenom-miner/src/trainer/dnabert2_trainer.rs`
- `xenom-miner/src/trainer/mgm1_trainer.rs`
- `seed-node/src/serving/inference_engine.rs`
- `mini-genome-model/src/lib.rs`
- `xenom-miner/Cargo.toml`

**Description:**
- The only inference framework is **Candle** (`candle-core`, `candle-nn`, `candle-transformers` version 0.3).
- DNABERT-2 (~110M parameters) and MGM-1 are implemented as Candle models.
- CPU, CUDA, and Metal (Apple Silicon) device targets are supported.
- The `InferenceEngine` in `seed-node/src/serving/inference_engine.rs` dispatches between `MaskedLM`, `CausalLM`, `Chat`, and `Embedding` model kinds and runs MLM / embedding inference.
- Multi-GPU training is supported; multi-GPU inference is not a primary feature.

**Missing functionality:**
- vLLM is not found.
- TensorRT is not found.
- llama.cpp is not found.
- ONNX Runtime is not found.
- No backend abstraction layer that would allow switching between Candle, vLLM, TensorRT, etc.

---

## 8. Miners

**Status:** 🟡 Partial

**Location:**
- `xenom-miner/src/main.rs`
- `xenom-miner/src/model_client.rs`
- `xenom-miner/src/model_cache.rs`
- `xenom-miner/src/trainer/` (all trainers)
- `xenom-miner/src/wallet/manager.rs`
- `xenom-miner/src/rpc/client.rs`
- `xenom-miner/src/block/builder.rs`

**Description:**
- The miner connects to the unified node over WebSocket.
- It fetches model checkpoints through `ModelCheckpointV2` RPC, which may be encrypted.
- `decrypt_v2` in `xenom-miner/src/model_client.rs` decrypts config/tokenizer/weights using the shared `XENO_MODEL_KEY`.
- It supports LoRA adapter training and can merge a freshly downloaded adapter with a cached base.
- It supports multi-GPU data-parallel training and gradient compression.
- It builds and signs `TrainingBlock` submissions with a BIP39-derived secp256k1 wallet.

**Critical gap vs. target architecture:**
- **Miners do receive full plaintext model weights.** The `model_client.rs` docstring explicitly states: "The returned `ModelBundle` is always plaintext; if the node sent encrypted files they are decrypted with the same `XENO_MODEL_KEY` used by the node." The miner decrypts and loads the full base weights into Candle, so it can train the model. This directly contradicts the requirement "NEVER receive full model weights."
- Differential privacy is not implemented.
- Secure enclaves / TEE training are not implemented.

---

## 9. Security

**Status:** 🟡 Partial

**Location:**
- `crypto/model-crypto/src/lib.rs`
- `crypto/model-crypto/src/gossip.rs`
- `xenom-miner/src/wallet/manager.rs`
- `xenom/src/training/gradient_validator.rs`
- `xenom/src/training/coordinator.rs`
- `consensus/pow/src/genome_pow.rs`
- `consensus/core/src/pow/training_proof.rs`

**Description:**
- AES-256-GCM for model file encryption.
- SHA-256 and Blake3 for hashing and gradient commitments.
- secp256k1 ECDSA for block signing and P2P gossip announcement signing.
- Gradient commitment via Blake3, verified in the validator.
- Training proof validation with loss improvement checks before block submission.
- Address prefix validation for network compatibility.
- OpenZeppelin `ReentrancyGuard` in smart contracts.
- `zeroize` is used in wallet modules.

**Missing functionality:**
- No signature verification for model authenticity on-chain (only `msg.sender` and authorized verifier lists).
- No memory-only decryption guarantee: files are decrypted into `Vec<u8>` in process memory, but there is no mlock, no secure enclave, and no explicit zeroization of plaintext buffers.
- No secure deletion / explicit zeroization of decrypted weights after inference/training.
- No confidential computing (SGX, SEV-SNP, Nitro Enclaves, TDX).
- No remote attestation.
- No TPM / HSM integration.
- ZK proof in `TrainingProof` is a placeholder; `coordinator.rs` checks the proof structure but does not verify a real ZK proof.
- No DDoS protection or API rate limiting.

---

## 10. Synchronization

**Status:** 🟡 Partial

**Location:**
- `xenom/src/training/checkpoint_sync.rs`
- `crypto/model-crypto/src/gossip.rs`
- `protocol/flows/src/v5/model_gossip.rs`
- `xenom/src/training/websocket_server.rs`
- `seed-node/src/p2p.rs`

**Description:**
- P2P gossip protocol announces new checkpoints with secp256k1 signatures.
- `ModelGossip` flow propagates `Announcement` structs over the Kaspa P2P layer.
- `CheckpointSync` downloads missing checkpoints from peers via WebSocket (with QUIC transfer described in a PRD but not fully wired).
- Hash-based version detection: `weights_hash` and `base_hash` in `ModelCheckpointInfoV2`.
- Bounded-staleness checkpoint cache (8 by default).
- Historical checkpoint storage for block-height-specific inference.

**Missing functionality:**
- CID-based retrieval (IPFS/libp2p) is not implemented; `cid` is currently a placeholder equal to `weights_hash`.
- No conflict resolution for divergent checkpoints.
- No Byzantine-fault-tolerant quorum on checkpoint validity.
- No bandwidth-efficient delta transfer (QUIC transfer is designed in a PRD but not wired into the main sync path).
- No full model replication protocol.

---

## 11. Documentation

**Status:** 🟡 Partial

**Location:**
- `README.md`
- `AGENTS.md`
- `STATUS.md`
- `CONCEPTS.md`
- `docs/ARCHITECTURE.md`
- `docs/adr/` (Architecture Decision Records)
- `docs/prd/` (Product Requirements Documents)
- `docs/issues/` (Issue write-ups)

**Description:**
- README covers build, devnet, and miner usage.
- `AGENTS.md` is a detailed operational guide for agents, including integration tests, build commands, CLI options, and wallet prefixes.
- ADRs document decisions (e.g. DNABERT-2 MLM vocabulary, historical checkpoint evaluation).
- PRDs cover LoRA, QUIC transfer, genome batch diversity, and other features.

**Missing functionality:**
- No API reference (OpenAI/gRPC schema).
- No contributor guide.
- No security audit report.
- No performance benchmarking documentation.
- No troubleshooting guide beyond `AGENTS.md`.

---

## Implementation Gap Summary Table

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| Blockchain / Model Registry | 🟡 Partial | On-chain `Model`/`ActiveModel` exist, but CID/IPFS, signature/public-key verification, and real staking are missing. | High |
| Training Pipeline | ✅ Implemented | DNABERT-2, MGM-1, multi-GPU, LoRA, FedAvg, gradient compression. | - |
| Model Storage | 🟡 Partial | Local encrypted filesystem only. IPFS/S3/Arweave not implemented. | High |
| Model Encryption | 🟡 Partial | AES-256-GCM with shared env key. No HKDF, KMS, TPM, HSM, or key rotation. | High |
| Orchestrator Service | 🟡 Partial | Unified node with gRPC inference and checkpoint history, but no scheduler, eviction, hot-swap, or load balancing. | High |
| AI Gateway | 🟡 Partial | OpenAI-compatible endpoints exist; streaming, batching, routing, auth, and rate limiting are missing. | High |
| Inference Backends | 🟡 Partial | Only Candle. vLLM, TensorRT, llama.cpp, ONNX missing. | High |
| Miners | 🟡 Partial | Training works, but **miners receive and decrypt full plaintext weights**, violating the target architecture. | Critical |
| Security | 🟡 Partial | Encryption and hashing exist, but no memory-only guarantee, no TEE, no remote attestation, no real ZK proofs. | Critical |
| Synchronization | 🟡 Partial | P2P gossip and WebSocket sync exist; CID retrieval, conflict resolution, and delta transfer missing. | Medium |
| Documentation | 🟡 Partial | README/ADRs/PRDs exist; API reference and security audit missing. | Low |

---

## Current Implementation

The current Xenomorph stack is a functional devnet for blockchain-incentivized AI training:

- A Kaspa-based full node (`xenom`) coordinates miners, validates training proofs, and mines blocks.
- A Candle-based inference engine (`seed-node`/`xenom` gRPC) serves predictions and embeddings.
- Miners train DNABERT-2 and MGM-1 with LoRA, multi-GPU, and FedAvg aggregation.
- Encrypted model storage and shared-key decryption protect files at rest.
- An API gateway offers OpenAI-compatible REST endpoints.
- Solidity contracts govern model registration, voting, and payments.

## Existing Components

- Blockchain consensus with Useful PoW.
- On-chain model registry and governance (proposals/votes).
- Encrypted local model storage.
- DNABERT-2 and MGM-1 Candle models.
- CUDA / Metal / CPU training and inference.
- Multi-GPU data-parallel training.
- LoRA adapter support.
- FedAvg gradient aggregation.
- Gradient compression (top-k) and blake3 commitments.
- Training proof validation.
- Historical checkpoint evaluation by block height.
- OpenAI-compatible API gateway skeleton.
- P2P gossip for checkpoint announcements.

## Missing Components

- Decentralized model storage (IPFS, S3, Arweave) with CID verification.
- Proper key management (HKDF, KMS, TPM, HSM, key rotation).
- Multiple inference backends (vLLM, TensorRT, llama.cpp, ONNX).
- Orchestrator features: scheduler, batching, eviction, hot-swap, load balancing.
- AI Gateway features: streaming, batching, routing, auth, rate limiting.
- Miner-side full-weight privacy (currently decrypts full weights; needs split/federated/distilled training or TEEs).
- Confidential computing and remote attestation.
- Real zero-knowledge training proofs.
- On-chain signature/public-key verification for model authenticity.
- CID-based P2P retrieval and bandwidth-efficient delta sync.

## Technical Debt

1. `cid` is a placeholder equal to `weights_hash`; no actual IPFS/libp2p integration.
2. The same `XENO_MODEL_KEY` is shared between node and miner, weakening the "miners never see plaintext weights" guarantee.
3. ZK proof field exists in `TrainingProof` but is only structurally validated.
4. `MultiGpuTrainer` recently moved from truncating large batches to chunking them; this works but loss metrics are computed on a reference chunk rather than the full batch.
5. `ModelGovernance` staking is a test placeholder.
6. The API gateway has `CORS(Any)` and no auth in production.

## Risks

| Risk | Severity | Description |
|------|----------|-------------|
| Miners receive plaintext full weights | Critical | Violates the core security model; any compromised miner exposes the model. |
| Shared static encryption key | Critical | `XENO_MODEL_KEY` shared between node and miner; key compromise exposes all models. |
| Single inference backend | High | Candle is the only option; limits model support and throughput. |
| No auth / rate limiting | High | API gateway is open to abuse in any public deployment. |
| No real ZK proofs | Medium | Training proofs rely on re-execution and loss checks, not cryptographic ZK. |
| Local-only storage | Medium | Models are not content-addressed or replicated; availability depends on Hugging Face and local disk. |
| No confidential computing | Medium | No protection against node-operator exfiltration of plaintext weights. |

## Recommended Implementation Order

1. **Fix miner weight exposure** — redesign the training protocol so miners train LoRA/adapters on encrypted activations or inside a TEE, without receiving the full base weights in plaintext.
2. **Key management** — implement HKDF, per-model keys, and integrate a KMS/Vault; stop sharing `XENO_MODEL_KEY`.
3. **Decentralized storage** — wire `cid` to real IPFS/libp2p or S3 and add CID verification.
4. **AI Gateway hardening** — add streaming, batching, authentication, and rate limiting.
5. **Additional inference backends** — add vLLM/TensorRT/ONNX abstraction for larger and faster models.
6. **Real ZK proofs** — replace placeholder ZK with a verifiable proof of training.
7. **Confidential computing** — add TEE support and remote attestation for the orchestrator.
8. **Governance hardening** — implement real staking and on-chain signature verification for model registration.

---

*Generated from a code-level audit of `/Users/ruimendes/CLionProjects/Xenomorph`.*
