# Xenomorph AI Architecture — Current Implementation Status

This is a factual audit of the current `xenom-AI-v2` branch against the target architecture where the blockchain is the source of truth, model weights are stored encrypted, only Orchestrator Services decrypt and run inference, and miners train LoRA/adapters without receiving plaintext full weights.

Every conclusion below is backed by code found in the repository. If a feature is not present, it is explicitly marked as missing.

---

## 1. Blockchain / Model Registry

**Status:** 🟡 Partial

**Location:**
- `smart-contracts/ModelRegistry.sol`
- `smart-contracts/ModelGovernance.sol`
- `smart-contracts/InferencePayments.sol`
- `seed-node/src/consensus/dynamic_registry.rs`
- `xenom/src/training/coordinator.rs`
- `xenom/src/training_block_service.rs`
- `api-gateway/src/governance/mod.rs`

**Description:**
- `ModelRegistry.sol` defines a `Model` struct with `name`, `description`, `version`, `modelHash` (`bytes32`), `submitter`, `blockHeight`, `active`/`deprecated` flags, `totalQueries`, and `totalEarnings` (`ModelRegistry.sol:12-23`). It also has `modelVerificationHash` and `verifiedModels` mappings, plus authorized submitter/verifier lists.
- `ModelGovernance.sol` implements permissionless, stake-weighted governance for model proposals. An `ActiveModel` struct stores `modelId`, `hfRepo`, `hfRevision`, `genesisCheckpoint` (`bytes32`), `vramRequired`, `rewardPerBlock`, `minStakeToTrain`, activation/deprecation blocks, and an active flag (`ModelGovernance.sol:35-46`). Voting lasts 7 days with a 1-day execution delay, 66% approval threshold, and 1M XEN quorum.
- `InferencePayments.sol` handles USDT payments for inference, distributing revenue 70% to seed nodes, 20% to training, 10% to treasury (`InferencePayments.sol:19-22`).
- The standalone `seed-node` reads the active model from the on-chain registry via `DynamicModelRegistry` (`seed-node/src/consensus/dynamic_registry.rs:30-131`).
- The unified `xenom` full node uses the locally configured `active_model_id` (`xenom/src/training/coordinator.rs:65`) and does not read `ModelGovernance` for the active model.
- The API gateway can query the `ModelGovernance` contract to list proposals, cast votes, and execute proposals (`api-gateway/src/governance/mod.rs:52-90`).

**Missing functionality:**
- CID / IPFS / content-addressed identifiers are not stored in the contracts. The `modelHash` field is a `bytes32` hash, not a CID.
- SHA-256 verification is declared as a field but the contract does not perform off-chain hash verification against delivered weights.
- Signature / public-key verification for model registration is not implemented in the contract; the `submitter` address is trusted from `msg.sender` only.
- The `ModelGovernance` `stakeBalance` mapping is a test placeholder (`ModelGovernance.sol:63` comment), not a real staking contract.
- On-chain `rewardPerBlock` is stored but not linked to the Kaspa consensus reward calculation; consensus uses a fitness multiplier instead.
- `ModelRegistry.sol` and `InferencePayments.sol` are not bound into Rust code (only `ModelGovernance` has Rust ABI integration).

---

## 2. Training Pipeline

**Status:** 🟡 Partial

**Location:**
- `xenom-miner/src/main.rs`
- `xenom-miner/src/trainer/` (`mod.rs`, `mock_trainer.rs`, `cpu_trainer.rs`, `dnabert2_trainer.rs`, `mgm1_trainer.rs`, `mgm1_multi_gpu.rs`, `multi_gpu.rs`, `gradient.rs`, `lora.rs`, `lora_only_trainer.rs`, `lora_lm_head.rs`)
- `xenom/src/training/coordinator.rs`
- `xenom/src/training/gradient_validator.rs`
- `seed-node/src/model/manager.rs`
- `seed-node/src/consensus/fedavg.rs`

**Description:**
- The miner supports multiple trainer backends: `mock`, `cpu`, `dnabert2`, `mgm1`, `gpu`, `cuda`, `rocm`, `metal`, and `lora` (`xenom-miner/src/main.rs:63-65`).
- DNABERT-2 training is implemented through `DnaBert2Trainer` and the `MultiGpuTrainer`/`GpuTrainer` path, which handles CUDA/Metal/CPU, FP16 mixed precision, multi-GPU data-parallel training, and gradient averaging (`xenom-miner/src/trainer/dnabert2_trainer.rs`, `multi_gpu.rs`, `gpu_trainer.rs`).
- Masked language modelling uses span masking and optional reverse-complement augmentation for genome data.
- LoRA (Low-Rank Adaptation) adapters are supported through `lora.rs` and `LoraConfig`, with target modules, rank, alpha, and dropout (`xenom-miner/src/lora.rs:8-20`).
- A secure LoRA-only track exists: `--trainer lora` uses `LoraOnlyTrainer` to fetch an encrypted `TrainingArtifact` (only LM head + embedding weights), request `AttestedForward` hidden states, train a LoRA adapter, and submit an encrypted `SubmitLoRAUpdate` (`xenom-miner/src/trainer/lora_only_trainer.rs:1-395`).
- FedAvg gradient aggregation is implemented in `seed-node/src/model/manager.rs` and in the unified node, with bounded-staleness checkpoint caching (default 8 historical checkpoints) (`seed-node/src/model/manager.rs:232-256`, `1190-1270`).
- Gradient compression with top-k sparsification and blake3 gradient commitments is in `xenom-miner/src/trainer/gradient.rs`.
- Training proof validation (loss improvement, gradient commitment, and full re-execution for MGM-1) is in `xenom/src/training/gradient_validator.rs` and `coordinator.rs`.
- Checkpoint handling includes historical weights keyed by hash and block-height-specific inference.

**Missing functionality:**
- The default `dnabert2`, `mgm1`, `gpu`, etc. trainers still download and load the full plaintext base weights, so the "miners only train adapters" guarantee is not enforced by default.
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
- `ModelStorage` writes `config.enc`, `tokenizer.enc`, and `weights.enc` using AES-256-GCM (`seed-node/src/model/storage.rs:91-99`).
- A `keyhash` file is stored alongside for verification (`seed-node/src/model/storage.rs:84-86`).
- Historical checkpoints are stored as `weights_<hash>.enc` (`seed-node/src/model/storage.rs:104-110`).
- The downloader fetches from Hugging Face (`hfRepo`/`hfRevision`) and falls back to GitHub for genome archives (`seed-node/src/model/downloader.rs`).
- The miner caches the combined checkpoint locally in `ModelCache` (`xenom-miner/src/model_cache.rs:15-76`).
- V2 checkpoint sync can send only the encrypted LoRA adapter when the miner already has the matching base hash (`seed-node/src/model/manager.rs:646-690`).

**Missing functionality:**
- IPFS integration is a placeholder: `cid` in `Announcement` and related RPC messages is set to the `weights_hash` until IPFS/libp2p CIDs are wired (`xenom/src/training/websocket_server.rs:140-141`).
- S3 and Arweave storage backends are not found.
- Distributed or replicated storage across nodes is not implemented.
- The miner caches the full decrypted `ModelBundle` as plaintext on disk (`xenom-miner/src/model_cache.rs:64-70`).

---

## 4. Model Encryption

**Status:** 🟡 Partial

**Location:**
- `crypto/model-crypto/src/lib.rs`
- `crypto/model-crypto/src/key_hierarchy.rs`
- `crypto/model-crypto/src/session.rs`
- `crypto/model-crypto/src/artifact_sign.rs`
- `crypto/model-crypto/src/secret_buffer.rs`
- `seed-node/src/model/storage.rs`
- `xenom-miner/src/model_client.rs`
- `seed-node/src/model/lora_artifact.rs`

**Description:**
- AES-256-GCM encryption for model files is implemented in `crypto/model-crypto/src/lib.rs:78-109`.
- The format is `nonce || ciphertext` with a 12-byte random nonce.
- The shared key is derived from `XENO_MODEL_KEY` (64-char hex or SHA-256 of the string), with a hard-coded default `xenom-devnet-model-key` for devnet (`crypto/model-crypto/src/lib.rs:41-58`).
- A per-model key hierarchy is implemented with HKDF-SHA256: `EK_m`, `KAuth_m`, and per-session `SK_{m,s}` (`crypto/model-crypto/src/key_hierarchy.rs:69-113`).
- ECDH over secp256k1 plus HKDF derives ephemeral session keys for encrypting LoRA artifacts and attested forward hidden states (`crypto/model-crypto/src/session.rs:25-54`).
- Artifacts are signed with `KAuth_m` and verified with `ArtifactVerifier` (`crypto/model-crypto/src/artifact_sign.rs:31-100`).
- `ModelSecret` and `ForwardSession` implement `Zeroize` / `ZeroizeOnDrop` (`crypto/model-crypto/src/key_hierarchy.rs:24`, `seed-node/src/model/manager.rs:267-274`).
- `SecretBuffer` provides `mlock(2)` and `ZeroizeOnDrop` for memory protection (`crypto/model-crypto/src/secret_buffer.rs:1-166`), but it is not used for decrypted model weights.

**Missing functionality:**
- `SecretBuffer` is not integrated into the main model loading path; weights are decrypted into plain `Vec<u8>` and loaded into Candle tensors (`xenom-miner/src/model_client.rs:18-21`).
- No KMS, HashiCorp Vault, TPM, HSM, or hardware-backed key storage integration.
- No key rotation mechanism.
- The per-model master key is stored as a plaintext hex file `model_master.key` unless `XENO_MODEL_MASTER_KEY` is set (`seed-node/src/model/lora_artifact.rs:119-150`).
- No explicit zeroization of decrypted plaintext model weights after inference/training.

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
- A dedicated "Orchestrator" binary does not exist; the unified `xenom` full node and the legacy `seed-node` fulfill the orchestrator role.
- `InferenceGrpcService` reuses `seed_node::serving::inference::InferenceService` to serve the `xenom.inference.Inference` gRPC API (`Predict`, `Embed`, `EvaluateMaskedLlm`, `GetModelInfo`, `ListModels`) (`seed-node/src/serving/inference.rs:68-265`).
- `ModelManager` loads encrypted model files, decrypts them on demand, and keeps a bounded LRU checkpoint cache in memory (`seed-node/src/model/manager.rs:1246-1270`).
- The orchestrator validates training proofs before building and mining blocks (`xenom/src/training/coordinator.rs:450-554`).
- Historical checkpoint inference is supported: `GetModelInfo`/`Predict`/`EvaluateMaskedLlm` accept an optional `block_height` and load the weights active at that block (`seed-node/src/serving/inference.rs:32-65`).
- Gossip-based checkpoint sync (`xenom/src/training/checkpoint_sync.rs`, `protocol/flows/src/v5/model_gossip.rs`) announces new checkpoints over the Kaspa P2P layer with secp256k1 signatures.

**Missing functionality:**
- No request scheduler or batching for inference.
- The inference engine cache is an unbounded `Mutex<HashMap>` with no eviction policy (`seed-node/src/serving/inference_engine.rs:147-150`).
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
- `api-gateway/src/state.rs`
- `api-gateway/src/payments/verifier.rs`

**Description:**
- The API gateway is an Axum HTTP server.
- OpenAI-compatible routes exist (`api-gateway/src/main.rs:59-63`):
  - `GET /v1/models`
  - `GET /v1/models/{model_id}`
  - `POST /v1/chat/completions`
  - `POST /v1/embeddings`
- `chat_completions` sanitizes input to DNA bases and forwards to the seed-node gRPC `Predict` endpoint (`api-gateway/src/handlers/openai.rs:131-191`).
- `embeddings` forwards to the seed-node gRPC `Embed` endpoint (`api-gateway/src/handlers/openai.rs:193-243`).
- `list_models` and `get_model` proxy the gRPC `ListModels` / `GetModelInfo` calls.
- Custom endpoints: `GET/POST /models`, `POST /predict/{model_id}`, payment webhooks, governance proposals/voting.
- Payment verification uses the `InferencePayments` contract (`api-gateway/src/payments/verifier.rs:56-76`).

**Missing functionality:**
- Streaming is explicitly **not** implemented: `chat_completions` returns `StatusCode::NOT_IMPLEMENTED` when `stream: true` (`api-gateway/src/handlers/openai.rs:135-137`).
- No request batching for inference.
- No routing logic (single seed-node backend).
- No authentication or authorization.
- No rate limiting.
- No request queuing or priority handling.
- CORS is enabled for all origins (`api-gateway/src/main.rs:72`).

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
- `seed-node/Cargo.toml`

**Description:**
- The only inference framework is **Candle** (`candle-core`, `candle-nn`, `candle-transformers`).
- DNABERT-2 (~110M parameters) and MGM-1 are implemented as Candle models.
- CPU, CUDA, and Metal (Apple Silicon) device targets are supported.
- The `InferenceEngine` dispatches between `MaskedLM`, `CausalLM`, `Chat`, and `Embedding` model kinds and runs MLM / embedding inference (`seed-node/src/serving/inference_engine.rs:40-53`).
- PyTorch `.bin`/`.pth` legacy zip files can be loaded through Candle's pickle loader (`xenom-miner/src/dnabert2.rs:510-528`).

**Missing functionality:**
- vLLM is not found.
- TensorRT is not found.
- llama.cpp is not found.
- ONNX Runtime is not found.
- TensorFlow is not found.
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
- It can fetch model checkpoints through `GetModelCheckpointV2` RPC, which may return full merged weights or an encrypted adapter if the miner's cached base hash matches (`xenom-miner/src/model_client.rs:84-191`).
- `decrypt_v2` in `xenom-miner/src/model_client.rs:13-33` decrypts config/tokenizer/weights using the shared `XENO_MODEL_KEY`.
- It supports LoRA adapter training and can merge a freshly downloaded adapter with a cached base (`xenom-miner/src/model_client.rs:65-75`).
- It supports multi-GPU data-parallel training and gradient compression.
- It builds and signs `TrainingBlock` submissions with a BIP39-derived secp256k1 wallet.
- The `--trainer lora` path uses `GetTrainingArtifact` and `AttestedForward` to train an LM-head LoRA adapter without holding the full base transformer (`xenom-miner/src/trainer/lora_only_trainer.rs:122-246`).

**Critical gap vs. target architecture:**
- **Default trainers receive and cache full plaintext model weights.** The `model_client.rs` docstring states: "The returned `ModelBundle` is always plaintext; if the node sent encrypted files they are decrypted with the same `XENO_MODEL_KEY` used by the node" (`xenom-miner/src/model_client.rs:82-83`). The miner decrypts and loads the full base weights into Candle, and the `ModelCache` writes them to disk as plaintext (`xenom-miner/src/model_cache.rs:64-70`).
- The LoRA-only track is an opt-in spike (`--trainer lora`), not the default.
- Differential privacy is not implemented.
- Secure enclaves / TEE training are not implemented.

---

## 9. Security

**Status:** 🟡 Partial

**Location:**
- `crypto/model-crypto/src/lib.rs`
- `crypto/model-crypto/src/key_hierarchy.rs`
- `crypto/model-crypto/src/artifact_sign.rs`
- `crypto/model-crypto/src/gossip.rs`
- `crypto/model-crypto/src/secret_buffer.rs`
- `xenom-miner/src/wallet/manager.rs`
- `xenom/src/training/gradient_validator.rs`
- `xenom/src/training/coordinator.rs`
- `seed-node/src/model/lora_merge.rs`
- `seed-node/src/serving/attested_forward.rs`

**Description:**
- AES-256-GCM for model file encryption.
- SHA-256 and Blake3 for hashing and gradient commitments.
- secp256k1 ECDSA for block signing, P2P gossip announcement signing, artifact signing, and LoRA delta signing.
- Gradient commitment via Blake3, verified in the validator (`xenom/src/training/gradient_validator.rs`).
- Training proof validation with loss improvement checks before block submission (`xenom/src/training/coordinator.rs:450-554`).
- Address prefix validation for network compatibility.
- `zeroize` is used for `ModelSecret`, `ForwardSession`, and LoRA session nonces.
- `AttestedForward` returns signed and encrypted hidden states; the signature covers the hidden-states hash, base checkpoint, and loss (`seed-node/src/serving/attested_forward.rs:126-134`).

**Missing functionality:**
- No signature / public-key verification for model authenticity on-chain (the `ModelRegistry` only stores `submitter` and `modelHash`).
- No memory-only decryption guarantee: files are decrypted into `Vec<u8>` in process memory, and there is no `mlock` or explicit zeroization of plaintext buffers for full model weights.
- No secure deletion / explicit zeroization of decrypted full weights after inference/training.
- No confidential computing (SGX, SEV-SNP, Nitro Enclaves, TDX).
- No remote attestation. `AttestedForward` is cryptographic attestation of a forward pass, not a TEE attestation report.
- No TPM / HSM integration.
- The ZK proof field in `TrainingProof` is a placeholder; `coordinator.rs` checks the proof structure but does not verify a real ZK proof (`xenom/src/training/coordinator.rs:526-545`).
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
- `seed-node/src/consensus/dynamic_registry.rs`

**Description:**
- P2P gossip protocol announces new checkpoints with secp256k1 signatures (`protocol/flows/src/v5/model_gossip.rs:75-149`).
- `ModelGossip` flow propagates `Announcement` structs over the Kaspa P2P layer.
- `CheckpointSync` downloads missing checkpoints from peers via WebSocket (`xenom/src/training/checkpoint_sync.rs:46-114`).
- Hash-based version detection: `weights_hash` and `base_hash` in `ModelCheckpointInfoV2` (`xenom-rpc/src/messages.rs:241-247`).
- Bounded-staleness checkpoint cache (8 by default).
- Historical checkpoint storage for block-height-specific inference.
- The standalone `seed-node` can read active models from the on-chain `ModelGovernance` contract (`seed-node/src/consensus/dynamic_registry.rs:30-131`).

**Missing functionality:**
- CID-based retrieval (IPFS/libp2p) is not implemented; `cid` is currently a placeholder equal to `weights_hash` (`xenom/src/training/websocket_server.rs:140-141`).
- No conflict resolution for divergent checkpoints.
- No Byzantine-fault-tolerant quorum on checkpoint validity.
- No bandwidth-efficient delta transfer (QUIC transfer is designed in a PRD but not wired into the main sync path).
- The unified `xenom` full node does not read the active model from the on-chain registry.

---

## 11. Documentation

**Status:** 🟡 Partial

**Location:**
- `README.md`
- `AGENTS.md`
- `STATUS.md`
- `CONCEPTS.md`
- `docs/ARCHITECTURE.md`
- `docs/adr/`
- `docs/prd/`
- `docs/issues/`

**Description:**
- README covers build, devnet, and miner usage.
- `AGENTS.md` is a detailed operational guide for agents, including integration tests, build commands, CLI options, and wallet prefixes.
- ADRs document decisions (e.g. DNABERT-2 MLM vocabulary, historical checkpoint evaluation).
- PRDs cover LoRA, secure model distribution, orchestrator, AI gateway, confidential computing, encrypted storage, migration, and other features.

**Missing functionality:**
- No generated API reference (OpenAI/gRPC schema).
- No contributor guide.
- No security audit report.
- No performance benchmarking documentation.
- No troubleshooting guide beyond `AGENTS.md`.

---

## Implementation Gap Summary Table

| Component | Status | Notes | Priority |
|-----------|--------|-------|----------|
| Blockchain / Model Registry | 🟡 Partial | `Model`/`ActiveModel` exist, but no CID/IPFS, on-chain signature/public-key verification, or real staking. `rewardPerBlock` not linked to consensus rewards. | High |
| Training Pipeline | 🟡 Partial | DNABERT-2, MGM-1, multi-GPU, LoRA, FedAvg, gradient compression all exist. Default trainers still receive full plaintext weights. | High |
| Model Storage | 🟡 Partial | Local encrypted filesystem only. IPFS/S3/Arweave not implemented; `cid` is a placeholder. Miner caches plaintext. | High |
| Model Encryption | 🟡 Partial | AES-256-GCM, HKDF, ECDH, and artifact signing exist. No KMS/TPM/HSM; `SecretBuffer` not used for weights; master key may be stored plaintext. | High |
| Orchestrator Service | 🟡 Partial | Unified node with gRPC inference and checkpoint history, but no scheduler, batching, inference cache eviction, hot-swap, or load balancing. | High |
| AI Gateway | 🟡 Partial | OpenAI-compatible endpoints exist; streaming, batching, routing, auth, and rate limiting are missing. | High |
| Inference Backends | 🟡 Partial | Only Candle. vLLM, TensorRT, llama.cpp, ONNX missing. | High |
| Miners | 🟡 Partial | Training works, but default trainers receive and decrypt full plaintext weights; `--trainer lora` is the only secure path. | Critical |
| Security | 🟡 Partial | Encryption and hashing exist, but no memory-only guarantee, no TEE, no remote attestation, no real ZK proofs, no on-chain model auth. | Critical |
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
- A new LoRA-only secure distribution track (`--trainer lora`) allows a miner to train an LM-head adapter using encrypted artifacts and attested hidden states without holding the full base model.

The largest remaining gap is that the default miner path still downloads and caches full plaintext model weights, and the orchestrator/inference path does not use true confidential computing or hardware-backed key storage.

---

## Existing Components

- Blockchain consensus with Useful PoW and genome PoW.
- On-chain model registry and governance (proposals/votes).
- Encrypted local model storage with AES-256-GCM.
- Per-model HKDF key hierarchy and ECDH session-key exchange.
- Artifact signing and verification with secp256k1.
- DNABERT-2 and MGM-1 Candle models.
- CUDA / Metal / CPU training and inference.
- Multi-GPU data-parallel training.
- LoRA adapter support and a secure LoRA-only training track.
- FedAvg gradient aggregation and bounded-staleness checkpoint cache.
- Gradient compression (top-k) and blake3 commitments.
- Training proof validation and historical checkpoint evaluation.
- OpenAI-compatible API gateway skeleton.
- P2P gossip for checkpoint announcements and WebSocket checkpoint sync.
- `mlock`-capable `SecretBuffer` and `ZeroizeOnDrop` secret wrappers.

---

## Missing Components

- Decentralized model storage (IPFS, S3, Arweave) with CID verification.
- Proper key management (KMS, TPM, HSM, key rotation, no plaintext master-key file).
- Multiple inference backends (vLLM, TensorRT, llama.cpp, ONNX).
- Orchestrator features: scheduler, batching, eviction, hot-swap, load balancing.
- AI Gateway features: streaming, batching, routing, auth, rate limiting.
- Miner-side full-weight privacy as the default (currently only `--trainer lora` avoids full weights).
- Confidential computing and remote attestation.
- Real zero-knowledge training proofs.
- On-chain signature/public-key verification for model authenticity.
- CID-based P2P retrieval and bandwidth-efficient delta sync.
- Integration between the unified `xenom` full node and the on-chain `ModelGovernance` active model.
- Model-specific reward linkage and real staking contract.

---

## Technical Debt

1. `cid` is a placeholder equal to `weights_hash`; no actual IPFS/libp2p integration.
2. The same `XENO_MODEL_KEY` is shared between node and miner for default full-weight checkpoints, weakening the "miners never see plaintext weights" guarantee.
3. `ModelManager` may write the per-model master key to a plaintext `model_master.key` file.
4. `SecretBuffer` exists but is not used in the main model decryption path; decrypted weights live as plaintext `Vec<u8>` and Candle tensors.
5. ZK proof field exists in `TrainingProof` but is only structurally validated.
6. The inference engine model cache is an unbounded `HashMap` with no eviction.
7. The unified `xenom` node does not read the active model from `ModelGovernance`; it relies on CLI arguments and local `ModelManager`.
8. `ModelRegistry` and `InferencePayments` are not bound to Rust services (only `ModelGovernance` is).
9. Miner `ModelCache` persists the full decrypted `ModelBundle` as plaintext on disk.

---

## Risks

- **Plaintext weights on miners:** Default trainers receive, decrypt, and cache full model weights, exposing the model to any compromised miner.
- **Weak key management:** Shared `XENO_MODEL_KEY` and optional plaintext `model_master.key` files create a single point of compromise.
- **No TEE:** There is no hardware-enforced boundary around the orchestrator's decrypted weights or the miner's training process.
- **No real ZK training proofs:** A miner could submit a fabricated or replayed training proof.
- **No on-chain model authenticity:** The blockchain cannot verify that a served model artifact was signed by the registered model authority.
- **No gateway security:** The API gateway has no authentication, rate limiting, or DDoS protection.
- **Single inference backend:** Candle-only support limits model diversity and production scalability.

---

## Recommended Implementation Order

1. **Make LoRA-only the default miner path:** Remove or deprecate the full-weight V2 download for DNA trainers; enforce per-miner encrypted artifacts and attested hidden states. Integrate `SecretBuffer` and zeroize plaintext model buffers.
2. **Integrate the full node with on-chain governance:** Have `xenom` read `activeModels` from `ModelGovernance.sol` and validate `genesisCheckpoint`/`modelHash` against the served model.
3. **Harden key management:** Replace the shared `XENO_MODEL_KEY` and plaintext `model_master.key` with a KMS/TPM/HSM-backed per-model master key and a real key-rotation policy.
4. **Add decentralized model storage:** Implement IPFS/S3/Arweave backends and store real CIDs with on-chain verification.
5. **Add an inference backend abstraction and additional backends:** Implement vLLM/TensorRT/ONNX support behind a common backend interface.
6. **Build real orchestrator services:** Add request scheduling, batching, inference cache eviction, hot-swap, and load balancing.
7. **Complete the AI Gateway:** Implement streaming, batching, routing, API-key authentication, and rate limiting.
8. **Implement confidential computing and remote attestation:** Integrate TEEs (SGX/SEV/TDX/Nitro) for the orchestrator and attestation verification for miners.
9. **Implement real ZK training proofs:** Replace the placeholder `zk_proof` field with a verifiable proof system.
10. **Wire model-specific rewards and real staking:** Link `rewardPerBlock` and `minStakeToTrain` into consensus rewards and a real staking contract.
