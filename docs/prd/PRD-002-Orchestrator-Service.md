# PRD-002: Model Orchestrator Service

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current orchestration logic is embedded inside the unified `xenom` full node (`xenom/src/training/coordinator.rs`) and the standalone `seed-node` (`seed-node/src/serving/inference.rs`, `seed-node/src/model/manager.rs`). There is no dedicated Model Orchestrator with lifecycle management, scheduling, cache eviction, hot-swap, or autoscaling. This PRD designs a standalone, production-grade Model Orchestrator service.

---

## 1. Problem

### 1.1 Current implementation

- `xenom/src/training/coordinator.rs` is a monolithic unified node that mixes consensus, training, inference, and model management.
- `seed-node/src/model/manager.rs` loads a single model into memory and keeps it indefinitely.
- `seed-node/src/serving/inference.rs` handles gRPC `Predict`/`Embed`/`EvaluateMaskedLlm` synchronously, one request at a time per model instance.
- `xenom/src/training/inference_service.rs` wraps the seed-node inference into an async gRPC service but does not add scheduling, queueing, or load balancing.

### 1.2 Attack vectors

- Orchestrator runs the full model in the same process as consensus and miner coordination; a memory corruption in training or inference can affect consensus.
- No model isolation; a malicious checkpoint can compromise the whole node.
- No runtime limits on inference requests; the node can be DoS'd by large inputs.

### 1.3 Architectural limitations

- Cannot run multiple models concurrently.
- Cannot load-balance inference across workers.
- No graceful model updates; switching models requires restart or manual reload.
- No memory-aware eviction; the node can OOM when switching between large models.

---

## 2. Target Architecture

### 2.1 Responsibilities

The Model Orchestrator is a dedicated service (or process pool) with the following responsibilities:

1. **Download** encrypted model artifacts from decentralized storage.
2. **Verify** SHA-256/Blake3 hash, signature, and on-chain `artifactHash`.
3. **Decrypt** only in RAM; never write plaintext to disk.
4. **Load** the model into a backend worker (Candle, vLLM, TensorRT-LLM, etc.).
5. **Cache** encrypted and decrypted models with memory-aware eviction.
6. **Schedule** inference requests across worker pool.
7. **Serve** gRPC and REST inference APIs.
8. **Validate** training proofs and LoRA deltas.
9. **Merge** LoRA deltas into the base model.
10. **Publish** new checkpoints with signed P2P announcements.
11. **Health-check** workers and restart or evict failed models.
12. **Metrics** for latency, throughput, queue depth, memory, and errors.

### 2.2 Trust boundaries

- Orchestrator holds the master key and is the only component that may decrypt base weights.
- Miners, AI Gateways, and P2P peers receive only signed, encrypted artifacts or inference responses.
- Blockchain is the source of truth for active model, approved checkpoints, and orchestrator public keys.

### 2.3 Process model

```
Orchestrator Process
  ├─ Control Plane (tokio async runtime)
  │   ├─ gRPC/REST API
  │   ├─ Model registry client
  │   ├─ Storage downloader
  │   ├─ Worker pool manager
  │   ├─ Scheduler
  │   ├─ Cache manager
  │   └─ Metrics exporter
  └─ Worker Pool (isolated processes/threads)
      ├─ Candle Worker
      ├─ vLLM Worker
      ├─ TensorRT-LLM Worker
      └─ ONNX Worker
```

---

## 3. Alternatives

### 3.1 Option A: Keep orchestration inside the unified node

**Pros:**
- Minimal change to current architecture.
- No new process to deploy.

**Cons:**
- Violates least privilege; consensus holds inference keys.
- Cannot scale inference independently from consensus.
- Single model in memory at a time.

### 3.2 Option B: Separate orchestrator service, single process

Run the orchestrator as a separate binary but in a single process with multiple backend threads.

**Pros:**
- Clear separation from consensus.
- Simpler deployment.

**Cons:**
- A crash in one backend crashes all models.
- No process-level isolation.

### 3.3 Option C: Separate orchestrator with worker processes (recommended)

Run the orchestrator as a control plane and spawn isolated worker processes per model/backend.

**Pros:**
- Process-level fault isolation.
- Independent resource limits per worker.
- Easy to support backends in different languages (vLLM/TGI in Python, TensorRT in C++).
- Easier to add TEEs or confidential computing per worker.

**Cons:**
- Higher IPC complexity.
- More moving parts.

### 3.4 Recommended

Option C, with Unix domain sockets (or TCP loopback) and shared memory for small tensors, gRPC for control, and a wire protocol (Cap'n Proto or flatbuffers) for high-throughput worker communication.

---

## 4. API Changes

### 4.1 gRPC

`proto/inference.proto` should be extended:

```protobuf
service Inference {
  rpc GetModelInfo(ModelInfoRequest) returns (ModelInfoResponse);
  rpc Predict(PredictRequest) returns (PredictResponse);
  rpc Embed(EmbedRequest) returns (EmbedResponse);
  rpc EvaluateMaskedLlm(EvaluateMaskedLlmRequest) returns (EvaluateMaskedLlmResponse);

  // New orchestrator control RPCs
  rpc LoadModel(LoadModelRequest) returns (LoadModelResponse);
  rpc UnloadModel(UnloadModelRequest) returns (UnloadModelResponse);
  rpc ListLoadedModels(ListLoadedModelsRequest) returns (ListLoadedModelsResponse);
  rpc GetMetrics(GetMetricsRequest) returns (GetMetricsResponse);
}

message LoadModelRequest {
  string model_id = 1;
  bytes base_checkpoint = 2;
  string backend = 3; // "candle", "vllm", "tensorrt", "onnx"
  bool use_lora = 4;
  uint64 block_height = 5;
}

message UnloadModelRequest {
  string model_id = 1;
  bytes base_checkpoint = 2;
}

message MetricsResponse {
  uint64 active_models = 1;
  uint64 queue_depth = 2;
  double avg_latency_ms = 3;
  uint64 requests_per_min = 4;
  double vram_used_gb = 5;
  double vram_total_gb = 6;
}
```

### 4.2 REST

A management REST API for operators:

- `POST /admin/models/{model_id}/load`
- `POST /admin/models/{model_id}/unload`
- `GET /admin/models`
- `GET /admin/metrics`
- `POST /admin/cache/evict`

### 4.3 Internal worker protocol

Each worker exposes a local gRPC or Unix socket with:

- `Init(model_artifact_bytes, backend_config)`
- `Predict(batch)`
- `Embed(batch)`
- `Evaluate(batch)`
- `ForwardAttest(batch)` — returns signed hidden states for miner LoRA training.
- `MergeLoRA(lora_delta)`
- `Shutdown()`

---

## 5. Blockchain Changes

- `ModelRegistry.sol` stores orchestrator `publicKey` and `artifactCID` per model version.
- `ModelGovernance.sol` adds `minOrchestratorStake` and an `approveOrchestrator` function.
- New event: `event OrchestratorApproved(bytes32 indexed modelId, address indexed orchestrator);`
- Block extra data may include the current active `artifactHash` for light clients.

---

## 6. Storage Changes

See `PRD-005-Encrypted-Storage.md`.

Highlights:
- Orchestrator fetches `artifactCID` from chain, downloads from IPFS/Arweave/S3.
- Orchestrator keeps encrypted blob cache on local disk and plaintext model in worker memory only.
- Cache eviction uses LRU and a memory budget (e.g., 80% of available VRAM).

---

## 7. Security Changes

- Orchestrator master key is loaded from KMS/Vault on startup; never in environment variables.
- Worker processes have no filesystem write access except to a temp directory for logs.
- Plaintext models are loaded into `mlock`-ed or `memfd_secret` memory.
- `seccomp`/AppArmor/SELinux profiles restrict worker syscalls.
- Keys and plaintext are zeroized on worker shutdown.

---

## 8. Networking

- Orchestrator registers a QUIC/WebSocket listen address in P2P gossip.
- Miners connect to orchestrator for training artifacts.
- AI Gateways connect via gRPC/REST for inference.
- Orchestrator-to-worker uses local UDS/TCP with mTLS.

---

## 9. Scheduler and Load Balancing

- Request queue per model.
- Worker pool with `min_workers` and `max_workers` per model.
- Dynamic scaling based on queue depth and VRAM usage.
- Batch similar requests when possible (for non-streaming).
- Streaming requests bypass batching and get a dedicated worker.
- Health checks every 10s; unhealthy workers are killed and replaced.

---

## 10. Metrics and Autoscaling

Metrics exported via Prometheus at `/metrics`:

- `xenom_inference_requests_total`
- `xenom_inference_latency_seconds`
- `xenom_inference_queue_depth`
- `xenom_model_vram_bytes`
- `xenom_worker_restarts_total`
- `xenom_model_load_failures_total`

Autoscaling rules (operator-configurable):
- If queue depth > threshold for 60s, start an additional worker.
- If VRAM > 90% for 120s, evict least recently used model.

---

## 11. Migration Plan

See `PRD-010-Migration-Plan.md`.

High-level:
1. Extract inference and model loading from `xenom` and `seed-node` into a new `xenom-orchestrator` crate.
2. Keep unified node as a thin coordinator until `xenom-orchestrator` is production-ready.
3. Gradually move gRPC inference to the orchestrator.
4. Deprecate in-process inference after governance vote.

---

## 12. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Orchestrator centralization | High | Allow multiple orchestrators per model; gateway routes to approved set. |
| Worker isolation failures | High | Use process sandboxes, seccomp, and no network for workers except control socket. |
| OOM during model switch | Medium | Memory-aware eviction and pre-flight checks before loading. |
| IPC overhead | Medium | Use shared memory for large tensors; benchmark before scaling. |

---

## 13. Deliverables

This PRD is the design input for:
- `PRD-008-Inference-Backends.md`
- `PRD-007-Miner-Redesign.md`
- `PRD-003-AI-Gateway.md`
