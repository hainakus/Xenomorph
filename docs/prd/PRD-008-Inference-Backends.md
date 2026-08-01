# PRD-008: Inference Backends

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current system supports only the Candle framework. To scale to larger models and higher throughput, the orchestrator must support vLLM, TensorRT-LLM, llama.cpp, and ONNX Runtime. This PRD designs a backend abstraction layer.

---

## 1. Problem

### 1.1 Current implementation

- `seed-node/src/serving/inference_engine.rs` uses `candle_core`.
- `xenom-miner/src/trainer/dnabert2_trainer.rs` and `mgm1_trainer.rs` use Candle.
- `xenom-miner/Cargo.toml` lists `candle-core`, `candle-nn`, `candle-transformers`.
- No abstraction over backends.

### 1.2 Limitations

- Candle is limited in large-model support and production serving.
- Cannot use state-of-the-art throughput engines like vLLM or TensorRT.
- No path to support non-Candle models.

---

## 2. Target Architecture

Introduce an `InferenceBackend` trait and a worker pool where each worker can be a different engine.

```rust
#[async_trait]
trait InferenceBackend: Send + Sync {
    async fn init(&mut self, config: &ModelConfig, weights_path: &Path, device: &Device) -> Result<()>;
    async fn predict(&self, request: PredictRequest) -> Result<PredictResponse>;
    async fn embed(&self, request: EmbedRequest) -> Result<EmbedResponse>;
    async fn evaluate_mlm(&self, request: EvaluateMlmRequest) -> Result<EvaluateMlmResponse>;
    async fn forward_attest(&self, batch: MlmBatch) -> Result<AttestedForwardResponse>;
    async fn health(&self) -> Result<BackendHealth>;
}

struct ModelConfig {
    model_id: String,
    backend: BackendType,
    safetensors_path: PathBuf,
    config_json_path: PathBuf,
    tokenizer_path: PathBuf,
    dtype: DType,
    max_batch_size: usize,
    max_seq_len: usize,
}

enum BackendType {
    Candle,
    Vllm,
    TensorRt,
    LlamaCpp,
    Onnx,
}
```

---

## 3. Backend Options

### 3.1 Candle

**Pros:**
- Already implemented.
- Rust-native.
- Good for DNABERT-2 and small models.

**Cons:**
- Limited serving optimizations.
- No PagedAttention.

### 3.2 vLLM

**Pros:**
- PagedAttention, continuous batching, high throughput.
- Widely used for LLM serving.

**Cons:**
- Python; requires running Python worker.
- More memory at startup.

### 3.3 TensorRT-LLM

**Pros:**
- Highest throughput on NVIDIA.
- Optimized kernels.

**Cons:**
- NVIDIA only.
- Complex build/compilation.

### 3.4 llama.cpp

**Pros:**
- Wide model support.
- Quantization.
- CPU inference.

**Cons:**
- Not ideal for batched inference.
- GGUF conversion required.

### 3.5 ONNX Runtime

**Pros:**
- Cross-platform.
- Good for encoder models like DNABERT-2.

**Cons:**
- Less optimized for autoregressive decoding.

### 3.6 Recommended

- **Candle:** default for DNABERT-2, MGM-1, and development.
- **ONNX Runtime:** fast path for DNABERT-2 on CPU/GPU.
- **vLLM:** for future causal/text-generation models.
- **TensorRT-LLM:** for high-throughput NVIDIA deployments.
- **llama.cpp:** for quantized edge inference.

---

## 4. Worker Model

Each backend runs as a separate worker process:

```
Orchestrator Control Plane
  ├─ Candle Worker (Rust, UDS)
  ├─ ONNX Worker (Rust or C++, UDS)
  ├─ vLLM Worker (Python, gRPC over UDS)
  ├─ TensorRT Worker (Python/C++, gRPC over UDS)
  └─ llama.cpp Worker (C++, gRPC over UDS)
```

Workers:
- Are started by the orchestrator on demand.
- Load the model into memory.
- Serve inference requests.
- Report health and metrics.
- Are killed when idle or evicted.

---

## 5. API Changes

### 5.1 gRPC

Extend `proto/inference.proto`:
```protobuf
message LoadModelRequest {
  string model_id = 1;
  bytes base_checkpoint = 2;
  string backend = 3; // "candle", "onnx", "vllm", "tensorrt", "llamacpp"
  uint64 block_height = 4;
  bool use_lora = 5;
}
```

### 5.2 Internal worker protocol

Use gRPC for Python workers and flatbuffers/Cap'n Proto for Rust workers.

---

## 6. Container / Runtime

- vLLM and TensorRT workers run in Docker/Podman containers for dependency isolation.
- Candle and ONNX workers can run as native processes.
- Each worker has resource limits (CPU, memory, GPU).

---

## 7. Migration Plan

1. Define `InferenceBackend` trait in Rust.
2. Refactor `inference_engine.rs` into `CandleBackend`.
3. Add `OnnxBackend` as second implementation.
4. Add vLLM/TGI worker behind feature flag.
5. Add TensorRT-LLM and llama.cpp workers.
6. Make backend selectable per model and per request.

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Backend output mismatch | High | Normalize outputs; run cross-backend tests. |
| Worker startup latency | Medium | Keep hot workers for active models. |
| Dependency hell (Python/CUDA) | Medium | Use containers and version pinning. |
| Resource contention | Medium | Set per-worker GPU memory limits. |

---

## 9. Deliverables

This PRD is the design input for:
- `PRD-002-Orchestrator-Service.md`
- `PRD-003-AI-Gateway.md`
