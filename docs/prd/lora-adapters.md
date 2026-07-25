# PRD: LoRA Adapters for Xenomorph UsefulPoW

## Problem Statement

Running a Xenomorph miner against a remote `xenom` node currently requires downloading the full DNABERT-2 checkpoint (~440 MB in F32) every time the active `base_checkpoint` changes. On a slow remote link this can take more than seven minutes per round, dominating round time and leaving GPUs idle. The current sync model sends the entire model weights even when only a small update has been produced by FedAvg.

## Solution

Introduce optional LoRA (Low-Rank Adaptation) fine-tuning. The full base model is downloaded once and kept frozen. All further useful work trains small, low-rank adapter matrices attached to selected `Linear` and `Embedding` layers. Only the adapter weights are synced between the node and the miner after the initial base download. This reduces the per-round payload from hundreds of megabytes to a few megabytes while preserving the existing FedAvg aggregation and block-submission pipeline.

## User Stories

1. As a remote GPU miner, I want to download the full model only once, so that slow network links do not dominate my training throughput.
2. As a remote GPU miner, I want subsequent rounds to sync only small adapter deltas, so that I can keep GPUs busy on training instead of waiting for weight downloads.
3. As a node operator, I want the active model to remain the source of truth, so that the base checkpoint integrity is preserved and the consensus layer still validates `base_checkpoint` and `gradients_commitment`.
4. As a devnet operator, I want to toggle LoRA with CLI flags and environment variables, so that I can compare full fine-tuning and adapter fine-tuning without rebuilding.
5. As a miner operator, I want the miner to fall back to full fine-tuning when LoRA is disabled, so that the existing behavior is unchanged by default.
6. As a model governor, I want the inference service to apply the latest adapter to the frozen base model, so that `/predict` returns results from the current active model.
7. As a node operator, I want FedAvg to aggregate adapter gradients instead of full gradients, so that aggregation remains lightweight and the checkpoint cache stays small.
8. As a miner operator, I want `ModelCache` to cache the base model and adapters separately, so that only the adapter is refetched when the active adapter changes.
9. As a developer, I want unit tests for LoRA layer forward passes and gradient flow, so that the implementation can be verified without a full GPU run.
10. As a node operator, I want the active model hash to reflect both the frozen base and the active adapter, so that `get_model_checkpoint_info` still produces a stable, verifiable `base_checkpoint`.
11. As a miner operator, I want `submit_block` and `submit_gradients` to continue using the same Borsh/WebSocket messages, so that no wire protocol changes are required.
12. As a devnet operator, I want LoRA rank, alpha, dropout, and target modules to be configurable per model, so that I can tune the quality/performance trade-off.

## Implementation Decisions

### 1. LoRA module layer

A new `lora` module will be added inside `xenom-miner`. It exposes:

- `LoraConfig` holding `rank`, `alpha`, `dropout`, and `target_modules`.
- `LoraLinear` and `LoraEmbedding` wrappers that store the frozen base parameters as plain `Tensor`s and the trainable low-rank matrices (`lora_a`, `lora_b`) as `Var`s.
- The forward pass computes `base_output + ((x @ lora_a^T) @ lora_b^T) * (alpha / rank)`.

Base weights are loaded directly from the safetensors/PyTorch buffer as non-trainable `Tensor`s; only LoRA matrices are registered in the `VarMap`/`VarBuilder`. This lets `ManualAdamW` continue iterating over the `VarMap` while implicitly freezing the base, because base tensors will not appear in the gradient graph.

### 2. DNABERT-2 architecture integration

`DnaBert2ForMaskedLM` and its sub-modules (`DnaBert2SelfAttention`, `DnaBert2GatedMlp`, `DnaBert2LMPredictionHead`, `DnaBert2Embeddings`) will accept an optional `LoraConfig`. When LoRA is enabled, the standard `candle_nn::linear`/`embedding` helpers are replaced with LoRA-aware constructors that build the frozen base layer plus adapter.

`DnaBert2ForMaskedLM::load` and `load_for_training` will accept the same `LoraConfig` so that inference and training use identical adapter architecture.

### 3. Trainer and gradient flow

`DnaBert2Trainer` gains an optional `lora_config` argument. When active:

- `forward`/`backward` produces gradients only for LoRA parameters.
- `apply_gradients` and `apply_sgd_gradients` update only LoRA parameters.
- `save_adapter_to_bytes` serializes only the trainable LoRA adapter tensors (used for `GradientUpdate` and `submit_gradients`).
- `save_weights_to_bytes` can serialize the merged base+adapter checkpoint (used for Phase 1 serving and for inference). When LoRA is disabled it behaves exactly as today.
- `load_weights_from_bytes` loads base weights as frozen `Tensor`s and adapter weights as trainable `Var`s when LoRA is enabled. In Phase 2 it can load only the adapter into an existing LoRA `VarMap` while keeping the frozen base unchanged.

`MultiGpuTrainer` will pass the `LoraConfig` to each replica. Gradient aggregation in `average_grad_maps` and `add_grad_maps` remains unchanged because the gradient map will simply contain fewer entries.

### 4. Model manager and checkpoint storage

`ModelManager` will store the base model and adapter checkpoints separately:

- `base/` holds the frozen `config.json`, `tokenizer.json`, and `model.safetensors` (or `pytorch_model.bin`).
- `adapters/` holds per-round adapter safetensors keyed by `base_hash` and `adapter_hash`.

`submit_gradients` will:

1. Decrypt the adapter gradient payload.
2. Apply the averaged adapter update to the active adapter replica.
3. Save the new adapter weights.
4. Track three hashes:
   - `base_hash` — immutable hash of the frozen base model.
   - `adapter_hash` — hash of the current active adapter.
   - `combined_hash = hash(base_hash, adapter_hash)` — the active checkpoint identifier exposed to `TrainingProof`, `GradientUpdate`, and `get_model_checkpoint_info`.

For serving, `ModelManager` will merge the frozen base and the active adapter into a single safetensors buffer (Phase 1) or return only the adapter (Phase 2). `InferenceEngine` uses the merged checkpoint for `predict`.

`get_model_checkpoint` is implemented in two phases:

- **Phase 1**: always sends the full base model plus the active adapter. This keeps the wire protocol and `ModelCache` changes minimal and lets LoRA training start immediately.
- **Phase 2**: sends the full base only when the miner does not have the cached base hash, and sends only the adapter otherwise. A new optional `cached_base_hash` field in `GetModelCheckpoint` (or a separate `GetModelAdapter` RPC) enables this optimization while remaining backward compatible.

The wire protocol stays backward compatible: `ModelCheckpoint` still contains a single `weights` buffer and a single `base_checkpoint` hash. In Phase 1 that buffer is the merged base+adapter and `base_checkpoint` is the `combined_hash`. In Phase 2, `ModelCheckpointInfo` can also expose `base_hash` and `adapter_hash` to let the miner decide whether it needs the base, and `GetModelCheckpoint` gains an optional `cached_base_hash` field so the node can return only the adapter.

### 5. Miner cache and sync

`ModelCache` will store:

- `base/model.safetensors` + `base/config.json` + `base/tokenizer.json` tagged with the `base_hash`.
- `adapter/<combined_hash>/adapter.safetensors` tagged with the `combined_hash`.

`fetch_model_checkpoint` behavior follows the two phases:

- **Phase 1**: the miner downloads the full base model plus the active adapter and caches both under the `combined_hash`.
- **Phase 2**: the miner first requests `ModelCheckpointInfo` (now exposing both `base_hash` and `combined_hash`). If the cached `base_hash` matches, it requests only the adapter via a new `GetModelAdapter` RPC; otherwise it downloads the full base + adapter bundle.

### 6. Inference engine

`InferenceEngine` will load the frozen base model once and apply the current adapter before `predict`. If an adapter is already loaded, only the new adapter weights are reloaded when the active checkpoint changes. This avoids reloading the full 440 MB model on every inference request.

### 7. CLI and environment configuration

New miner flags (under `GpuArgs` or a new `LoraArgs`):

- `--lora` (bool, default false)
- `--lora-rank` (default 8)
- `--lora-alpha` (default 16)
- `--lora-dropout` (default 0.0)
- `--lora-target-modules` (comma list, default `query,key,value,transform_dense,up_proj,down_proj`)

Equivalent environment variables: `XENO_LORA`, `XENO_LORA_RANK`, `XENO_LORA_ALPHA`, `XENO_LORA_DROPOUT`, `XENO_LORA_TARGET_MODULES`.

The node will also accept `--lora-*` flags so that the default adapter shape is known when serving `get_model_checkpoint_info`.

### 8. Active checkpoint hash

The active checkpoint identifier exposed on the wire is the `combined_hash = hash(base_hash, adapter_hash)`. `TrainingProof` and `GradientUpdate` continue to use a single `base_checkpoint` field; when LoRA is enabled that field carries the `combined_hash`. Internally the node and miner also track `base_hash` and `adapter_hash` separately so the frozen base never needs to be re-downloaded and the adapter payload stays small.

### 9. Backwards compatibility

When LoRA is disabled, all paths produce exactly the current behavior: full weights are loaded, trained, saved, and synced. Enabling LoRA is opt-in per miner and per node.

## Testing Decisions

- **Unit tests** for `LoraLinear`/`LoraEmbedding`: verify forward output shape, verify that gradients only flow to adapter matrices, verify that base weight values remain unchanged after an optimizer step.
- **Unit tests** for `DnaBert2Trainer` with LoRA: train one synthetic MLM batch and confirm `save_weights_to_bytes` produces a buffer much smaller than the base model.
- **Integration test** in `tests/integration` (or a new `test_lora_sync.rs`): start a `xenom` node, a miner with `--lora`, assert that `get_model_checkpoint` returns valid base+adapter weights and that training produces a `GradientUpdate` payload whose weights buffer is much smaller than the full model. A second integration test (Phase 2) asserts that only the adapter is transferred when the cached base matches.
- **Prior art**: existing `dnabert2.rs` tests, `multi_gpu.rs` tests, and `test_miner_node.rs` provide the harness for model loading, training, and node-miner interaction.
- A good test only asserts externally observable behavior (payload size, loss decrease, hash stability) and does not depend on the internal tensor layout of the LoRA matrices.

## Out of Scope

- INT8/quantized base weights (separate performance optimization).
- Merging multiple LoRA adapters into a single model.
- Pre-training from scratch with randomly initialized base weights.
- Changing the consensus `TrainingProof` format; existing fields remain sufficient.
- Removing full fine-tuning as an option.
- Distributed training across multiple miners for the same batch (different from FedAvg aggregation).

## Further Notes

- Rank 8 adapters for DNABERT-2 are expected to be in the low single-digit megabytes, a ~100x reduction over the full F32 checkpoint.
- The forward-pass compute overhead of LoRA is small (two extra small matmuls per targeted layer) compared to the base layer.
- A higher `alpha` (e.g., `alpha = 2 * rank`) gives stronger adaptation; lower ranks reduce sync size.
- Target modules default to attention `query`/`key`/`value` plus MLP `up_proj`/`down_proj` and LM-head `transform_dense`, covering the most parameter-efficient positions for MLM fine-tuning.
- The node should still keep the full base model encrypted and intact; this PRD respects the requirement that the node model is the source of truth.
