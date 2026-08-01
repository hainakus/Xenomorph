# LoRA Track — Phase 1: Secure LoRA-Only Training

## Goal

Implement the first phase of `PRD-007-Miner-Redesign`: miners train a LoRA
adapter on the LM head without ever loading the full base transformer.

## What was added

### 1. RPC messages (`xenom-miner` and `seed-node`)

New `rpc::messages` types mirror the PRD-007 workflow:

- `GetTrainingArtifact` / `TrainingArtifact` — encrypted LoRA/adapter artifact distribution.
- `AttestedForwardRequest` / `AttestedForwardResponse` — orchestrator-signed hidden states.
- `SubmitLoRAUpdate` — encrypted LoRA delta submission.
- `ArtifactType` enum (`Base`, `LoRA`, `Adapter`, `GradientTask`).
- `RpcRequest` and `RpcResponse` variants: `GetTrainingArtifact`, `AttestedForward`, `SubmitLoRAUpdate`, `TrainingArtifact`, `AttestedForward`, `LoRAUpdateAck`.

The `seed-node` and `xenom-miner` message files are still duplicated; this is
legacy and will be consolidated in a shared crate later.

### 2. `XenomRpcClient` methods (`xenom-miner/src/rpc/client.rs`)

- `get_training_artifact(...)`
- `attested_forward(request)`
- `submit_lora_update(update)`

### 3. Orchestrator `AttestedForward` handler (`seed-node/src/serving/attested_forward.rs`)

The seed-node can receive an `AttestedForwardRequest`, load the base DNABERT-2
checkpoint, run `encode` up to the LM head, compute the base LM loss, and return
an `AttestedForwardResponse` with hidden states, hidden-state hash, loss, and a
placeholder signature.

This is an initial skeleton: it does not yet encrypt hidden states to the miner's
session key or sign them with the orchestrator auth key.

### 4. `LoraLmHead` (`xenom-miner/src/trainer/lora_lm_head.rs`)

A minimal trainable LM head for the miner:

- Loads only the frozen LM head base weights (`lm_head.transform.dense`,
  `lm_head.transform.layer_norm`, `lm_head.bias`, tied `word_embeddings`).
- Uses the existing `ModelBuilder` to add a LoRA adapter on
  `lm_head.transform.dense`.
- Provides `forward(hidden_states)`, `compute_loss(...)`, `train_step(...)`,
  `save_adapter()`, and `load_adapter()`.
- Uses `ManualAdamW` so gradients flow only to `lora_a` / `lora_b`.

### 5. `LoraOnlyTrainer` (`xenom-miner/src/trainer/lora_only_trainer.rs`)

Drives the full LoRA-only loop:

- Calls `get_training_artifact`.
- Builds `LoraLmHead` from the returned artifact.
- Calls `attested_forward` with a batch.
- Trains the LoRA adapter for `local_steps`.
- Submits the adapter via `submit_lora_update`.

### 6. `GetTrainingArtifact` in seed-node (`seed-node/src/model/lora_artifact.rs`)

Extracts only the frozen LM head + tied embedding weights from a DNABERT-2
safetensors checkpoint and packages them in a `TrainingArtifact`.  For this
spike the artifact is not encrypted and the signature is a placeholder.

### 7. Server stubs

`seed-node/src/rpc/server.rs` now handles the new `RpcRequest` variants:

- `AttestedForward` — calls `serving::attested_forward`.
- `GetTrainingArtifact` — returns an error until distribution is wired.
- `SubmitLoRAUpdate` — returns a placeholder `LoRAUpdateAck`.

## Tests

### `xenom-miner`

```bash
cargo test -p xenom-miner lora_lm_head
cargo test -p xenom-miner rpc
cargo test -p xenom-miner --test lora_track_spike
cargo test -p xenom-miner --test integration_tests
```

- `lora_lm_head` unit test: passes.
- `rpc` roundtrip tests: pass.
- `lora_track_spike` end-to-end integration test: passes.
- `integration_tests`: all 3 existing tests still pass.

Note that `test_mgm1_balanced_local_training` is pre-existing and unrelated to
these changes.

### `seed-node`

```bash
cargo test -p seed-node
```

All 46 tests pass.

## Limitations and next steps

1. `GetTrainingArtifact` extracts the LM head but does **not** encrypt the
   artifact or sign it.  It needs to:
   - Derive a session key from the miner's public key.
   - Encrypt the LM head base weights and LoRA seed.
   - Sign the artifact with the model auth key.

2. `AttestedForward` response hidden states are currently returned in plaintext.
   They must be encrypted with the session key and signed by the orchestrator.

3. `SubmitLoRAUpdate` must validate, decrypt, and merge the LoRA delta on the
   orchestrator.

4. `LoraOnlyTrainer` is a spike.  It must be wired into `xenom-miner/src/main.rs`
   behind `--trainer lora` or similar, and must fetch real training batches.

5. A shared `rpc::messages` crate should be extracted from `xenom-miner` and
   `seed-node` to eliminate the duplicated message definitions.

## Files changed

- `xenom-miner/src/rpc/messages.rs`
- `xenom-miner/src/rpc/client.rs`
- `xenom-miner/src/trainer/mod.rs`
- `xenom-miner/src/trainer/lora_lm_head.rs` (new)
- `xenom-miner/src/trainer/lora_only_trainer.rs` (new)
- `xenom-miner/src/lora.rs` (`varmap()` accessor, `get_base_tensor` is now `pub(crate)`)
- `xenom-miner/tests/integration_tests.rs`
- `xenom-miner/tests/lora_track_spike.rs` (new)
- `seed-node/src/rpc/messages.rs`
- `seed-node/src/rpc/server.rs`
- `seed-node/src/serving/mod.rs`
- `seed-node/src/serving/attested_forward.rs` (new)
- `seed-node/src/model/mod.rs`
- `seed-node/src/model/lora_artifact.rs` (new)
- `docs/architecture/lora-track-phase1.md` (new)
