# LoRA Track — Phase 1: Secure LoRA-Only Training

## Goal

Implement the first phase of `PRD-007-Miner-Redesign`: miners train a LoRA
adapter on the LM head without ever loading the full base transformer.

## What was added

### 1. Shared RPC messages crate (`xenom-rpc`)

All Borsh RPC messages were extracted into a new `xenom-rpc` crate
(`xenom-rpc/src/messages.rs`) and are re-exported by `xenom-miner::rpc::messages`
and `seed-node::rpc::messages`.  This removes the previous duplication between
the miner and the orchestrator.

New message types:

- `GetTrainingArtifact` / `TrainingArtifact` — LoRA/adapter artifact distribution.
- `AttestedForwardRequest` / `AttestedForwardResponse` — orchestrator-signed hidden states.
- `SubmitLoRAUpdate` — encrypted LoRA delta submission.
- `ArtifactType` enum (`Base`, `LoRA`, `Adapter`, `GradientTask`).
- `RpcRequest` and `RpcResponse` variants: `GetTrainingArtifact`, `AttestedForward`, `SubmitLoRAUpdate`, `TrainingArtifact`, `AttestedForward`, `LoRAUpdateAck`.

### 2. `XenomRpcClient` methods (`xenom-miner/src/rpc/client.rs`)

- `get_training_artifact(...)`
- `attested_forward(request)`
- `submit_lora_update(update)`

### 3. Orchestrator `AttestedForward` handler (`seed-node/src/serving/attested_forward.rs`)

The seed-node can receive an `AttestedForwardRequest`, load the base DNABERT-2
checkpoint, run `encode` up to the LM head, compute the base LM loss, and return
an `AttestedForwardResponse` with hidden states, hidden-state hash, loss,
ephemeral public key, session nonce, and a model auth signature.  The hidden
states are encrypted to the miner's ECDH public key.

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

- Calls `get_training_artifact` and caches the artifact across rounds.
- Builds `LoraLmHead` from the returned artifact.
- Tokenizes `GenomeTrainingBatchMsg` sequences into an MLM batch.
- Calls `attested_forward` with a batch.
- Trains the LoRA adapter for `local_steps` using `ManualAdamW`.
- Submits the adapter via `submit_lora_update`.

### 6. `GetTrainingArtifact` in seed-node (`seed-node/src/model/lora_artifact.rs`)

Extracts only the frozen LM head + tied embedding weights from a DNABERT-2
safetensors checkpoint and packages them in a `TrainingArtifact`.  The artifact
is encrypted and signed using `model-crypto`:

- ECDH with an ephemeral orchestrator key and the miner's public key.
- HKDF-SHA256 session key derivation.
- AES-256-GCM encryption.
- secp256k1 artifact signature with the model auth key.
- Orchestrator master key loaded from `XENO_MODEL_MASTER_KEY` or persisted to disk.

### 7. `--trainer lora` in `xenom-miner`

The miner CLI now accepts `--trainer lora`.  When selected, the miner enters a
dedicated async loop (`run_lora_loop` in `xenom-miner/src/main.rs`) that:

- Fetches genome batches from the orchestrator.
- Trains a LoRA LM head via `LoraOnlyTrainer`.
- Builds a `TrainingBlock` and submits it (or logs it in `--dry-run`).

### 8. Server handlers

`seed-node/src/rpc/server.rs` now handles the new `RpcRequest` variants:

- `AttestedForward` — calls `serving::attested_forward`.
- `GetTrainingArtifact` — packages the LM head via `model::lora_artifact`.
- `SubmitLoRAUpdate` — calls `model::lora_merge::apply_lora_update`.

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

## Implementation status

1. `GetTrainingArtifact` encrypts and signs the artifact:
   - Uses ECDH between an ephemeral orchestrator key and the miner's public key.
   - Derives a session key via HKDF with a per-artifact nonce.
   - Encrypts the artifact with AES-256-GCM.
   - Signs the plaintext artifact hash with the model auth key.
   - The miner verifies the signature and decrypts with its secp256k1 secret.
   - The orchestrator master key is loaded from `XENO_MODEL_MASTER_KEY` (hex 64)
     or persisted to `<models_dir>/<model_id>/model_master.key`.

2. `AttestedForward` response hidden states are encrypted and signed:
   - The request carries `miner_public_key`.
   - The orchestrator runs the base model, serializes hidden states, and
     computes `hidden_states_hash`.
   - Signs `hidden_states_hash || base_checkpoint || loss` with the model auth key.
   - Encrypts the hidden states with an ECDH session key and a random nonce.
   - The miner decrypts with its secp256k1 secret and verifies the signature.

3. `SubmitLoRAUpdate` is validated, decrypted, signed, and merged by the
   orchestrator:
   - The miner signs `lora_delta_hash || base_checkpoint` with its secp256k1
     secret.
   - The LoRA delta is encrypted by reusing the `AttestedForward` ECDH session
     key; the orchestrator caches the ephemeral secret keyed by its public key
     and decrypts the delta before merging.
   - The orchestrator verifies the signature with `auth_public_key` (miner's
     public key).
   - It verifies `lora_delta_hash` against the decrypted delta.
   - It loads the base checkpoint, merges `W_new = W_base + (alpha/rank) * (lora_b @ lora_a)`,
     stores the new active checkpoint, and returns the new `weights_hash`.
   - Cached forward sessions have a 5-minute TTL and are evicted on lookup.

4. `--trainer lora` uses real `loss_before` and `loss_after`:
   - `loss_before` is the base LM head loss reported by the orchestrator in the
     `AttestedForward` response.
   - `loss_after` is the LoRA LM head loss after the local training steps.
   - The miner's secp256k1 identity is derived from the BIP39 wallet seed
     (`wallet.secret_bytes()`), so the same wallet always produces the same ECDH
     public key and LoRA delta signatures.

5. Session nonces for `TrainingArtifact` and `AttestedForward` are generated
   randomly per message using `rand::thread_rng().fill_bytes`.

## Limitations and next steps

1. `--trainer lora` still requires a genome merkle root and does not yet
   integrate with the full `TrainingBlock` / `TrainingProof` validation pipeline
   in a way that reports the new checkpoint produced by `SubmitLoRAUpdate`.

2. The `LoraOnlyTrainer` keeps the LM head artifact and the latest forward
   session in memory between rounds.  Secure zeroization of LoRA weights and
   session state on shutdown is not yet implemented.

3. `xenom-rpc` only holds `rpc::messages`.  The WebSocket client/codec still
   lives in `xenom-miner` and could be moved later if `seed-node` or the unified
   `xenom` node needs the same client.

## Files changed

- `xenom-miner/src/rpc/messages.rs`
- `xenom-miner/src/rpc/client.rs`
- `xenom-miner/src/trainer/mod.rs`
- `xenom-miner/src/trainer/lora_lm_head.rs` (new)
- `xenom-miner/src/trainer/lora_only_trainer.rs` (new)
- `xenom-miner/src/main.rs`
- `xenom-miner/src/lora.rs` (`varmap()` accessor, `get_base_tensor` is now `pub(crate)`)
- `xenom-miner/tests/integration_tests.rs`
- `xenom-miner/tests/lora_track_spike.rs` (new)
- `xenom-rpc/Cargo.toml` (new)
- `xenom-rpc/src/lib.rs` (new)
- `xenom-rpc/src/messages.rs` (new)
- `crypto/model-crypto/src/session.rs` (new)
- `Cargo.toml` (workspace members)
- `seed-node/src/rpc/messages.rs`
- `seed-node/src/rpc/server.rs`
- `seed-node/src/serving/mod.rs`
- `seed-node/src/serving/attested_forward.rs` (new)
- `seed-node/src/model/mod.rs`
- `seed-node/src/model/lora_artifact.rs` (new)
- `seed-node/src/model/lora_merge.rs` (new)
- `docs/architecture/lora-track-phase1.md` (new)
