# PRD-007: Miner Redesign

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current `xenom-miner` decrypts and loads full plaintext model weights. This violates the target security architecture. This PRD redesigns the miner to train LoRA/adapters without full base weights.

---

## 1. Problem

### 1.1 Current implementation

- `xenom-miner/src/model_client.rs::decrypt_v2` decrypts `config`, `tokenizer`, and `weights`.
- `fetch_model_checkpoint` returns a plaintext `ModelBundle`.
- `merge_adapter_into_base` merges LoRA into base on the miner.
- The miner runs `DnaBert2ForMaskedLM` with the full model in Candle.

### 1.2 Attack vectors

- Full weights can be dumped from memory.
- A malicious miner can redistribute the model.
- No enforcement of adapter-only training.

### 1.3 Architectural limitations

- Cannot achieve model confidentiality if miners hold the base.
- Full model training is too heavy for edge miners.
- No separation of concerns between model owner and trainer.

---

## 2. Target Architecture

Miners:
- Receive only `config`, `tokenizer`, and an encrypted **LoRA adapter seed**.
- Never receive the base model.
- Train LoRA matrices using attested forward passes from the orchestrator.
- Submit encrypted LoRA deltas.
- Can optionally train inside a TEE for extra assurance.

---

## 3. Alternatives

### 3.1 LoRA-only distribution

Miner downloads LoRA seed and tokenizer/config. Base stays orchestrator-side. Forward pass is done by orchestrator.

**Pros:**
- Minimal bandwidth.
- Base never leaves orchestrator.

**Cons:**
- Miner depends on orchestrator for forward pass.
- Attestation bandwidth can be high.

### 3.2 Gradient-only protocol (federated)

Orchestrator sends encrypted activations; miner computes gradients and sends them back.

**Pros:**
- Strongest base confidentiality.

**Cons:**
- High bandwidth per batch.
- High orchestrator load.

### 3.3 Split learning

Miner runs forward on input embedding layer; orchestrator runs rest.

**Pros:**
- No base weights on miner.

**Cons:**
- Leaks partial activation patterns.
- Complex protocol.

### 3.4 Recommended: Hybrid LoRA-only with attested forward

- Most practical for production.
- Base never leaves orchestrator.
- LoRA seed is small.
- Attestation can be batched.

---

## 4. API Changes

See `PRD-001-Secure-Model-Distribution.md`.

New RPCs:
- `GetTrainingArtifact` → returns LoRA seed.
- `AttestedForward` → returns signed hidden states.
- `SubmitLoRAUpdate` → returns validation result.

Remove:
- `ModelCheckpointV2.weights` full plaintext path.
- `merge_adapter_into_base` on miner.

---

## 5. Miner Workflow

1. Connect to approved orchestrator.
2. `GetTrainingArtifact(model_id, base_checkpoint, cached_base_hash, miner_public_key)`.
3. Decrypt LoRA seed with session key.
4. Build `LoRAModel` with uninitialized/rank-initialized LoRA matrices.
5. Fetch training batch.
6. Request `AttestedForward` for the batch.
7. Compute LoRA gradients using hidden states.
8. Update LoRA matrices with local AdamW.
9. After `local_steps`, submit `LoRAUpdate`.
10. Securely erase session key and LoRA seed.

---

## 6. Training Algorithms

### 6.1 LoRA gradient from attested hidden states

Given hidden states `H` and labels `Y`, the miner computes the output layer loss and back-propagates only through the LoRA matrices:

```
H' = H + (H @ A @ B) * (alpha / rank)   # LoRA update on hidden states
logits = H' @ W_out                       # output projection
cross_entropy(logits, Y)
gradient w.r.t. A and B only
```

This requires `H` and `W_out` to be known. `W_out` is the LM head, which is part of the base and must not be sent. Options:
- **Option A (recommended):** Orchestrator also sends the encrypted LM head projection matrix. This is a small matrix and can be sent per session. The miner can train LoRA without the full base.
- **Option B:** The orchestrator computes the final logits and returns them; the miner back-propagates only to LoRA. This hides `W_out` but reveals labels and logits.

For this PRD, **Option A** is recommended: send `W_out` (and optionally `W_in` for attention LoRA) encrypted under the session key. These matrices are small compared to the full model and do not reveal the base transformer layers.

---

## 7. Security

- Miner process has no access to base transformer weights.
- LoRA seed and LM head are encrypted and zeroized after use.
- Miner wallet key is separate from model session key.
- Optionally run miner training in a TEE.

---

## 8. Blockchain Changes

- `ModelGovernance.sol` adds `minStakeToTrain` and `loraOnly` flag.
- `OrchestratorRegistry` approves miners for a model.
- Block rewards only for validated LoRA deltas from approved miners.

---

## 9. Migration Plan

1. Add `GetTrainingArtifact` and `AttestedForward` to RPC.
2. Implement `LoRAModel` in Candle that can train with only LoRA + LM head.
3. Update `xenom-miner` to use new RPCs.
4. Keep full-weight path as legacy under `XENO_MINER_LEGACY_MODE`.
5. Deprecate after governance vote.

---

## 10. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| LM head reveals model info | Medium | Send only necessary output matrices; rotate session keys. |
| Attestation latency | Medium | Batch requests; cache hidden states. |
| LoRA extraction over time | High | Limit local steps; merge deltas only on orchestrator. |

---

## 11. Deliverables

This PRD is the design input for:
- `PRD-001-Secure-Model-Distribution.md`
- `PRD-002-Orchestrator-Service.md`
- `PRD-006-Confidential-Computing.md`
