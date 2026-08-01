# PRD-009: Zero-Knowledge Training Proofs

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current `TrainingProof` in `consensus/core/src/pow/training_proof.rs` contains a `ZKProof` field, but validation is structural only (`xenom/src/training/coordinator.rs` checks the proof exists, not its cryptographic validity). This PRD designs a real ZK proof system for training.

---

## 1. Problem

### 1.1 Current implementation

- `TrainingProof` has fields `loss_before`, `loss_after`, `base_checkpoint`, `gradients_commitment`, and `zk_proof`.
- `coordinator.rs` checks `zk_proof` length and parses it but does not verify a cryptographic proof.
- `gradient_validator.rs` re-executes training to verify `loss_before` and `loss_after`.

### 1.2 Attack vectors

- Malicious miner can lie about gradients.
- Re-execution is expensive and may not match due to floating-point differences.
- No cryptographic guarantee that a block is backed by valid training.

### 1.3 Architectural limitations

- Consensus cannot independently verify training.
- Re-execution does not scale to large models and many miners.

---

## 2. Target Architecture

Replace placeholder ZK with a **succinct proof of training** that can be verified cheaply by consensus.

Requirements:
- Prove that `loss_after` was obtained by applying the claimed gradients to the base model.
- Prove that the gradients were computed from a valid training batch.
- Prove that the miner did not have access to full base weights (optional, tied to `PRD-007`).
- Verification must be fast enough for consensus (ideally < 1s on CPU).

---

## 3. Alternatives

### 3.1 STARKs for training trace

Generate a STARK proving that a deterministic training step (forward, loss, backward, optimizer update) was executed correctly.

**Pros:**
- Transparent setup.
- Fast verification.
- No trusted ceremony.

**Cons:**
- Prover time is high for transformer models.
- Need to arithmetize neural network operations.

### 3.2 SNARKs with Groth16 / PLONK

Pre-compile the training step into a circuit; generate a SNARK.

**Pros:**
- Small proofs.
- Fast verification.

**Cons:**
- Trusted setup per circuit update.
- Prover requires GPU/FPGA for large models.

### 3.3 Sum-check based proofs (GKR)

Use sum-check protocols to prove layer-wise forward/backward passes.

**Pros:**
- More efficient than full STARKs for matrix ops.
- No trusted setup.

**Cons:**
- Complex implementation.
- Limited tooling.

### 3.4 Verifiable computation of gradient commitment (recommended for Phase 1)

Instead of proving the entire training step, prove that the submitted `gradients_commitment` is the blake3 hash of a tensor delta whose application to the base model reduces the loss from `loss_before` to `loss_after`.

The proof is a Merkle commitment to the gradient delta plus an evaluation of the loss function. The orchestrator/validator re-executes the forward pass on a small subset of parameters (random checkpoint) and the miner provides an opening proof.

**Pros:**
- Feasible with existing Merkle/commitment tools.
- Faster than full STARKs.

**Cons:**
- Still requires some re-execution.

### 3.5 Recommended long-term: STARK-based training trace

Use a STARK prover (e.g., `starkware`, `risc0`, `succinct`) to prove the entire LoRA training step. Start with verifiable gradient commitment for Phase 1 and migrate to full STARKs once tooling matures.

---

## 4. Design

### 4.1 Proof of LoRA delta (Phase 1)

Given:
- Base model hash `B`.
- LoRA delta `D`.
- Gradient commitment `C = blake3(D)`.
- `loss_before`, `loss_after`.
- Training batch `X` (committed by genome Merkle root).

Prover (miner) does:
1. Compute `D` by training LoRA on `X`.
2. Compute `C = blake3(D)`.
3. Compute `loss_before` and `loss_after` by forward evaluation on the (orchestrator-provided) hidden states and LM head.
4. Build a Merkle tree over `D`.
5. Generate a random challenge `r` from `C || B || block_hash`.
6. Provide opening for a random subset of `D` at indices derived from `r`.

Verifier (orchestrator) does:
1. Recompute `C` from `D`.
2. Verify Merkle opening.
3. Recompute the sampled gradient entries and confirm they are consistent with `loss_after - loss_before`.

### 4.2 STARK-based (Phase 2)

- Define a Cairo/RISC-V program that runs the LoRA training step.
- Miner runs the prover inside `risc0` or `succinct`.
- Verifier runs the STARK verification in Rust, cheaply.

---

## 5. Blockchain Changes

- `consensus/core/src/pow/training_proof.rs` adds `proof_type` field.
- `xenom/src/training/gradient_validator.rs` verifies ZK proof instead of full re-execution.
- `DifficultyTarget` may require a minimum proof version.
- New events for proof scheme upgrades.

---

## 6. API Changes

- `TrainingBlock` includes `zk_proof` bytes and `proof_type`.
- Miner submits proof along with `GradientUpdate`.
- Orchestrator validates proof before block submission.

---

## 7. Migration Plan

1. Keep re-execution validation.
2. Add verifiable gradient commitment (Merkle opening) as `proof_type = 1`.
3. Add STARK verification as `proof_type = 2`.
4. Gradually require `proof_type >= 1` for blocks.
5. Eventually require `proof_type = 2`.

---

## 8. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Prover too slow | High | Start with Merkle-based; move to STARKs incrementally. |
| Proof verification mismatch | High | Extensive test vectors across devices. |
| Floating-point non-determinism | High | Use fixed-point or deterministic floating-point in circuits. |
| STARK tooling immaturity | Medium | Use `risc0` or `succinct` zero-knowledge VMs. |

---

## 9. Deliverables

This PRD is the design input for:
- `PRD-007-Miner-Redesign.md`
- `PRD-002-Orchestrator-Service.md`
