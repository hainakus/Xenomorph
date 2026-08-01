# PRD-010: Migration Plan

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current architecture must evolve into the target secure architecture without downtime, without losing models, and without forking the chain. This PRD defines a phased, backward-compatible migration.

---

## 1. Migration Goals

- No chain fork.
- No model unavailability.
- Backward compatibility for miners and gateways.
- Zero downtime for active inference.
- Governance approval for breaking changes.

---

## 2. Phase 1: Critical Security (software hardening)

### 2.1 Key management

- Replace shared `XENO_MODEL_KEY` with per-model master keys.
- Use HKDF to derive `EK_m` and session keys.
- Store master key in KMS/Vault or at least in a file with restricted permissions (intermediate step).

### 2.2 Software hardening

- `mlock` model plaintext on orchestrator.
- `zeroize` keys and plaintext buffers.
- `seccomp`/`AppArmor` for inference workers.
- Add model artifact signing and verification.

### 2.3 Storage

- Introduce storage adapter interface.
- Add IPFS/S3 support alongside local filesystem.
- Backfill CIDs as placeholders.

### 2.4 Compatibility

- Keep full-weight WebSocket transfer as legacy.
- Add `ArtifactType` and `GetTrainingArtifact` RPC.

---

## 3. Phase 2: Storage Redesign

### 3.1 Content addressing

- Compute `blake3(encrypted_blob)` as `artifactHash`.
- Store real CIDs in `ModelRegistry`.
- Gossip real CIDs.

### 3.2 Decentralized storage

- Pin encrypted artifacts on IPFS.
- Store genesis checkpoints on Arweave.
- Use S3 as fallback.

### 3.3 Verification

- All downloads verify hash and signature.
- Remove plaintext WebSocket transfer for new models.

---

## 4. Phase 3: Orchestrator

### 4.1 Extract orchestrator

- Create `xenom-orchestrator` crate.
- Move inference and model management out of unified `xenom` node.
- Keep `xenom` as consensus + training coordinator.

### 4.2 Worker pool

- Implement worker process model.
- Add Candle worker.
- Add ONNX worker.

### 4.3 Scheduler

- Add request queue and health checks.
- Add metrics.

---

## 5. Phase 4: AI Gateway

### 5.1 Add auth/rate limiting

- API key management.
- JWT/OAuth support.
- Token bucket rate limiting.

### 5.2 Streaming and batching

- Implement SSE streaming.
- Add request batching.

### 5.3 Multi-orchestrator

- Connection pool to multiple orchestrators.
- Health-aware routing.
- Usage accounting.

---

## 6. Phase 5: Inference Backends

- Add `InferenceBackend` trait.
- Refactor Candle.
- Add ONNX Runtime.
- Add vLLM worker.
- Add TensorRT-LLM worker.
- Add llama.cpp worker.

---

## 7. Phase 6: Miner Redesign

### 7.1 LoRA-only mode

- Add `LoRAModel` that does not require full base.
- Send encrypted LoRA seed and LM head to miner.
- Implement `AttestedForward`.

### 7.2 Gradual enforcement

- New models default to LoRA-only.
- Legacy models continue with full weights until deprecated.
- Governance votes on deprecation.

### 7.3 Reward changes

- Only validate LoRA deltas for new models.
- Full-weight training rewards reduced over time.

---

## 8. Phase 7: Confidential Computing

### 8.1 TEE workers

- SEV-SNP inference workers.
- NVIDIA CC for GPU inference.
- Attestation registry.

### 8.2 Miner TEE

- TEE LoRA training.
- Attestation required for block rewards.

---

## 9. Phase 8: ZK Proofs

- Verifiable gradient commitment.
- STARK-based training proofs.
- On-chain verification.

---

## 10. Backward Compatibility

### 10.1 RPC

- Append new RPC variants to preserve binary compatibility.
- Add `legacy` flag to `GetModelCheckpoint`.

### 10.2 Blockchain

- Deploy v2 contracts with fallback to v1.
- Use governance for deprecation.

### 10.3 Models

- Old models can continue in legacy mode.
- New models use secure distribution.

---

## 11. Downtime Avoidance

- Blue-green orchestrator deployment.
- Run old and new gateway versions behind a load balancer.
- Keep redundant IPFS pins and S3 replicas.

---

## 12. Rollback

- Each phase has a rollback plan:
  - Revert config flags.
  - Re-deploy previous contract version.
  - Re-enable legacy RPCs.

---

## 13. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Governance delays | High | Phased proposals; clear incentives. |
| Miner attrition | High | Maintain rewards during transition; provide tooling. |
| Contract upgrade bugs | High | Use UUPS; test on devnet/simnet. |
| Performance regression | Medium | Benchmark each phase; keep legacy fallback. |
| Storage cost | Medium | Use IPFS for distribution; Arweave only for genesis. |
