# Implementation Roadmap

## Status
Design-only roadmap. No production code is produced in this document.

This roadmap orders the PRDs into executable phases. Each phase lists deliverables, dependencies, difficulty, risk, and expected impact.

---

## Phase 1: Critical Security Hardening

**Goal:** Fix the most critical gaps without changing the core protocol.

| PRD | Deliverable |
|-----|-------------|
| PRD-001 | Introduce HKDF key hierarchy; sign model artifacts; per-model encryption keys. |
| PRD-005 | Add storage adapter interface; IPFS/S3 support; verify hash and signature on download. |
| PRD-006 | `mlock`, `zeroize`, `seccomp` for orchestrator; software-only hardening. |

**Dependencies:** None.

**Difficulty:** Medium.

**Risk:** Low to Medium. Backward-compatible if legacy `XENO_MODEL_KEY` remains as fallback.

**Expected Impact:** Removes the single shared key and adds integrity verification; raises the bar for model extraction.

---

## Phase 2: Encrypted Storage and Content Addressing

**Goal:** Decentralize model storage and make all artifacts content-addressed.

| PRD | Deliverable |
|-----|-------------|
| PRD-005 | Real IPFS pinning, Arweave for genesis, S3 fallback; LRU cache. |
| PRD-004 | Update `ModelRegistry` with `artifactCID`, `artifactHash`, `signature`, `publicKey`. |

**Dependencies:** Phase 1.

**Difficulty:** Medium.

**Risk:** Medium. IPFS availability and gas costs for on-chain CIDs.

**Expected Impact:** Models can be distributed from decentralized storage; chain becomes source of truth for CIDs.

---

## Phase 3: Standalone Orchestrator

**Goal:** Extract inference and model management from the consensus node into a dedicated service.

| PRD | Deliverable |
|-----|-------------|
| PRD-002 | New `xenom-orchestrator` binary; worker pool; scheduler; health checks; metrics. |
| PRD-008 | `InferenceBackend` trait; Candle and ONNX workers. |

**Dependencies:** Phase 2.

**Difficulty:** High.

**Risk:** High. Adds a new process; must not break consensus or miner RPC.

**Expected Impact:** Scalable inference, process isolation, multi-backend support.

---

## Phase 4: AI Gateway Production Hardening

**Goal:** Make the gateway ready for public use.

| PRD | Deliverable |
|-----|-------------|
| PRD-003 | API keys, JWT/OAuth, rate limiting, streaming, batching, usage accounting, multi-orchestrator routing. |

**Dependencies:** Phase 3.

**Difficulty:** Medium.

**Risk:** Medium. New database and auth system.

**Expected Impact:** Public API with monetization and abuse controls.

---

## Phase 5: Additional Inference Backends

**Goal:** Support state-of-the-art inference engines.

| PRD | Deliverable |
|-----|-------------|
| PRD-008 | vLLM, TensorRT-LLM, llama.cpp workers. |

**Dependencies:** Phase 3.

**Difficulty:** High.

**Risk:** High. Complex dependencies and output consistency.

**Expected Impact:** Higher throughput, larger model support, deployment flexibility.

---

## Phase 6: Miner Redesign (LoRA-Only)

**Goal:** Miners no longer receive plaintext base weights.

| PRD | Deliverable |
|-----|-------------|
| PRD-007 | `LoRAModel`, `AttestedForward`, `GetTrainingArtifact`; remove full-weight path. |
| PRD-001 | Encrypted LoRA seed and LM head distribution. |

**Dependencies:** Phase 3.

**Difficulty:** Very High.

**Risk:** Very High. Changes the fundamental training protocol; may affect miner rewards and participation.

**Expected Impact:** Achieves the core security guarantee: base model stays with orchestrator.

---

## Phase 7: Confidential Computing

**Goal:** Add hardware-enforced isolation.

| PRD | Deliverable |
|-----|-------------|
| PRD-006 | SEV-SNP/TDX/NVIDIA CC workers; remote attestation registry; KMS key release. |

**Dependencies:** Phase 3 and 6.

**Difficulty:** Very High.

**Risk:** High. Hardware availability and attestation complexity.

**Expected Impact:** Strong protection against orchestrator operator exfiltration and miner cheating.

---

## Phase 8: ZK Training Proofs

**Goal:** Replace placeholder ZK with cryptographic training proofs.

| PRD | Deliverable |
|-----|-------------|
| PRD-009 | Verifiable gradient commitment; STARK-based training proof; on-chain verification. |

**Dependencies:** Phase 6.

**Difficulty:** Very High.

**Risk:** High. Prover performance and non-determinism.

**Expected Impact:** Consensus can verify training cheaply; removes re-execution bottleneck.

---

## Phase 9: Production Hardening

**Goal:** Prepare for mainnet.

| PRD | Deliverable |
|-----|-------------|
| PRD-010 | Full migration; rollback testing; security audit; performance benchmarks; documentation. |

**Dependencies:** All previous phases.

**Difficulty:** Medium.

**Risk:** Medium. Final audit may require changes.

**Expected Impact:** Production-ready network.

---

## Summary Matrix

| Phase | Difficulty | Risk | Dependencies | Expected Impact |
|-------|------------|------|--------------|-----------------|
| 1 — Security hardening | Medium | Low-Medium | None | High |
| 2 — Encrypted storage | Medium | Medium | Phase 1 | High |
| 3 — Orchestrator | High | High | Phase 2 | Very High |
| 4 — Gateway | Medium | Medium | Phase 3 | High |
| 5 — Backends | High | High | Phase 3 | High |
| 6 — Miner redesign | Very High | Very High | Phase 3 | Critical |
| 7 — Confidential computing | Very High | High | Phase 3, 6 | Very High |
| 8 — ZK proofs | Very High | High | Phase 6 | Very High |
| 9 — Production hardening | Medium | Medium | All | Critical |

---

## Governance Decision Points

1. **Phase 1:** Approve new key hierarchy and signature verification.
2. **Phase 2:** Approve storage backend selection and on-chain CID format.
3. **Phase 3:** Approve separation of orchestrator from consensus node.
4. **Phase 6:** Approve LoRA-only training and deprecation of full-weight mode.
5. **Phase 7:** Approve TEE attestation requirements.
6. **Phase 8:** Approve ZK proof scheme and verification costs.

---

## Breaking Changes

- **Phase 1:** Old `XENO_MODEL_KEY` becomes deprecated; new models use HKDF.
- **Phase 2:** `ModelRegistry` v2 schema; old v1 contracts remain read-only.
- **Phase 3:** `xenom` binary no longer serves gRPC inference by default; orchestrator required.
- **Phase 4:** API keys required for gateway; CORS restricted.
- **Phase 6:** Miners must upgrade to LoRA-only mode; old miners stop earning.
- **Phase 7:** TEE may become required for high-value models.
- **Phase 8:** New `TrainingProof` format; consensus verification changes.
