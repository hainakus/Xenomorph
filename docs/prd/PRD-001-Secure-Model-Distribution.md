# PRD-001: Secure Model Distribution

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current Xenomorph implementation stores model weights encrypted with AES-256-GCM and shares the same `XENO_MODEL_KEY` between orchestrators and miners. The miner then decrypts the full base weights and loads them into Candle for training. This document designs a production-ready secure model distribution protocol that ensures miners never receive plaintext full weights while still allowing useful training.

---

## 1. Problem

### 1.1 Current implementation

- `seed-node/src/model/storage.rs` and `xenom-miner/src/model_client.rs` encrypt/decrypt model files using the same `XENO_MODEL_KEY`.
- `xenom-miner/src/model_client.rs::decrypt_v2` returns a `ModelBundle` whose docstring explicitly states: "The returned `ModelBundle` is always plaintext; if the node sent encrypted files they are decrypted with the same `XENO_MODEL_KEY` used by the node."
- `ModelCheckpointV2` may be sent as full weights or as a LoRA adapter, but the miner decrypts and merges adapters with a cached plaintext base, ultimately loading the full model into `DnaBert2ForMaskedLM`.

### 1.2 Attack vectors

- **Model extraction:** A malicious or compromised miner can dump the decrypted base weights from memory or temporary files and redistribute them.
- **Key leakage:** The same symmetric key is present in every miner and every orchestrator. Compromise of any single miner key file exposes the model to all participants.
- **No privilege separation:** There is no cryptographic boundary between the orchestrator (who needs plaintext) and the miner (who should not).
- **Weak authentication:** Miners authenticate via BIP39-derived addresses for block signing, but the download RPC does not tie checkpoint delivery to a specific authorization credential for model access.

### 1.3 Architectural limitations

- Full fine-tuning on the miner is incompatible with the security goal of keeping base weights confidential.
- A single shared key cannot support per-model, per-version, or per-miner key rotation.
- The current design cannot enforce that the miner deletes plaintext weights after training.

---

## 2. Target Architecture

### 2.1 Trust boundaries

- **Orchestrators** are trusted to decrypt and run inference. They hold the model master key.
- **Miners** are untrusted with respect to the base model. They may train LoRA or adapters but must not reconstruct the full base weights.
- **Blockchain** is the source of truth for active model ID, base checkpoint hash, and approved adapter hashes.
- **Storage layer** is untrusted: all stored artifacts must be encrypted and content-addressed.

### 2.2 Security guarantees

- Miners receive only encrypted artifacts.
- The base model cannot be reconstructed from miner-held data alone.
- Each model version has its own key hierarchy.
- Key material is ephemeral on miners and securely erased after use.
- All model artifacts are signed and integrity-verified before use.

### 2.3 Distribution model

```
Orchestrator
  ├─ holds master key MK_m for model m
  ├─ downloads/loads full base weights
  ├─ for each miner session, derives ephemeral session key SK_{m,s}
  ├─ encrypts a LoRA initialization or a forward-pass service credential with SK_{m,s}
  └─ sends encrypted artifact to miner

Miner
  ├─ receives only encrypted artifact (never base weights)
  ├─ requests a training task over authenticated WebSocket
  ├─ receives masked input / adapter gradients / projection matrices encrypted to SK_{m,s}
  ├─ trains LoRA or adapter locally on the provided data
  └─ submits encrypted gradient update or LoRA delta to orchestrator
```

---

## 3. Alternatives

### 3.1 Option A: LoRA-only distribution

Orchestrator keeps the full base. Miners download only a LoRA adapter (initialized to zero or to a previously merged adapter) plus tokenizer/config. The base is never sent.

**Pros:**
- Minimal protocol change.
- Base weights remain orchestrator-side.
- Compatible with current `LoraConfig`.

**Cons:**
- Still requires the LoRA adapter to be sent to the miner; if the miner accumulates and merges many adapters, the full model can be reconstructed.
- Miner cannot validate its own adapter update against the base without a forward pass, creating a trust dependency.

### 3.2 Option B: Encrypted activation / split learning

Orchestrator runs forward passes on the base model for each miner batch, sends activations to the miner, and the miner computes gradients on its LoRA adapter using those activations.

**Pros:**
- Base model never leaves the orchestrator.
- Miner has no access to base weights or embeddings.

**Cons:**
- Massive bandwidth per batch (hidden states for every token).
- High orchestrator compute cost (all forward passes on orchestrator).
- Latency and availability issues for large miner pools.

### 3.3 Option C: Homomorphic / functional encryption for gradients

Miner receives an encrypted view of the model that allows it to compute gradients without decryption.

**Pros:**
- Strongest confidentiality.

**Cons:**
- Not practical for transformer-scale models today; orders of magnitude slowdown.
- No production-ready Rust libraries for the required HE operations.

### 3.4 Recommended: Option D — Hybrid base-free LoRA with forward attestation

Combine LoRA-only distribution with periodic orchestrator attestation:

1. Orchestrator keeps the full base and decrypts only in RAM.
2. Miner downloads `config`, `tokenizer`, and an encrypted **LoRA adapter seed** (initialized, not the full base).
3. For the first training batch, the miner requests an attested forward evaluation from the orchestrator: the orchestrator runs a forward pass on a small representative batch, signs the resulting per-token hidden states and the loss, and returns them.
4. The miner uses the signed activations to compute its LoRA adapter gradients locally.
5. After N local steps, the miner submits the encrypted LoRA delta. The orchestrator validates by re-running the training batch on its own base+merged adapter and comparing the gradient commitment.
6. The orchestrator merges the validated delta into the base, produces a new encrypted full checkpoint, and publishes the new base hash.

**Rationale:**
- Base model never leaves the orchestrator.
- Miner performs useful gradient computation without reconstructing the model.
- Compatible with current gradient validation infrastructure.
- Bandwidth is dominated by the LoRA adapter (small) and one attested forward batch per training round.

---

## 4. API Changes

### 4.1 gRPC / WebSocket RPC

Current RPC messages are in `xenom-miner/src/rpc/messages.rs` and `seed-node/src/rpc/messages.rs`.

**New fields:**

```rust
// In ModelCheckpointV2
pub struct ModelCheckpointV2 {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub base_hash: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    // artifact may be a LoRA adapter, an encrypted projection, or empty for from-scratch
    pub artifact: Vec<u8>,
    pub artifact_type: ArtifactType, // Base, LoRA, Adapter, GradientTask
    pub artifact_hash: [u8; 32],
    pub encrypted: bool,
    pub recipient_key_fingerprint: [u8; 32], // optional, for audit
    pub signature: [u8; 64], // orchestrator signature over artifact_hash || base_hash
}

pub enum ArtifactType {
    Base,
    LoRA,
    Adapter,
    GradientTask,
}

// New request/response
pub struct GetTrainingArtifact {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub cached_base_hash: Option<[u8; 32]>,
    pub miner_public_key: [u8; 33],
}

pub struct AttestedForwardRequest {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub labels: Vec<Vec<u32>>,
    pub mask: Vec<Vec<u8>>,
    pub miner_public_key: [u8; 33],
}

pub struct AttestedForwardResponse {
    pub hidden_states_hash: [u8; 32], // hash of serialized hidden states
    pub hidden_states: Vec<u8>,       // encrypted with session key
    pub loss: f64,
    pub token_count: u32,
    pub signature: [u8; 64],
    pub ephemeral_public_key: [u8; 33],
    pub session_nonce: [u8; 12],
    pub auth_public_key: [u8; 33],
}

pub struct SubmitLoRAUpdate {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub lora_delta: Vec<u8>,       // encrypted adapter safetensors
    pub lora_delta_hash: [u8; 32], // blake3 of plaintext adapter
    pub gradient_commitment: [u8; 32],
    pub participant_weight: f32,
    pub miner_address: String,
    pub miner_public_key: [u8; 33],
    pub encrypted: bool,
    pub ephemeral_public_key: [u8; 33],
    pub session_nonce: [u8; 12],
    pub signature: [u8; 64],
    pub auth_public_key: [u8; 33],
}
```

### 4.2 P2P gossip

Current `Announcement` in `crypto/model-crypto/src/gossip.rs` already has `cid`, `weights_hash`, `signature`.

**Changes:**
- `cid` must no longer be a placeholder; it must be a real content identifier of the encrypted artifact.
- Add `artifact_type` and `artifact_hash` fields.
- Sign `model_id || weights_hash || artifact_hash || cid || timestamp || is_genome || node_address || public_key || listen_addr`.

### 4.3 proto/inference.proto

No immediate breaking changes to the public inference API. `block_height` remains supported. Internal training RPC (WebSocket/Borsh) changes per section 4.1.

---

## 5. Blockchain Changes

### 5.1 ModelRegistry.sol

Add fields to `Model`:
- `string artifactCID` — content identifier of the current encrypted artifact.
- `bytes32 artifactHash` — hash of the encrypted artifact.
- `bytes32 publicKey` — orchestrator/owner public key for model artifact signatures.
- `bytes signature` — signature over `modelId || version || artifactHash || artifactCID`.
- `bytes32 encryptionKeyHash` — hash of the model's master key (for audit, never the key itself).

Add events:
- `event ModelArtifactUpdated(bytes32 indexed modelId, bytes32 artifactHash, string artifactCID, uint256 blockHeight);`
- `event ModelSignatureUpdated(bytes32 indexed modelId, bytes signature);`

### 5.2 ModelGovernance.sol

Add fields to `ActiveModel`:
- `string artifactCID`
- `bytes32 artifactHash`
- `bytes32 publicKey`
- `bool baseWeightsPubliclyAvailable` — must be `false` for any model in secure mode.

Add proposal validation:
- Reject proposals where `baseWeightsPubliclyAvailable == true` unless explicitly whitelisted by governance.

### 5.3 New contract: ModelKeyEscrow (optional, future)

A contract that stores key material only in encrypted form, released to approved orchestrators. This is out of scope for Phase 1; recommended for Phase 3+.

---

## 6. Storage Changes

### 6.1 Content addressing

Every artifact (full model, LoRA, adapter, genome) is stored as an encrypted blob and addressed by `blake3(encrypted_blob)`.

- `IPFS`: pin encrypted blobs; `artifactCID` is the IPFS CID.
- `Arweave`: store encrypted blobs permanently; `artifactCID` is the Arweave transaction ID.
- `S3`: use bucket + key derived from `artifact_hash`; object metadata contains `signature` and `encryption_key_hash`.

### 6.2 Cache and garbage collection

- Orchestrator cache keeps the most recent `N` full checkpoints and all pending LoRA deltas.
- Miner cache keeps only the current LoRA adapter and config/tokenizer; no base weights.
- GC policy: delete local plaintext LoRA seeds after `XENO_MINER_CACHE_TTL` minutes of inactivity; do not delete encrypted artifacts before on-chain deprecation.

### 6.3 Integrity verification

Before loading, verify:
1. Downloaded blob hash matches `artifactHash`.
2. Signature over artifact metadata is valid against `publicKey` in the contract.
3. Decryption with the master key succeeds and the decrypted plaintext passes safetensors validation.

---

## 7. Security Changes

### 7.1 Key hierarchy

```
MK_m     = master key for model m (only orchestrator)
EK_m     = HKDF-SHA256(MK_m, "enc", version)  // encryption key
SK_{m,s} = HKDF-SHA256(EK_m, "session" || miner_public_key || nonce)  // per-session key
KAuth_m  = HKDF-SHA256(MK_m, "auth")  // signing key for model artifacts
```

- `MK_m` never leaves the orchestrator.
- `EK_m` is derived per model version.
- `SK_{m,s}` is derived per miner session; the miner receives `SK_{m,s}` encrypted to its public key (or ephemeral ECDH).

### 7.2 Encryption

- Continue using AES-256-GCM for model artifacts.
- For per-miner LoRA seeds, use X25519 + AES-256-GCM or HPKE (RFC 9180).
- Replace `XENO_MODEL_KEY` with a per-model master key stored in a KMS/Vault.

### 7.3 Model signing

- Orchestrator signs `artifactHash` with `KAuth_m`.
- On-chain `ModelRegistry` stores `publicKey = secp256k1(KAuth_m)`.
- Miner verifies signature before loading any artifact.

### 7.4 Memory-only decryption

- Orchestrator decrypts the base model to a `mlock`-ed or `memfd_secret`-backed memory region.
- Use `zeroize` on all keys and temporary plaintext buffers.
- Implement explicit `secure_erase` for model plaintext after unload or on shutdown.

### 7.5 Key rotation

- When a model version updates, derive a new `EK_m` from a new `MK_m`.
- Re-encrypt all historical checkpoints with the new `EK_m` or keep separate key per checkpoint.
- Blockchain records `encryptionKeyHash` for each version.

---

## 8. Networking

### 8.1 Download flow

```
Miner -> WebSocket RPC: GetTrainingArtifact { model_id, base_checkpoint, cached_base_hash, miner_public_key }
Orchestrator -> validates request, derives SK_{m,s}, encrypts LoRA seed with SK_{m,s}
Orchestrator -> returns ModelCheckpointV2 { artifact: encrypted_seed, artifact_type: LoRA, signature }
Miner -> verifies signature, decrypts seed with SK_{m,s}
Miner -> trains LoRA on local data
Miner -> requests AttestedForward for gradient validation
Orchestrator -> returns encrypted attested hidden states
Miner -> computes LoRA delta
Miner -> SubmitLoRAUpdate
```

### 8.2 Checkpoint synchronization

- P2P gossip announces `artifactCID` + `artifactHash` (not the full artifact).
- Orchestrators fetch missing artifacts from IPFS/Arweave/S3.
- Miners do not participate in P2P checkpoint sync; they request artifacts directly from orchestrators.

### 8.3 Orchestrator discovery

- Blockchain `ModelRegistry` stores `publicKey` and a list of approved orchestrator endpoints (optional off-chain registry).
- Miners discover orchestrators via on-chain events and P2P gossip listen addresses.

---

## 9. Miner Changes

### 9.1 No full weights

- Remove `fetch_model_checkpoint` full-path decryption.
- `xenom-miner/src/model_client.rs` returns only `config`, `tokenizer`, and an encrypted LoRA/adapter artifact.
- `merge_adapter_into_base` is removed from the miner; merging happens only on the orchestrator.

### 9.2 Training workflow

1. Download config/tokenizer/LoRA seed.
2. Build a `LoRAModel` that wraps the base architecture with initialized LoRA matrices.
3. For the first batch, request `AttestedForward`.
4. Compute LoRA gradients using attested hidden states.
5. Apply local AdamW to LoRA matrices.
6. After `local_steps`, serialize LoRA delta, encrypt with session key, and submit.

### 9.3 Memory management

- Keep LoRA matrices and current batch in memory.
- Do not persist decrypted base weights.
- Securely erase session keys after training round.

---

## 10. Migration Plan

See `PRD-010-Migration-Plan.md`.

High-level:
1. Introduce new `ArtifactType` fields alongside existing `weights` (backward-compatible).
2. Update orchestrators to support secure distribution.
3. Deprecate full-weight downloads for miners after a governance vote.
4. Switch active models to LoRA-only distribution.
5. Remove `XENO_MODEL_KEY` from miner defaults.

---

## 11. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| LoRA extraction via accumulated deltas | High | Limit number of local steps; apply differential privacy; merge deltas on orchestrator side and never return merged weights. |
| Attested forward bandwidth | Medium | Batch multiple miner requests; cache hidden states for identical batches. |
| Miner cannot train without base | Medium | Use split learning or TEE as future enhancements. |
| Key management complexity | High | Use a KMS/Vault and per-model keys from day one. |
| Backward compatibility | Medium | Keep legacy RPC for one protocol version; deprecate via governance. |

---

## 12. Deliverables

This PRD is the design input for:
- `PRD-002-Orchestrator-Service.md`
- `PRD-007-Miner-Redesign.md`
- `PRD-005-Encrypted-Storage.md`
