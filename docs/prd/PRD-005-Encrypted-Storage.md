# PRD-005: Encrypted Storage

## Status
Design-only PRD. No production code is produced in this document.

## Context

Current model storage uses AES-256-GCM encrypted files on the local filesystem (`seed-node/src/model/storage.rs`), with a shared `XENO_MODEL_KEY` and no decentralized storage. This PRD designs a content-addressed, decentralized, and encrypted storage layer.

---

## 1. Problem

### 1.1 Current implementation

- `seed-node/src/model/storage.rs` writes `config.enc`, `tokenizer.enc`, `weights.enc` to `XENO_MODELS_DIR`.
- `xenom-miner/src/model_client.rs` downloads full plaintext model weights over WebSocket.
- The `cid` in `Announcement` is a placeholder equal to `weights_hash`.
- No IPFS, Arweave, or S3 integration.

### 1.2 Attack vectors

- Local disk theft exposes encrypted files; with the shared key, all models are compromised.
- No content verification: a malicious node can serve a modified encrypted file.
- Single point of failure: if the orchestrator's disk fails, the model is unavailable.

### 1.3 Architectural limitations

- Cannot scale storage across regions.
- Cannot prove that an artifact is stored by the network.
- Miners cannot fetch from a decentralized source.

---

## 2. Target Architecture

- All model artifacts are encrypted before storage.
- Artifacts are content-addressed by `blake3(encrypted_blob)`.
- Storage backends: IPFS (libp2p), Arweave, S3-compatible object stores.
- Orchestrator and miners download by CID and verify hash.
- Local cache is encrypted and periodically garbage-collected.

---

## 3. Alternatives

### 3.1 Option A: IPFS only

**Pros:**
- True decentralization.
- CID-based addressing.

**Cons:**
- File availability depends on pinning.
- Large model files (100MB+) are slow to pin/fetch.
- Corporate firewalls may block libp2p.

### 3.2 Option B: Arweave only

**Pros:**
- Permanent, tamper-proof storage.
- Incentivized replication.

**Cons:**
- High one-time cost.
- Slow for frequent updates (each checkpoint is a new transaction).

### 3.3 Option C: S3 only

**Pros:**
- Fast, cheap, familiar.

**Cons:**
- Centralized; single provider failure.

### 3.4 Recommended: Option D — Multi-backend with IPFS primary, S3/Arweave fallback

Use a pluggable storage adapter interface:

```rust
#[async_trait]
trait ModelStorageBackend {
    async fn put(&self, cid: &str, data: &[u8]) -> Result<()>;
    async fn get(&self, cid: &str) -> Result<Vec<u8>>;
    async fn exists(&self, cid: &str) -> Result<bool>;
    async fn delete(&self, cid: &str) -> Result<()>;
}
```

**Rationale:**
- IPFS is the default for decentralized distribution.
- S3 is the fallback for speed and firewall compatibility.
- Arweave is used for genesis checkpoints and governance-approved base models.

---

## 4. Storage Adapter Design

### 4.1 IPFS

- Use `rust-ipfs` or `ipfs-api` crate.
- Pin encrypted blobs.
- CIDv1 with raw codec and blake3 multihash.
- Provide public HTTP gateway fallback.

### 4.2 Arweave

- Use `arweave-rs` or HTTP API.
- Bundle encrypted data with tags: `model_id`, `version`, `artifact_hash`.
- CID is `ar://<transaction_id>`.

### 4.3 S3

- Bucket path: `{bucket}/{model_id}/{version}/{artifact_hash}.enc`.
- Object metadata: `Content-Type: application/octet-stream`, `x-amz-meta-model-id`, `x-amz-meta-artifact-hash`.
- Presigned URLs for authenticated miners.

### 4.4 Local cache

- Path: `{XENO_MODELS_DIR}/cache/{artifact_hash:0:2}/{artifact_hash:2:2}/{artifact_hash}.enc`.
- Store only encrypted blobs.
- LRU eviction with `XENO_MODEL_CACHE_SIZE_GB`.
- Verify hash on read.

---

## 5. Integrity and Verification

1. Download blob for CID.
2. Compute `blake3(blob)` and compare with `artifactHash` from blockchain.
3. Decrypt with `EK_m` (per-model encryption key).
4. Verify decrypted safetensors header and compute plaintext hash.
5. Verify `signature` over `artifactHash || artifactCID` using on-chain `publicKey`.

---

## 6. Encryption

- Use AES-256-GCM with 12-byte random nonce (keep current format `nonce || ciphertext`).
- Derive per-model key `EK_m = HKDF-SHA256(MK_m, "enc", model_id || version)`.
- For per-miner LoRA seeds, use X25519/HPKE.

---

## 7. Garbage Collection

- GC based on:
  - Model deprecated on-chain.
  - Cache size exceeded (LRU).
  - TTL for temporary artifacts.
- Never delete the latest active checkpoint.
- Historical checkpoints retained according to `XENO_CHECKPOINT_HISTORY_SIZE`.

---

## 8. API Changes

### 8.1 P2P gossip

- `Announcement.cid` becomes a real CID string or a compact encoding.
- Sign `cid` + `artifactHash` + `model_id`.

### 8.2 RPC

- `GetTrainingArtifact` returns `artifactCID` instead of `weights`.
- Miner downloads from storage backend and verifies.

---

## 9. Migration Plan

1. Implement storage adapter interface and IPFS/S3 backends.
2. Update `seed-node` and `xenom` to upload artifacts to IPFS and store CID on-chain.
3. Update miners to download by CID.
4. Backfill existing models to IPFS with placeholder CIDs.
5. Deprecate full-weight WebSocket transfer.

---

## 10. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| IPFS availability | Medium | Always pin on multiple orchestrators; S3 fallback. |
| Arweave cost | Medium | Use only for genesis and governance milestones. |
| S3 centralization | Medium | Support multiple S3-compatible providers. |
| Hash mismatch | High | Verify all downloaded blobs before decryption. |

---

## 11. Deliverables

This PRD is the design input for:
- `PRD-001-Secure-Model-Distribution.md`
- `PRD-002-Orchestrator-Service.md`
- `PRD-004-Model-Registry.md`
