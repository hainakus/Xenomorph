# Phase 1: Critical Security Hardening — Implementation Notes

## Scope

This phase implements the foundational cryptographic primitives required by `PRD-001-Secure-Model-Distribution`, `PRD-005-Encrypted-Storage`, and `PRD-006-Confidential-Computing`:

- **HKDF-SHA256 key hierarchy** for per-model master/encryption/session/auth keys.
- **Artifact signing and verification** with secp256k1.
- **`mlock` + `zeroize` secret buffers** for software-only memory hardening.

## Crate

All new code lives in `crypto/model-crypto` alongside the existing AES-256-GCM helpers.

## New modules

### `crypto/model-crypto/src/key_hierarchy.rs`

Implements the PRD-001 key hierarchy:

```text
MK_m     = master key for model m (orchestrator only)
EK_m     = HKDF-SHA256(MK_m, "enc",   model_id || version)
SK_{m,s} = HKDF-SHA256(EK_m, "session", miner_public_key || nonce)
KAuth_m  = HKDF-SHA256(MK_m, "auth",  model_id)
```

Key types:

- `ModelSecret([u8; 32])` — zeroizing secret with `derive`, `encrypt`, `decrypt`, and `hash` helpers.
- `ModelKeyHierarchy { master, model_id, version }` — derives `EK_m`, `KAuth_m`, and per-session `SK_{m,s}`.

Tests verify:
- Deterministic derivation.
- Different models produce different keys.
- Session keys vary by miner public key and nonce.
- `ModelSecret` zeroizes.
- Derived encryption keys can encrypt and decrypt AES-GCM artifacts.

### `crypto/model-crypto/src/artifact_sign.rs`

Implements artifact signatures as specified in PRD-001:

- `ArtifactSigner::from_auth_key(KAuth_m)` — signs `artifact_hash || base_hash`.
- `ArtifactVerifier::from_public_key([u8; 33])` — verifies the signature.
- `ArtifactSignature` — holds 64-byte signature + 33-byte compressed public key.

The signing domain is `b"xenom-model-artifact-v1"` and the message is a SHA-256 hash of the domain + artifact hash + optional base hash. This mirrors the on-chain `ModelRegistry.publicKey` field.

Tests verify:
- Sign/verify roundtrip.
- Tampered artifact hash is rejected.
- Wrong public key is rejected.
- An all-zero secret key is rejected.

### `crypto/model-crypto/src/secret_buffer.rs`

Software memory hardening from PRD-006:

- `SecretBuffer` — growable `Vec<u8>` wrapper.
- `mlock()` / `munlock()` — on Unix uses `libc::mlock` and `libc::munlock`.
- `ZeroizeOnDrop` — overwrites the buffer with zeros when dropped.
- `secure_erase()` — explicit zeroization and unlock.

`mlock` may fail for unprivileged processes due to `RLIMIT_MEMLOCK`; the code allows this in tests.

Tests verify:
- Buffer read/write via `AsRef`/`AsMut`.
- `mlock`/`munlock` roundtrip.
- Drop zeroizes memory.

## Integration with existing code

The existing `encrypt`/`decrypt` AES-256-GCM functions in `lib.rs` are unchanged. `ModelSecret` wraps them as `encrypt`/`decrypt` methods so the new hierarchy can directly produce and consume encrypted artifacts.

The old `derive_encryption_key()` (based on `XENO_MODEL_KEY`) is kept as a legacy fallback and marked for deprecation in a later phase.

## Dependencies added

- `hkdf = "0.12.4"` (workspace)
- `zeroize` derive feature enabled workspace-wide
- `libc = "0.2"` in `crypto/model-crypto`

## Tests

Run with:

```bash
cargo test -p model-crypto
cargo clippy -p model-crypto
```

All 18 tests pass and clippy is clean for `model-crypto`.

## Next steps

To complete Phase 1, the following still need to be wired into production code:

1. Replace `XENO_MODEL_KEY` with a per-model `ModelKeyHierarchy` in `seed-node` and `xenom-orchestrator`.
2. Sign every `ModelCheckpointV2` artifact with `ArtifactSigner` before storage/P2P gossip.
3. Verify artifact signatures on the miner before loading.
4. Add `artifact_hash` and `artifact_type` to `Announcement` and include them in the signing digest.
5. Use `SecretBuffer` for decrypted plaintext in `seed-node/src/model/storage.rs` and `xenom-miner/src/model_client.rs`.
6. Add a KMS/Vault abstraction for `MK_m` (out-of-band for now).

## Commit

These changes are committed as part of the Phase 1 security-hardening track.
