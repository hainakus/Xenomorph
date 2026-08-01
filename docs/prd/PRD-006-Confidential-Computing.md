# PRD-006: Confidential Computing

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current system has no confidential computing. Model keys are derived from environment variables, plaintext weights are loaded into ordinary process memory, and miners receive full decrypted weights. This PRD designs how to integrate TEEs and remote attestation in the future.

---

## 1. Problem

### 1.1 Current implementation

- `crypto/model-crypto/src/lib.rs` derives `XENO_MODEL_KEY` from environment.
- `seed-node/src/model/storage.rs` decrypts to a `Vec<u8>` and writes encrypted files to disk.
- `xenom-miner/src/model_client.rs` decrypts and loads full plaintext weights.
- No `mlock`, no `zeroize` of model plaintext, no TEE.

### 1.2 Attack vectors

- Cold boot attacks on orchestrator RAM.
- Malicious admin dumps model from memory.
- Compromised miner exfiltrates weights.
- Supply-chain attacks on model artifacts.

### 1.3 Architectural limitations

- No hardware-based isolation.
- No proof that the orchestrator is running the expected code.
- Cannot enforce "decrypt only in RAM" cryptographically.

---

## 2. Target Architecture

Confidential computing is an **optional, future hardening layer**, not a blocker for the core redesign.

Layers:
1. **Software hardening (Phase 1):** `mlock`, `zeroize`, `seccomp`, minimal permissions.
2. **TEE inference workers (Phase 2):** run inference workers inside AMD SEV-SNP, Intel TDX, or NVIDIA Confidential Computing (CC) enclaves.
3. **Remote attestation (Phase 3):** attestation reports bound to model keys; model artifacts only decrypt inside attested enclaves.
4. **Miner TEE training (Phase 4):** miners train LoRA inside TEEs; remote attestation required for block rewards.

---

## 3. Alternatives

### 3.1 AMD SEV-SNP

**Pros:**
- Mature, cloud-ready (AWS, Azure, GCP).
- Protects memory from hypervisor.
- Can run entire Linux VMs.

**Cons:**
- Requires specific CPU support.
- Attestation infrastructure complexity.

### 3.2 Intel TDX

**Pros:**
- Strong isolation.
- Cloud availability growing.

**Cons:**
- Newer, less battle-tested.

### 3.3 NVIDIA Confidential Computing

**Pros:**
- GPU-accelerated TEE for inference and training.
- H100/H200 support.

**Cons:**
- Expensive hardware.
- Limited availability.

### 3.4 Recommended: Multi-TEE support

Start with **AMD SEV-SNP for orchestrator inference workers** (widest availability), then add **NVIDIA CC for GPU inference** and **Intel TDX** as options.

---

## 4. Design

### 4.1 Orchestrator TEE worker

- The orchestrator control plane runs outside the TEE.
- Each inference backend worker runs inside a TEE VM or container.
- The TEE worker:
  1. Generates an attestation report.
  2. Requests the model master key from a KMS only if the report is valid.
  3. Receives encrypted model artifact.
  4. Decrypts and loads into TEE memory.
  5. Serves inference.
  6. Destroys key and plaintext on exit.

### 4.2 Key release policy

KMS policy:
```json
{
  "release_key_if": {
    "measurement": "sha256:<expected_enclave_measurement>",
    "signer": "xenom-orchestrator-v<version>",
    "debug": false,
    "model_id": "<model_id>",
    "min_uptime": 0
  }
}
```

### 4.3 Remote attestation flow

1. TEE worker boots and generates an attestation report.
2. Worker sends report + nonce to KMS/Vault.
3. KMS validates report against known measurements.
4. KMS releases `EK_m` encrypted to the TEE's public key.
5. Worker decrypts `EK_m` inside TEE and loads model.
6. Periodically re-attest (every 5 minutes).

### 4.4 Blockchain attestation registry

- `OrchestratorRegistry.sol` stores an `attestationHash` per orchestrator.
- Miners only trust orchestrators whose attestation hash matches a known-good measurement.
- Slashing if an orchestrator fails to provide fresh attestation.

---

## 5. API Changes

### 5.1 gRPC

Add to `proto/inference.proto`:
```protobuf
message AttestationReport {
  bytes report = 1;
  bytes quote = 2;
  string tee_type = 3; // "sev", "tdx", "nvidia-cc"
}

rpc GetAttestation(GetAttestationRequest) returns (GetAttestationResponse);
```

### 5.2 Blockchain

- `OrchestratorRegistry.sol` adds `attestationHash` and `teeType` fields.
- New event `AttestationUpdated(address indexed orchestrator, bytes32 attestationHash, string teeType)`.

---

## 6. Miner TEE

- Miners run LoRA training inside a TEE.
- Attestation proves that the LoRA delta was computed without leaking base weights.
- The orchestrator only accepts LoRA deltas from attested miners.
- This is Phase 4; until then, use split learning/LoRA-only distribution.

---

## 7. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| TEE side-channel attacks | High | Keep TEE code minimal; use constant-time crypto; disable debug. |
| Attestation verification bugs | High | Use vendor SDKs; pin measurements in governance. |
| Hardware availability | Medium | Make TEE optional; non-TEE mode still works with software hardening. |
| Performance overhead | Medium | Benchmark SEV-SNP overhead; use NVIDIA CC for GPU workloads. |

---

## 8. Migration Plan

1. Implement software hardening (`mlock`, `zeroize`, `seccomp`).
2. Add optional TEE worker support behind feature flags.
3. Add attestation registry contract.
4. Gradually require attestation for high-value models via governance.

---

## 9. Deliverables

This PRD is the design input for:
- `PRD-002-Orchestrator-Service.md`
- `PRD-007-Miner-Redesign.md`
- `PRD-009-ZK-Training-Proofs.md`
