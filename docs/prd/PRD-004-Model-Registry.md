# PRD-004: Model Registry

## Status
Design-only PRD. No production code is produced in this document.

## Context

The current on-chain model registry is implemented in `smart-contracts/ModelRegistry.sol` and `smart-contracts/ModelGovernance.sol`. It stores `modelHash`, `version`, `submitter`, and `blockHeight`, but lacks CID/IPFS, public keys, signatures, and real staking. This PRD designs an enhanced registry that is the source of truth for the target architecture.

---

## 1. Problem

### 1.1 Current implementation

- `ModelRegistry.sol` stores `Model` with `modelHash` (a `bytes32` hash) and `modelVerificationHash`.
- `ModelGovernance.sol` stores `ActiveModel` with `hfRepo`, `hfRevision`, and `genesisCheckpoint`.
- No CID, IPFS, or content-addressed storage reference.
- No signature or public-key field for model artifact verification.
- No staking integration; `stakeBalance` is a test placeholder.

### 1.2 Attack vectors

- A malicious submitter can register a model with an arbitrary `modelHash`; the contract does not verify off-chain data.
- No public-key binding: anyone can claim to be the orchestrator for a model.
- No staking to slash bad model proposals.

### 1.3 Architectural limitations

- The blockchain cannot independently verify the integrity of a model artifact.
- Miners and gateways must trust off-chain registries for CIDs and orchestrator lists.

---

## 2. Target Architecture

The Model Registry is the single source of truth for:
- Model identity (`model_id`, `version`).
- Content address (`artifactCID`, `artifactHash`).
- Cryptographic identity (`publicKey`, `signature`).
- Lifecycle (`active`, `deprecated`, `activationBlock`, `deprecationBlock`).
- Governance (`proposals`, `votes`, `staking`, `rewards`).
- Approved orchestrators (`orchestratorAllowlist`).

---

## 3. Alternatives

### 3.1 Option A: Extend existing `ModelRegistry` and `ModelGovernance`

**Pros:**
- Backward compatible.
- Minimal deployment risk.

**Cons:**
- `Model` struct grows and may exceed EVM gas limits for reads.
- Governance and registry tightly coupled.

### 3.2 Option B: Split into `ModelRegistry`, `ModelGovernance`, `OrchestratorRegistry`

**Pros:**
- Cleaner separation of concerns.
- `OrchestratorRegistry` can be reused for gateway approval.

**Cons:**
- More contracts to deploy and audit.

### 3.3 Recommended: Option B

Use three contracts:
1. `ModelRegistry` — model metadata and lifecycle.
2. `ModelGovernance` — proposals, voting, staking.
3. `OrchestratorRegistry` — approved orchestrators and their public keys.

---

## 4. Blockchain Changes

### 4.1 `ModelRegistry.sol`

```solidity
struct Model {
    string name;
    string description;
    string version;
    bytes32 artifactHash;        // blake3 or sha256 of encrypted artifact
    string artifactCID;          // IPFS/Arweave/S3 content id
    bytes32 encryptionKeyHash;   // hash of model master key (not the key)
    address submitter;
    bytes32 publicKey;           // orchestrator secp256k1 public key hash
    bytes signature;             // submitter signature over artifactHash || artifactCID
    uint256 blockHeight;
    bool active;
    bool deprecated;
    string category;
    uint256 totalQueries;
    uint256 totalEarnings;
    uint256 minPaymentRate;
}

mapping(bytes32 => Model) public models;
mapping(string => bytes32) public modelIdByName; // name hash -> modelId
mapping(bytes32 => bytes32[]) public versionHistory; // model name hash -> versions

// Events
 event ModelRegistered(bytes32 indexed modelId, bytes32 indexed nameHash, address indexed submitter, bytes32 artifactHash, string artifactCID);
 event ModelArtifactUpdated(bytes32 indexed modelId, bytes32 artifactHash, string artifactCID, uint256 blockHeight);
 event ModelSignatureUpdated(bytes32 indexed modelId, bytes signature);
 event ModelActivated(bytes32 indexed modelId, uint256 blockHeight);
 event ModelDeprecated(bytes32 indexed modelId, uint256 blockHeight);
```

### 4.2 `ModelGovernance.sol`

```solidity
struct ActiveModel {
    string modelId;
    bytes32 modelHash;
    string artifactCID;
    bytes32 artifactHash;
    bytes32 publicKey;
    uint256 vramRequired;
    uint256 rewardPerBlock;
    uint256 minStakeToTrain;
    uint256 minOrchestratorStake;
    uint256 activationBlock;
    uint256 deprecationBlock;
    bool active;
}

mapping(address => uint256) public stakeBalance;
mapping(uint256 => ModelProposal) public proposals;

// Staking
function stake() external payable;
function unstake(uint256 amount) external;

// New validation
error InsufficientStake(uint256 required, uint256 actual);
error InvalidSignature();
error CIDTooLong();
```

### 4.3 `OrchestratorRegistry.sol` (new)

```solidity
struct Orchestrator {
    address owner;
    bytes publicKey;    // secp256k1 33-byte compressed
    string endpoint;    // URL or multiaddr
    bytes32 modelId;
    uint256 staked;
    bool approved;
}

mapping(address => mapping(bytes32 => Orchestrator)) public orchestrators;

function register(bytes32 modelId, bytes calldata publicKey, string calldata endpoint) external payable;
function approve(address orchestrator, bytes32 modelId) external onlyOwnerOrGovernance;
function slash(address orchestrator, bytes32 modelId, uint256 amount) external onlyOwnerOrGovernance;

// Events
 event OrchestratorRegistered(bytes32 indexed modelId, address indexed orchestrator, bytes publicKey);
 event OrchestratorApproved(bytes32 indexed modelId, address indexed orchestrator);
 event OrchestratorSlashed(bytes32 indexed modelId, address indexed orchestrator, uint256 amount);
```

### 4.4 Validation rules

- `registerModel` requires `msg.value >= minStakeToRegister`.
- `artifactCID` must be non-empty and ≤ 128 bytes.
- `artifactHash` must be non-zero.
- `signature` must be a valid secp256k1 signature over `keccak256(abi.encode(modelId, version, artifactHash, artifactCID))`.
- `publicKey` must match the recovered address from `signature`.
- `ModelGovernance` proposals require `stakeBalance[msg.sender] >= PROPOSAL_THRESHOLD`.
- Voting is stake-weighted.

---

## 5. Storage Changes

- The registry does not store weights; it only stores CIDs and hashes.
- Encrypted artifacts live off-chain on IPFS/Arweave/S3.
- Version history is on-chain for auditability.

---

## 6. Security Changes

- Model artifacts must be signed by the registered `publicKey`.
- Staking creates an economic deterrent against malicious model submissions.
- `OrchestratorRegistry` ensures only approved orchestrators can serve a model.
- Slashing for serving invalid artifacts or failing remote attestation.

---

## 7. Synchronization

- `ModelArtifactUpdated` events trigger orchestrators and gateways to update their caches.
- P2P gossip announces new checkpoints using `artifactHash` and `artifactCID`.
- Miners use the registry to find approved orchestrators and current artifacts.

---

## 8. Migration Plan

1. Deploy `OrchestratorRegistry`.
2. Upgrade `ModelRegistry` to v2 (or deploy new contract with v2 schema).
3. Backfill existing models with placeholder `artifactCID` and `signature` (governance vote required).
4. Update `xenom`/`seed-node` and `api-gateway` to read v2 schema.
5. Deprecate v1 registry after a transition period.

---

## 9. Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Contract upgrade bugs | High | Use OpenZeppelin UUPS or deploy v2 and migrate. |
| High gas costs | Medium | Keep structs compact; use off-chain CID for large data. |
| Malicious majority vote | Medium | Require high quorum and approval threshold; allow owner veto during bootstrap. |

---

## 10. Deliverables

This PRD is the design input for:
- `PRD-005-Encrypted-Storage.md`
- `PRD-001-Secure-Model-Distribution.md`
- `PRD-002-Orchestrator-Service.md`
