# ADR 001: P2P Model and Genome Metadata Gossip

## Status
Implemented.

## Context
`seed-node` is currently the single source of model/genome checkpoints for `xenom-miner`.
This creates a centralisation bottleneck and a single point of failure for model
 downloads (see issue #59 B2).  We need a decentralised way for nodes to discover
which peers hold a given checkpoint, without requiring a central lookup service.

The two candidate transports were:
1. **libp2p** — mature gossipsub, but would add a large new dependency graph and a
   second peer-discovery mechanism to a codebase that already runs Kaspa P2P.
2. **Kaspa P2P layer** (`kaspa-p2p-lib` / `protocol/flows`) — already used by
   `xenom-node` for block/transaction propagation.  It provides peer discovery,
   routing, broadcasts, and handshake-based identity.

Decision: reuse the Kaspa P2P layer.  It keeps the transport, handshake and
identity model consistent and avoids duplicating peer discovery.

## Decision

Introduce a small metadata-gossip sub-protocol on top of Kaspa P2P:

### 1. Announcement payload
A `CheckpointAnnouncement` carries:

```text
model_id          : String        (e.g. "multimolecule/dnabert2")
weights_hash      : [u8; 32]      (active/base checkpoint hash)
cid               : [u8; 32]      (content identifier / multihash, placeholder for IPFS/libp2p CID)
timestamp         : u64           (unix seconds, for TTL and replay protection)
node_address      : String        (xenom/kaspa address of the announcing node)
public_key        : [u8; 33]      (secp256k1 compressed public key of the signer)
listen_addr       : NetAddress    (optional P2P/WS address where the checkpoint can be fetched)
is_genome         : bool          (true for genome archives)
signature         : [u8; 64]      (secp256k1 signature over all preceding fields)
```

`genome` archives use the same message with `model_id` replaced by the genome
 merkle root and `weights_hash` by the archive hash.  The same `genome` flag
distinguishes the two kinds of announcement.

### 2. Transport messages
Two new protobuf payloads are added to `protocol/p2p/proto/messages.proto`
inside the `KaspadMessage` oneof:

- `CheckpointAnnouncementMessage` — a signed announcement.
- `RequestCheckpointMessage`       — asks peers for announcements for a given
  `model_id` / `weights_hash`.

Corresponding `KaspadMessagePayloadType` variants are added in
`protocol/p2p/src/core/payload_type.rs`, and conversion code in
`rpc/service/src/converter/protocol.rs` (or `protocol/p2p/src/convert`) is
updated.

### 3. Flows
A single `ModelGossipFlow` lives under `protocol/flows/src/v5/model_gossip.rs`
and is registered in both `protocol/flows/src/v5/mod.rs` and
`protocol/flows/src/v6/mod.rs`:

- subscribes to `CheckpointAnnouncementMessage` and `RequestCheckpointMessage`;
- validates signatures and inserts valid announcements into the local
  `GossipRegistry`;
- answers checkpoint requests with known announcements from the registry;
- exposes `FlowContext::announce_checkpoint` for broadcasting locally-signed
  announcements.

`seed-node` runs a client-only P2P gossip peer in `seed-node/src/p2p.rs`
(`P2pGossipHandle`) that performs the same handshake, receives/verifies
announcements, answers requests, and broadcasts its own checkpoint metadata.

### 4. Registry
A `GossipRegistry` is shared through `FlowContext`:

```rust
pub struct GossipRegistry {
    entries: HashMap<[u8; 32], Vec<Announcement>>,
    ttl: Duration,
}
```

- Lookup keyed by `weights_hash` (or CID).
- Duplicate and expired announcements are dropped.
- Signature verification uses the existing Kaspa/Xenom secp256k1 helpers and the
  address contained in the announcement.

### 5. Producers and consumers
- `seed-node`: signs and broadcasts an announcement whenever a model is loaded
  or a new FedAvg checkpoint becomes active.
- `xenom-node`: forwards/re-broadcasts announcements and maintains a registry.
- `xenom-miner`: discovers peers from the registry and later uses them to
  download checkpoint bytes (issue #59 B2).

## Consequences

- **Positive**: removes the seed-node bottleneck for checkpoint discovery;
  keeps transport and identity consistent with the rest of the Kaspa-derived
  stack; easy to extend to genome archives later.
- **Positive**: authentication comes from reusing the existing wallet/node
  identity — no new PKI is needed.
- **Negative**: `xenom-miner` must either run a minimal P2P client or ask a
  local `xenom-node`/`seed-node` for peer lists.  This is addressed in slice 4.
- **Negative**: adding protobuf payload types touches the conversion/routing
  surface (`payload_type.rs`, `converter/protocol.rs`) and both v5/v6 flow
  registries.  Changes must keep binary compatibility with older nodes.

## Implementation slices
1. Messages — protobuf definitions, payload types and conversion. Done.
2. Registry and signing — `Announcement`, `GossipRegistry`, `GossipIdentity`,
   validation. Done.
3. Flows and broadcast — `ModelGossipFlow` and `FlowContext` helpers on
   `xenom-node`. Done.
4. Seed-node P2P gossip — `P2pGossipHandle` joins the network and announces
   active checkpoints. Done.
5. Miner discovery — `xenom-miner` queries the gossip registry (issue #59 B2).

## References
- Issue #58 B1: P2P model and genome metadata gossip
- Issue #59 B2: Direct P2P transfer of model checkpoints
- `protocol/p2p/proto/messages.proto`
- `protocol/p2p/src/core/payload_type.rs`
- `protocol/flows/src/v5/model_gossip.rs`
- `seed-node/src/p2p.rs`
- `crypto/model-crypto/src/gossip.rs`
- `seed-node/src/model/manager.rs`
