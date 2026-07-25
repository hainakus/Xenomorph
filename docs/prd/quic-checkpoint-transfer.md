# PRD: QUIC Direct Transfer for Model Checkpoints

## Status

Draft — ready for review.

## Problem Statement

Model checkpoints for `multimolecule/dnabert2` and similar models are hundreds of megabytes (≈ 400 MB for the full DNABERT-2 weights). Today these checkpoints travel over the same WebSocket/Borsh RPC channel that carries tiny control messages: `GetTrainingBatch`, `SubmitGradients`, `SubmitBlock`, and heartbeat traffic.

This causes three concrete problems:

1. **Slow startup / re-sync.** A miner that needs the latest checkpoint waits for the full payload inside a single WebSocket binary message. Compression and framing overhead, plus head-of-line blocking behind control traffic, make this slower than a dedicated bulk transport.
2. **Head-of-line blocking.** A large checkpoint transfer delays small, latency-sensitive RPC messages on the same connection.
3. **Seed-node bottleneck.** Every checkpoint byte passes through the seed-node / unified `xenom` node. When many miners reconnect at once, the single WebSocket path becomes a bandwidth and CPU bottleneck.

The P2P metadata gossip subsystem (`ADR 001`) already lets nodes announce which checkpoints they hold, including an optional `listen_addr`. The missing piece is an efficient **data plane** that uses those announced endpoints to move the actual checkpoint bytes.

## Solution

Add a dedicated **QUIC bulk transfer layer** for checkpoint files. The WebSocket/Borsh RPC path keeps control-plane responsibilities (batches, gradients, blocks, metadata). QUIC handles only the data plane: `config.json`, `tokenizer.json`, and `weights.safetensors` bytes.

This is **Phase 1** of the P2P transfer roadmap: QUIC direct, with peer addresses discovered through existing gossip or a new lightweight RPC query. **Phase 2** (libp2p with DHT and NAT traversal) is explicitly out of scope for this PRD.

Target sync times for a 4 MB checkpoint bundle:

| Scenario | Target sync time | Simultaneous peers |
|---|---|---|
| LAN (1 Gbps) | 50–100 ms | 100+ |
| WAN optimal (100 Mbps) | 400–800 ms | 50+ |
| WAN average (20 Mbps) | 1.5–2 s | 20+ |
| Mobile (4G) | 3–5 s | 10 |

## User Stories

1. As a miner on a 100 Mbps WAN link, I want to download a 400 MB checkpoint in under one second, so that I can start training sooner.
2. As a miner on a mobile connection, I want checkpoint downloads to tolerate variable latency and packet loss, so that I can still participate even on flaky networks.
3. As a miner behind a restrictive firewall, I want the miner to fall back to the existing WebSocket RPC path if QUIC is blocked, so that I never lose the ability to sync.
4. As a seed-node operator, I want bulk transfers to happen on a separate port and protocol, so that the WebSocket RPC path stays responsive for gradient submission and block propagation.
5. As a seed-node operator, I want to limit the number of concurrent QUIC transfers, so that a small seed-node is not overwhelmed by many reconnecting miners.
6. As a network operator, I want checkpoint transfers to be encrypted and authenticated, so that an attacker cannot tamper with model weights in transit.
7. As a miner, I want to verify the hash of every file received over QUIC before using it, so that I never train on a corrupted or malicious checkpoint.
8. As a miner with the base weights already cached, I want to download only the LoRA adapter over QUIC, so that I save bandwidth and time.
9. As a node running the unified `xenom` binary, I want to announce my QUIC transfer endpoint inside checkpoint gossip announcements, so that peers can discover and connect to me directly.
10. As a miner without a direct P2P gossip connection, I want to ask my seed-node via RPC for a list of peers that hold a given checkpoint, so that I can then fetch from one of them over QUIC.
11. As a miner, I want the client to try multiple announced peers and pick the fastest reachable one, so that I am not stuck waiting for a slow or unreachable seed.
12. As a devops engineer, I want metrics on QUIC transfer speed, success rate, and fallback events, so that I can monitor the health of the bulk transfer network.
13. As a node operator, I want the QUIC endpoint to bind to a configurable address and to support external IP/port overrides (e.g. for Docker/NAT), so that the announced address matches what miners can reach.
14. As a developer, I want the QUIC transfer code to be testable without a real network, so that I can write fast unit and integration tests for the protocol.

## Implementation Decisions

### Transport and dependencies

- Use the **`quinn`** crate for QUIC. It is a mature, tokio-native Rust implementation of QUIC that supports both client and server without the heavy dependency graph of libp2p.
- Do **not** introduce `libp2p` in this PRD. libp2p remains the target for Phase 2 (automated discovery, DHT, multi-hop relays).
- QUIC runs over UDP. If UDP is blocked, the miner transparently falls back to the existing WebSocket/TCP `get_model_checkpoint_v2` RPC.

### Security model

- QUIC itself is always TLS encrypted. Use **self-signed X.509 certificates** generated locally on first startup (or derived from the node's gossip secp256k1 identity if feasible).
- Authentication is **out-of-band via the P2P gossip announcement**. The `Announcement` already contains a `public_key` and a secp256k1 signature over `model_id || weights_hash || cid || timestamp || is_genome || node_address || public_key || listen_addr`. The miner trusts the `listen_addr` only because it is signed by an announcement it has already verified. The certificate can be ignored or pinned to the announced public key.
- Hash verification of every downloaded file is mandatory. The miner knows the expected `weights_hash` and `base_hash` from `get_model_checkpoint_info_v2` before it fetches anything.

### Control plane vs data plane

- **WebSocket/Borsh RPC remains the control plane.** It handles:
  - `GetTrainingBatch` / `GetGenomeTrainingBatch`
  - `SubmitGradients`
  - `SubmitBlock`
  - `GetModelCheckpointInfoV2`
  - `GetModelCheckpointV2` (fallback)
  - `GetCheckpointPeers` (new — see below)
  - heartbeats and balance/difficulty queries
- **QUIC is the data plane only.** It transfers `config`, `tokenizer`, and `weights` bytes. It does not carry RPC request/response semantics.

### Protocol on top of QUIC

- A single QUIC connection carries one or more **unidirectional streams** (or bidirectional streams, TBD during detailed design). Each stream carries one file transfer.
- Each request is a small Borsh-serialized frame:
  ```
  CheckpointFileRequest {
      model_id: String,
      weights_hash: [u8; 32],
      file_type: CheckpointFileType, // Config, Tokenizer, Weights, Adapter
  }
  ```
- The response is a length-prefixed byte stream. A tiny header indicates status (`Ok`, `NotFound`, `NotAuthorized`) and total length, followed by the raw bytes. The client reads exactly `length` bytes, then hashes and verifies them.
- Files can be served **encrypted on disk** (`config.enc`, `tokenizer.enc`, `weights.enc`). The server reads those bytes directly and sends them as-is; the miner decrypts with the same `XENO_MODEL_KEY` used today. This avoids double encryption/decryption on the server.

### New RPC: `GetCheckpointPeers`

- Add `RpcRequest::GetCheckpointPeers { model_id, weights_hash }` and `RpcResponse::CheckpointPeers(Vec<Announcement>)`.
- The unified `xenom` node and the standalone `seed-node` implement this by querying the local `GossipRegistry`.
- This lets a miner that does **not** run its own P2P gossip client discover peers that have the checkpoint, simply by asking its trusted seed-node.

### Server modules

- Introduce a `CheckpointTransferServer` component:
  - Binds to `XENO_QUIC_LISTEN` / `--quic-listen` (default port TBD, e.g. 17111).
  - Accepts incoming QUIC connections.
  - Authenticates the requested `model_id` + `weights_hash` against locally stored checkpoints via `ModelStorage`.
  - Streams the requested encrypted file bytes.
  - Enforces a configurable maximum number of concurrent transfer streams.
- The unified `xenom` node starts the server alongside the miner WebSocket server.
- The standalone `seed-node` starts its own instance.

### Client modules

- Introduce a `CheckpointTransferClient` component in `xenom-miner`:
  - Takes one or more `Announcement`s (or explicit `SocketAddr`s + expected hashes).
  - Opens a QUIC connection, sends `CheckpointFileRequest`, reads the response.
  - Verifies hashes, decrypts files, assembles a `ModelBundle`.
  - Falls back to the existing `XenomRpcClient::get_model_checkpoint_v2` on any failure.
- Integrate the client into `model_client::fetch_model_checkpoint`:
  1. Call `get_model_checkpoint_info_v2` to obtain `base_checkpoint` and `base_hash`.
  2. If base is cached, request adapter-only.
  3. Try QUIC peers for the missing file(s).
  4. On QUIC failure, fall back to WebSocket `get_model_checkpoint_v2`.

### Announcements and `listen_addr`

- `FlowContext::announce_checkpoint` and `P2pGossipHandle::announce` already accept an optional `listen_addr` argument. Populate this argument with the QUIC server endpoint when the server is enabled.
- If `XENO_QUIC_LISTEN` is `0.0.0.0:PORT` and `XENO_EXTERNAL_IP` is set, the announced address must use the external IP.
- If QUIC is disabled, `listen_addr` remains `None` and behavior is unchanged.

### LoRA adapter support

- The client requests `CheckpointFileType::Adapter` when it already has the matching base weights cached.
- The server must be able to produce the current adapter bytes (already cached in `CachedCheckpoint::adapter_bytes` or serialized on demand) and send them as a single file.
- The client merges the received adapter with its local base, exactly as it does today for the WebSocket path.

### Concurrency and backpressure

- Server uses a `tokio::sync::Semaphore` to cap concurrent transfers. Default cap TBD (e.g. 64).
- Client opens one QUIC connection per peer and may run multiple file requests in parallel over that connection, or open multiple peer connections and race them.
- Timeouts: connect timeout (e.g. 5 s), per-file timeout (e.g. 60 s + bandwidth-dependent).

### Configuration and defaults

| Component | Flag / env var | Default | Meaning |
|---|---|---|---|
| `xenom` / `seed-node` | `--quic-listen` / `XENO_QUIC_LISTEN` | `0.0.0.0:17111` | QUIC bulk transfer bind address |
| `xenom` / `seed-node` | `--quic-external` / `XENO_QUIC_EXTERNAL` | same as bound | Address announced to peers |
| `xenom` / `seed-node` | `--quic-max-transfers` / `XENO_QUIC_MAX_TRANSFERS` | 64 | Max concurrent transfer streams |
| `xenom-miner` | `--quic` / `XENO_QUIC_ENABLED` | true | Enable QUIC downloads |
| `xenom-miner` | `--quic-timeout` / `XENO_QUIC_TIMEOUT` | 60 s | Per-file transfer timeout |
| `xenom-miner` | `--quic-peers` | 3 | Number of peers to try in parallel |

## Testing Decisions

### What makes a good test

Tests should exercise the **external behaviour** of the transfer layer: a client asks for a file, the server returns it, and the client verifies the content/hashes. They should not depend on internal framing details or on real network conditions.

### Unit tests

- `CheckpointTransferClient` + `CheckpointTransferServer` roundtrip with a 1 KB, 1 MB, and 4 MB synthetic file over `localhost`/`127.0.0.1`.
- Hash verification rejects a file whose bytes are tampered mid-transfer.
- Server returns `NotFound` for an unknown `weights_hash`.
- Concurrent transfers are correctly capped by the server's semaphore.
- Client fallback: when the QUIC server is unreachable, `fetch_model_checkpoint` falls back to the existing WebSocket path.

### Integration tests

- End-to-end test in the existing `tests/integration` suite:
  - start an Anvil devnet + seed-node + miner;
  - trigger a checkpoint update;
  - have the miner fetch the new checkpoint via QUIC and verify it trains correctly.
- QUIC-only benchmark test (not run in CI by default):
  - measure transfer time for 4 MB / 100 MB / 400 MB payloads;
  - assert 4 MB completes in < 200 ms on `localhost`.

### Prior art

- `seed-node/src/p2p.rs` already has peer-to-peer announcement/signing/verification tests (`crypto/model-crypto/src/gossip.rs`).
- `xenom-miner/src/model_client.rs` has adapter/base merging tests.
- `xenom-miner/src/rpc/client.rs` has WebSocket reconnection tests.
- `tests/integration` already spins up isolated devnets and mock WebSocket seed nodes.

## Out of Scope

- **libp2p integration** — this is Phase 2, reserved for when automated discovery, DHT-based content routing, or censorship resistance become requirements.
- **DHT / content routing / NAT traversal relays** — QUIC direct assumes the announced `listen_addr` is reachable; if it is not, the client falls back to WebSocket.
- **HTTP/3** — the protocol on top of QUIC is a custom, simple Borsh-framed request/response, not HTTP.
- **Partial/ranged/resumable downloads** — each file is transferred in one stream. If a transfer fails, the client retries the whole file from another peer.
- **Uploading checkpoints from miner to peer** — miners only download. Upload/announce flows remain through the existing RPC + P2P gossip path.

## Further Notes

- The `cid` field in `Announcement` is currently a placeholder equal to `weights_hash`. Until IPFS/libp2p CIDs are wired, the QUIC protocol uses `model_id` + `weights_hash` as the file identifier.
- UDP-based QUIC may be blocked by some corporate firewalls or mobile carriers. The WebSocket fallback must remain fully functional and be the default code path when QUIC is disabled or unreachable.
- Certificate management should be transparent to operators: generate a self-signed cert on first startup, persist it next to the gossip key, and rotate it only when the key changes. Client trust is derived from the already-signed gossip announcement, not from a CA.
- Consider adding a small header to the QUIC response that advertises the file size before the bytes, so the client can show progress and validate the expected size.
