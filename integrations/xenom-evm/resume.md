# Xenom EVM L2 — Step-by-Step Explanation (English)

## 1. What this README is describing

This README describes an **Ethereum-compatible node** running **on top of Xenom**.

So:

* **Xenom L1** = the base blockchain, a Kaspa fork
* **Xenom EVM L2** = an upper execution layer that behaves like Ethereum

In simple terms:

* **L1** = base chain / foundation / final anchor layer
* **L2** = EVM execution layer / Ethereum-style interface

---

## 2. What L1 means here

In this README, **L1** is the **Xenom network itself**.

That base layer handles things like:

* peer-to-peer networking
* blocks / DAG / consensus inherited from the fork
* DAA score
* base mining
* stronger permanence when something is anchored there

So when it says:

> running as an L2 on top of the Xenom (Kaspa-fork) network

it means the EVM node does **not replace** the main Xenom chain. It sits **above it**.

---

## 3. What L2 means here

The **L2** is the **Xenom EVM node**.

It provides:

* Ethereum compatibility
* `eth_*` methods
* MetaMask support
* contract deployment
* EVM execution through `revm` 
* an EVM mempool
* logs and receipts
* JSON-RPC and WebSocket APIs

To a Web3 developer, this looks like an Ethereum node, but it belongs to the Xenom ecosystem.

---

## 4. Architecture explained

### `xenom-evm-core` 

This is the core execution engine:

* EVM state machine
* `revm` executor
* mempool
* log indexing
* anchor store

### `xenom-evm-rpc` 

This is the API layer:

* HTTP JSON-RPC
* WebSocket
* `eth_*` methods
* `xenom_*` methods

### `xenom-evm-node` 

This is the runnable binary:

* CLI flags
* block sequencing
* block production
* DAA-tied mode

---

## 5. Build

```bash
cargo build -p xenom-evm-node --release
```

This builds the **L2 EVM node binary**.

It is not the full L1 chain here; it is specifically the EVM node.

---

## 6. Run in devnet

```bash
./target/release/xenom-evm-node \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --block-time 2000
```

This starts the node in devnet mode and:

* sets chain ID `1337` 
* exposes RPC on `127.0.0.1:8545` 
* mines one EVM block every 2 seconds

It also:

* pre-funds the canonical devnet address with **10,000 ETH**
* serves HTTP and WebSocket on the same port

That "ETH" is the L2 accounting unit, not Ethereum mainnet funds.

---

## 7. State persistence

With:

```bash
--state-dir /var/lib/xenom-evm
```

the node writes state snapshots as JSON after every block.

Without it, state lives only in memory.

So this L2 currently uses:

* in-memory execution
* optional JSON snapshot persistence

---

## 8. Tied to L1 DAA score

```bash
--l1-node grpc://127.0.0.1:36669
```

This is one of the most important parts.

The README says:

> mines one EVM block per L1 DAA increment

That means:

* whenever the **Xenom L1** advances in **DAA score**
* the **EVM L2** produces one block

So the L2 follows the rhythm of the L1.

This creates a real relationship between:

* L1 progression
* L2 block production

That is an important anchoring/cadence mechanism.

---

## 9. CLI flags

The key flags are:

* `--devnet` → test/dev mode
* `--chain-id` → Ethereum chain identity
* `--rpc-addr` → RPC listen address
* `--block-time` → fixed block interval
* `--state-dir` → persistence
* `--l1-node` → connect L2 to Xenom L1

`--l1-node` is especially important because it is the clearest explicit L1/L2 bridge in operation.

---

## 10. MetaMask support

The README shows that you can add this network to MetaMask.

That means the goal is to make the L2 feel like a normal Ethereum-like network.

You can:

* import an account
* see balance
* send ETH
* deploy contracts
* use Remix
* use Ethereum-compatible dapps

---

## 11. JSON-RPC methods

There are two families.

### Standard Ethereum methods

Examples:

* `eth_chainId` 
* `eth_blockNumber` 
* `eth_getBalance` 
* `eth_sendRawTransaction` 
* `eth_call` 
* `eth_estimateGas` 
* `eth_getLogs` 

These make the node compatible with standard Ethereum tooling.

### Xenom-specific methods

Examples:

* `xenom_latestStateRoot` 
* `xenom_pendingCount` 
* `eth_faucet` 
* `xenom_anchor` 
* `xenom_getAnchor` 

These are where the L2 begins to expose Xenom-specific settlement/anchoring capabilities.

---

## 12. What `xenom_anchor` does

```text
xenom_anchor(payload_hex) -> anchor_id
```

It stores arbitrary hex payload data on the EVM L2 and returns a `keccak256`-based ID.

Practical flow:

1. prepare data
2. encode as hex
3. call `xenom_anchor` 
4. payload is stored
5. receive `anchor_id` 

That ID becomes an immutable content reference.

This is useful for anchoring:

* proofs
* settlement data
* result hashes
* structured records

---

## 13. WebSocket subscriptions

This is standard Ethereum-style live event support:

* subscribe to new blocks
* subscribe to logs

Useful for:

* frontends
* explorers
* indexers
* real-time monitors

---

## 14. `eth_getLogs` 

This lets clients filter contract events such as:

* ERC-20 transfers
* approvals
* custom application events

It is essential for practical Ethereum compatibility.

---

## 15. Most important part: Genetics-L2 integration

Now we get to the connection between genetics jobs and the EVM L2.

The README says:

> The `genetics-l2-settlement` daemon can anchor validated job settlement payloads on the EVM L2

This means there is a separate daemon called:

* `genetics-l2-settlement` 

Its job is to take validated genetics job results and anchor them on the EVM L2.

---

## 16. The Genetics-L2 flow

The README gives 5 steps. Expanded:

### Step 1 — poll validated jobs

The settlement daemon polls the coordinator for jobs marked `validated`.

That means some earlier pipeline already performed:

* job submission
* computation
* validation

### Step 2 — build `SettlementPayload` 

It creates a settlement payload containing things like:

* job ID
* results Merkle root
* winner
* scores
* hashes

This is a compact, verifiable summary of the result.

### Step 3 — call `xenom_anchor(payload_hex)` 

Instead of writing directly to L1 at this stage, the daemon anchors the payload in the **EVM L2**.

### Step 4 — get an `anchor_id` 

The EVM L2 returns a `keccak256`-based immutable identifier.

### Step 5 — register that ID as `txid` 

The coordinator stores that anchor ID in the payout record.

So the payout system now has a verifiable reference for settlement.

---

## 17. What that means architecturally

Simple summary:

* genetics jobs produce results
* the coordinator validates them
* the settlement daemon builds a verifiable payload
* the EVM L2 stores/anchors it
* the system uses the `anchor_id` as the reference

So the L2 acts like a practical settlement and anchoring layer.

---

## 18. How that relates to L1

This is the key distinction.

The README shows two paths:

### Path A — devnet/testnet

Settlement payloads are anchored on the **EVM L2**.

### Path B — mainnet

The settlement hash is committed directly to **L1** via:

* the stratum bridge
* `get_block_template` 
* the `extra_data` field
* coinbase payload

So:

* in dev/test environments, the L2 is used as a convenient settlement layer
* in mainnet, the final permanent commitment goes directly to L1

---

## 19. Mainnet path explained

The README says:

> providing permanent L1 anchoring without EVM

That means:

* production-grade final commitment does not require the EVM layer
* the settlement hash is embedded directly into L1 block-related data
* the base chain becomes the final anchor of truth

This strongly suggests:

* L2 is for execution, integration, APIs, workflow convenience
* L1 is for strongest permanence and final anchoring

---

## 20. What "Genetics-L2" likely means

From the README, "Genetics-L2" does **not** look like a separate standalone blockchain.

It looks more like:

* a settlement subsystem
* a daemon/pipeline
* using the EVM L2 as its anchoring layer

So in practical architecture:

* **Xenom L1** = base chain
* **Xenom EVM L2** = Ethereum-compatible execution layer
* **Genetics-L2 settlement** = genetics settlement logic using that L2

---

## 21. L2 consensus model

The README explicitly says:

> Consensus: none (centralized sequencer)

This is very important.

It means the L2 does not currently have its own decentralized consensus.

Instead:

* one sequencer orders transactions
* one sequencer produces L2 blocks

So this L2 is operationally useful, but its trust/security model is not the same as a fully decentralized L1.

That is another reason why L1 anchoring matters so much.

---

## 22. L1 state root anchoring

At the end, the README mentions:

> EVM `xenom_latestStateRoot` → stratum bridge `extra_data` 

This suggests the L2 state root may be committed to L1 through the bridge.

If fully implemented, the model is roughly:

1. L2 executes transactions
2. L2 produces a new state root
3. that state root is committed into L1-related data
4. L1 becomes the anchor layer for L2 state

That is a recognizable anchored-L2 design pattern.

---

## 23. One-line conceptual summary

The **Xenom L1** is the base blockchain, the **Xenom EVM L2** is an Ethereum-compatible execution layer on top of it, and the **Genetics-L2 settlement flow** uses that L2 to anchor validated job results, while mainnet final anchoring goes directly to L1.

---

## 24. Full simplified flow

### Normal EVM flow

1. start `xenom-evm-node` 
2. MetaMask connects to RPC
3. users send transactions
4. L2 executes them through `revm` 
5. L2 produces blocks
6. state/logs/receipts are stored

### L1-coupled flow

1. L2 connects to Xenom L1
2. L2 watches DAA score increments
3. L2 mines one EVM block per L1 increment
4. L2 timing is coupled to L1

### Genetics flow

1. coordinator validates jobs
2. settlement daemon builds payload
3. payload is anchored on L2 via `xenom_anchor` 
4. daemon gets `anchor_id` 
5. payout record stores that anchor ID
6. on mainnet, final commitment can go directly to L1

---

## 25. Technical reading of the design

The most likely reading of this README is:

* the project wants a developer-friendly **EVM layer**
* it also wants a way to **record settlement for genetics jobs**
* in testing environments, it uses the **EVM L2** for that
* in production, the final source of truth should live on **Xenom L1**

That is consistent with a system where:

* L2 is for usability, integration, and execution
* L1 is for stronger immutability and final commitment
