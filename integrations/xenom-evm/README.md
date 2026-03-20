# Xenom EVM L2

An Ethereum-compatible JSON-RPC node built on [revm](https://github.com/bluealloy/revm), running as an L2 on top of the Xenom (Kaspa-fork) network.

## Architecture

```
integrations/xenom-evm/
├── xenom-evm-core/   # EvmChain state machine (revm executor, mempool, log indexing, anchor store)
├── xenom-evm-rpc/    # jsonrpsee JSON-RPC + WebSocket server (30+ eth_* methods)
└── xenom-evm-node/   # Binary: CLI flags, block sequencer, DAA-tied mining
```

---

## Quick Start

### Build

```bash
cargo build -p xenom-evm-node --release
```

### Run (devnet)

```bash
./target/release/xenom-evm-node \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --block-time 2000        # mine a block every 2 s
```

The node will:
- Pre-fund the canonical devnet address with **10 000 ETH**
- Mine blocks at the specified interval
- Serve JSON-RPC over HTTP **and** WebSocket on the same port

### With state persistence

```bash
./target/release/xenom-evm-node \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --state-dir /var/lib/xenom-evm
```

State is saved atomically as `evm-state-<chain_id>.json` after every block.

### Tied to L1 DAA score

```bash
./target/release/xenom-evm-node \
  --devnet \
  --rpc-addr 127.0.0.1:8545 \
  --l1-node grpc://127.0.0.1:36669   # mines one EVM block per L1 DAA increment
```

---

## CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--devnet` | — | Enable devnet mode (chain ID 1337, pre-fund devnet address) |
| `--chain-id <N>` | `1337` | EVM chain ID |
| `--rpc-addr <HOST:PORT>` | `127.0.0.1:8545` | JSON-RPC listen address |
| `--block-time <MS>` | `2000` | Fixed-interval block time in milliseconds |
| `--state-dir <PATH>` | — | Directory for atomic JSON state snapshots |
| `--l1-node <URL>` | — | Xenom node gRPC URL; mines one EVM block per L1 DAA tick |

---

## MetaMask Setup

### 1 — Add Network

In MetaMask → **Settings → Networks → Add a network manually**:

| Field | Value |
|-------|-------|
| Network Name | Xenom Devnet |
| New RPC URL | `http://127.0.0.1:8545` |
| Chain ID | `1337` |
| Currency Symbol | `ETH` |
| Block Explorer URL | *(leave blank)* |

### 2 — Import devnet account

In MetaMask → **Import Account** → paste the private key:

```
0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80
```

This is the Hardhat/Foundry devnet key for address `0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266`, pre-funded with 10 000 ETH.

> **Note**: This key is public knowledge. Never use it on mainnet.

### 3 — Verify

You should see `10000 ETH` in MetaMask. You can now:
- Send ETH between accounts
- Deploy contracts via Remix (set environment to **Injected Provider - MetaMask**)
- Interact with deployed contracts

---

## JSON-RPC Methods

### Standard Ethereum

| Method | Notes |
|--------|-------|
| `eth_chainId` | Returns `0x539` (1337) |
| `eth_blockNumber` | Current head block |
| `eth_getBalance` | Balance at latest/specific block |
| `eth_getCode` | Contract bytecode |
| `eth_getTransactionCount` | Nonce |
| `eth_sendRawTransaction` | Enqueue signed EIP-155/2930/1559 tx |
| `eth_getTransactionReceipt` | Full receipt including logs |
| `eth_call` | Read-only EVM call |
| `eth_estimateGas` | Simulates execution, returns gas estimate |
| `eth_gasPrice` | Returns `0x3b9aca00` (1 Gwei) |
| `eth_maxPriorityFeePerGas` | Returns `0x3b9aca00` (1 Gwei) |
| `eth_feeHistory` | EIP-1559 fee history |
| `eth_getBlockByNumber` | Block by number or tag |
| `eth_getBlockByHash` | Block by hash |
| `eth_getTransactionByHash` | Transaction object |
| `eth_getLogs` | Filter logs by block range, address, topic |
| `eth_getStorageAt` | Raw storage slot value |
| `eth_syncing` | Always `false` |
| `net_version` | Chain ID as decimal string |
| `web3_clientVersion` | `"xenom-evm/0.1.0"` |
| `eth_newFilter` | Stub (returns `0x1`) |
| `eth_uninstallFilter` | Stub (returns `true`) |

### Xenom-specific

| Method | Params | Returns | Description |
|--------|--------|---------|-------------|
| `xenom_latestStateRoot` | — | `"0x..."` | Keccak state root after last block |
| `xenom_pendingCount` | — | `N` | Pending transactions in mempool |
| `eth_faucet` | `address` | `"ok"` | Devnet faucet: send 10 ETH |
| `xenom_anchor` | `payload_hex` | `anchor_id` | Store arbitrary hex payload on-chain; returns `keccak256(data)` as anchor ID |
| `xenom_getAnchor` | `anchor_id` | `{anchorId, blockNumber, payloadHex, payloadLen}` | Retrieve anchored payload by ID |

---

## WebSocket Subscriptions (`eth_subscribe`)

Connect to `ws://127.0.0.1:8545` and send:

```json
{"jsonrpc":"2.0","method":"eth_subscribe","params":["newHeads"],"id":1}
```

You will receive a subscription ID, then notifications for every mined block:

```json
{
  "jsonrpc": "2.0",
  "method": "eth_subscribe",
  "params": {
    "subscription": 1234567890,
    "result": {
      "number": "0x1a",
      "hash": "0xabc...",
      "parentHash": "0xdef...",
      "timestamp": "0x...",
      "gasUsed": "0x5208",
      "gasLimit": "0x1c9c380"
    }
  }
}
```

### Subscribe to logs (ERC-20 Transfer events)

```json
{
  "jsonrpc": "2.0",
  "method": "eth_subscribe",
  "params": [
    "logs",
    {
      "address": "0xYourTokenAddress",
      "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
    }
  ],
  "id": 2
}
```

### Unsubscribe

```json
{"jsonrpc":"2.0","method":"eth_unsubscribe","params":["<sub_id>"],"id":3}
```

---

## eth_getLogs — Filter Reference

```json
{
  "jsonrpc": "2.0",
  "method": "eth_getLogs",
  "params": [{
    "fromBlock": "0x1",
    "toBlock": "latest",
    "address": "0xYourContract",
    "topics": ["0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"]
  }],
  "id": 1
}
```

- `fromBlock` / `toBlock`: hex number or `"latest"`
- `address`: optional contract filter
- `topics[0]`: optional event signature filter

---

## Genetics-L2 Integration

The `genetics-l2-settlement` daemon can anchor validated job settlement payloads on the EVM L2 when running on devnet/testnet.

### How it works

1. Settlement daemon polls the coordinator for `validated` jobs
2. Builds a `SettlementPayload` (job ID, results Merkle root, winner, scores, hashes)
3. Calls `xenom_anchor(payload_hex)` on the EVM node
4. Receives a `keccak256`-based anchor ID (immutable, stored in EVM state)
5. Registers the anchor ID as `txid` in the coordinator's payout record

### Start the settlement daemon with EVM anchoring

```bash
# Terminal 1: start EVM node
./target/release/xenom-evm-node --devnet --rpc-addr 127.0.0.1:8545

# Terminal 2: start settlement daemon
./target/release/genetics-l2-settlement \
  --devnet \
  --coordinator http://localhost:8091 \
  --submit \
  --evm-node http://127.0.0.1:8545 \
  --poll-ms 10000
```

### Verify an anchor

```bash
curl -s -X POST http://127.0.0.1:8545 \
  -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","method":"xenom_getAnchor","params":["0x<anchor_id>"],"id":1}' \
  | jq .
```

Response:
```json
{
  "result": {
    "anchorId": "0xabc...",
    "blockNumber": "0x5",
    "payloadHex": "0x...",
    "payloadLen": 312
  }
}
```

### Mainnet path

On mainnet the settlement hash is committed via the stratum bridge `get_block_template` `extra_data` field (coinbase payload), providing permanent L1 anchoring without EVM.

---

## CORS

All origins, methods and headers are allowed (`Access-Control-Allow-Origin: *`). MetaMask browser extension can connect directly without a proxy.

---

## Supported Transaction Types

| Type | EIP | Description |
|------|-----|-------------|
| `0x0` | Legacy | Pre-EIP-155 (no replay protection) |
| `0x0` | EIP-155 | Legacy with chain ID replay protection |
| `0x1` | EIP-2930 | Access list transactions |
| `0x2` | EIP-1559 | Fee market transactions (maxFeePerGas / maxPriorityFeePerGas) |

---

## Development Notes

- **State**: `InMemoryDB` (revm) — all state is in-memory, optionally snapshotted to JSON
- **Chain ID**: 1337 devnet / configurable via `--chain-id`
- **Gas price**: fixed 1 Gwei (`0x3b9aca00`)
- **Block gas limit**: 30 000 000 (`0x1c9c380`)
- **Consensus**: none (centralized sequencer); DAA-tied mode follows L1 block cadence
- **L1 state root anchoring**: EVM `xenom_latestStateRoot` → stratum bridge `extra_data`
