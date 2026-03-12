# Xenom Stratum Bridge

Bridges Stratum v1 mining software to a Xenom (Xenomorph) node via gRPC.  
Implements the **Xenom Genome PoW Stratum extension** so that any miner that speaks Stratum can connect and mine the Xenom network.

---

## Quick start

```bash
# From the workspace root
cargo build --release -p xenom-stratum-bridge

./target/release/xenom-stratum-bridge \
    --rpcserver  localhost:36669 \
    --listen     0.0.0.0:1444 \
    --mining-address xenom:your_reward_address_here
```

Point your miner at `stratum+tcp://<bridge-host>:1444`.

---

## CLI options

### Connectivity

| Flag | Default | Description |
|---|---|---|
| `--rpcserver` | `localhost:36669` | Xenom node gRPC endpoint |
| `--listen` | `0.0.0.0:1444` | Stratum TCP listen address |
| `--mining-address` | *(required)* | Pool coinbase reward address |
| `--poll-interval-ms` | `200` | Block-template poll interval (ms) |

### Variable Difficulty (VarDiff)

| Flag | Default | Description |
|---|---|---|
| `--vardiff-initial` | `1` | Starting share difficulty per miner |
| `--vardiff-min` | `0.1` | Minimum share difficulty |
| `--vardiff-max` | `1000000` | Maximum share difficulty |
| `--vardiff-target-spm` | `20` | Target shares per minute per miner |
| `--vardiff-retarget-secs` | `60` | How often to retarget difficulty |

### PPLNS Accounting

| Flag | Default | Description |
|---|---|---|
| `--pplns-window` | `10000` | Share window size for payout proportions |
| `--payout-file` | *(none)* | JSON file to persist payout records |
| `--stats-interval-secs` | `300` | Log pool stats every N seconds (`0` = off) |

### Auto-Payout (requires node `--utxoindex`)

| Flag | Default | Description |
|---|---|---|
| `--pool-private-key` | *(none)* | Pool wallet private key (32 bytes hex). Enables auto-payouts. |
| `--confirm-depth` | `1000` | DAA-score depth before payout is triggered |
| `--min-payout-sompi` | `100000` | Minimum per-miner payout (0.001 XENOM) |
| `--pool-fee-percent` | `1.0` | Pool operator fee % deducted before distribution |
| `--fee-per-output` | `2000` | Estimated tx fee per output (sompi) |

---

## Obtaining `--pool-private-key`

`--pool-private-key` is a **secp256k1 private key** (32 bytes = 64 lowercase hex characters).  
The `--mining-address` you pass to the bridge **must be the P2PK address derived from this exact key** — coinbase rewards land there and are spent on payouts.

### Option 1 — Generate a fresh keypair (recommended)

The bridge has a built-in key generator. Run it **once**, save the output securely:

```bash
./xenom-stratum-bridge --keygen
```

Example output:
```
  Private key  : a3f1c8...64 hex chars...
  Pool address : xenom:qp...your address...

Use these flags when starting the bridge:
  --mining-address xenom:qp... \
  --pool-private-key a3f1c8...
```

> ⚠️ **The private key controls all pool coinbase rewards. Store it like a password — never commit it to version control.**

### Option 2 — Use an existing Xenom wallet key

If you already have a pool wallet and know its private key hex (e.g. exported from the Xenom CLI wallet):

1. Confirm the key is 64 hex characters (32 bytes).
2. Derive the matching address and verify it equals your `--mining-address`.  
   You can use `--keygen` as a sanity check: import the same hex and compare.

### Option 3 — Environment variable (recommended for production)

Avoid putting the key on the command line (it appears in `ps`/logs). Use an env var instead:

```bash
export POOL_PRIVKEY="a3f1c8...64hex..."
./xenom-stratum-bridge \
  --mining-address xenom:qp... \
  --pool-private-key "$POOL_PRIVKEY" \
  ...
```

Or use a systemd `EnvironmentFile`:

```ini
# /etc/xenom-pool/secrets.env   (chmod 600, owned by pool user)
POOL_PRIVKEY=a3f1c8...64hex...
```

```ini
# /etc/systemd/system/xenom-pool.service
[Service]
EnvironmentFile=/etc/xenom-pool/secrets.env
ExecStart=/usr/local/bin/xenom-stratum-bridge \
  --mining-address xenom:qp... \
  --pool-private-key ${POOL_PRIVKEY} \
  --utxoindex ...
```

### Key / address relationship

```
secp256k1 secret key (32 B)
  └─ x-only public key (32 B)
       └─ Address::new(Prefix::Mainnet, Version::PubKey, pubkey_bytes)
            = xenom:q...   ← this is your --mining-address
```

The node must be started with `--utxoindex` so the bridge can query UTXOs at that address to fund payout transactions.

### REST API & Database

| Flag | Default | Description |
|---|---|---|
| `--api-listen` | `0.0.0.0:1445` | HTTP REST API address (empty string = disable) |
| `--pool-name` | `Xenom Pool` | Pool name shown in API responses |
| `--db-path` | `pool.db` | SQLite database file path (empty string = disable) |

---

## REST API

The bridge exposes a JSON REST API on port **1445** by default.

| Endpoint | Description |
|---|---|
| `GET /api/pool` | Pool overview: name, uptime, connected miners, hashrate, totals |
| `GET /api/miners` | All miners (connected and historical) with per-miner stats |
| `GET /api/miners/:worker` | Single miner by worker name or address |
| `GET /api/blocks` | Found blocks with payout status |
| `GET /api/payments` | Payout records with per-miner proportions and tx IDs |
| `GET /api/network` | Node DAG info: virtual DAA score, network hashrate, difficulty |
| `GET /api/stats` | Combined snapshot of pool + network + miners in one call |

### Example responses

```bash
curl http://localhost:1445/api/pool
```
```json
{
  "name": "Xenom Pool",
  "uptime_secs": 3600,
  "connected_miners": 3,
  "pool_hashrate_hps": 45000000,
  "total_shares": 12400,
  "total_blocks": 7,
  "pending_payouts": 3,
  "paid_payouts": 4
}
```

```bash
curl http://localhost:1445/api/network
```
```json
{
  "virtual_daa_score": 9182345,
  "network_hashrate_hps": 1800000000000,
  "sink_hash": "a3f1...c2d8",
  "difficulty": 12489234.5
}
```

```bash
curl http://localhost:1445/api/miners
```
```json
[
  {
    "worker": "xenom:qp...ab.rig1",
    "address": "xenom:qp...ab",
    "connected_since": 1741729200,
    "last_share_at": 1741729800,
    "shares_submitted": 240,
    "blocks_found": 1,
    "current_difficulty": 4096,
    "hashrate_hps": 2730,
    "connected": true
  }
]
```

CORS is enabled (`*`) so the API can be consumed directly by a web dashboard.

---

## Stratum Protocol — Xenom Genome PoW Extension

### Handshake (client → server)

```json
{"id":1,"method":"mining.subscribe","params":["miner/1.0",null]}
{"id":2,"method":"mining.authorize","params":["wallet.worker","x"]}
```

### Subscribe response (server → client)

```json
{
  "id": 1,
  "result": [
    [["mining.set_difficulty","<en1>"],["mining.notify","<en1>"]],
    "<extranonce1_hex>",
    4
  ],
  "error": null
}
```

- `extranonce1_hex` — 8 hex chars (4 bytes), server-assigned per connection
- `4` — `extranonce2_size` in bytes

### `mining.notify` (server → client)

```json
{
  "id": null,
  "method": "mining.notify",
  "params": [
    "000000000000000a",   // job_id (16 hex chars)
    "a3f1...c2d8",       // pre_pow_hash (64 hex chars = 32 bytes)
    "1c00ffff",          // bits — compact difficulty target (8 hex chars)
    "7e49...ab12",       // epoch_seed (64 hex chars = 32 bytes)
    "0000019612345678",  // timestamp in ms (16 hex chars)
    true                 // clean_jobs — abandon previous work
  ]
}
```

#### What miners must do with these fields

1. **`pre_pow_hash`** — the block header hash with `nonce=0` and `timestamp=0`.  
   Miner computes:  
   `genome_mix_hash(packed_genome, epoch_seed, full_nonce, pre_pow_hash)`

2. **`epoch_seed`** — selects which 1 MB genome fragment is mixed.

3. **`bits`** — convert to 256-bit target to check `genome_mix_hash_result < target`.

4. **`timestamp`** — include unchanged in the submitted block header  
   (the bridge uses the template timestamp when reconstructing the block).

5. **Full nonce** (64-bit) = `extranonce1 (high 32 bits)` ‖ `extranonce2 (low 32 bits)`  
   GPU kernels sweep `extranonce2`; `extranonce1` partitions the nonce space across connections.

### `mining.submit` (client → server)

```json
{
  "id": 4,
  "method": "mining.submit",
  "params": [
    "wallet.worker",    // worker name
    "000000000000000a", // job_id
    "0a1b2c3d"          // extranonce2 (8 hex chars = 4 bytes, big-endian)
  ]
}
```

The bridge reconstructs the full nonce, builds the complete block, and submits it to the node via gRPC.  
The **node performs full Genome PoW validation**; the bridge returns `true`/error based on the node response.

---

## Architecture

```
Xenom node (gRPC)
    │  get_block_template (poll every 200 ms)
    │  submit_block
    ▼
JobManager (Arc<RwLock>)
    │  watch::channel — broadcasts new jobs
    ▼
StratumServer (TCP :1444)
    ├─ MinerA — extranonce1=00000001
    ├─ MinerB — extranonce1=00000002
    └─ MinerC — extranonce1=00000003
```

- Each connection gets a **unique `extranonce1`**, partitioning the 64-bit nonce space.  
  At 4 BPS a single miner's 4-byte extranonce2 space (4 × 10⁹ nonces) outlasts a job window.
- All submitted nonces are **forwarded to the node**; no genome dataset is required at the bridge.
- Up to **64 recent jobs** are retained for late submissions (stale after 60 s).

---

## Notes

- The genome dataset (`grch38.xenom`, ~739 MB) must be present **at the miner**, not at the bridge.
- This bridge implements **solo-style stratum** (no share accounting / per-miner payout).  
  For pool payouts a separate share-accounting layer and the genome file at the bridge are needed.
- The `mining.set_difficulty` value is informational; all submissions are forwarded regardless.
