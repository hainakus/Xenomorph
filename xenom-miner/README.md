# xenom-miner

Useful Proof-of-Work (UsefulPoW) miner for the Xenomorph blockchain.

The miner performs real machine-learning work (Candle on CPU) or a fast mock trainer, produces a ZK proof of the computed gradients, builds a signed training block, and submits it to a Xenomorph seed node over a Borsh/WebSocket RPC connection.

## Features

- CLI built with `clap`.
- BIP39 mnemonic wallet with encrypted storage in `~/.xenom-miner/`.
- Borsh-serialized RPC messages over WebSocket with automatic reconnection.
- Real CPU training with `candle-core`/`candle-nn`.
- Deterministic mock trainer for dry-runs and tests.
- ZK proof generation/verification (hash-based prototype for this milestone).
- Signed training blocks using secp256k1.
- Graceful shutdown on `Ctrl+C`.

## Build

```bash
cd xenom-miner
cargo build --release
```

## Run

Dry-run with the mock trainer (no seed node required):

```bash
./target/release/xenom-miner \
  --wallet xnom:qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq \
  --rpc-url ws://localhost:16110 \
  --model-id dnabert2 \
  --threads 4 \
  --mock-mode \
  --dry-run
```

Real CPU training against a local seed node:

```bash
./target/release/xenom-miner \
  --rpc-url ws://localhost:16110 \
  --model-id dnabert2 \
  --threads 4
```

The first run creates an encrypted wallet in `--data-dir` (default `~/.xenom-miner`).

## Test

```bash
cargo test
cargo clippy -- -D warnings
```

## Project Structure

```
xenom-miner/
├── Cargo.toml
├── README.md
├── src/
│   ├── main.rs                 # CLI + async mining loop
│   ├── lib.rs                  # Public module exports
│   ├── block/
│   │   ├── mod.rs
│   │   └── builder.rs          # Training block + PoW header builder
│   ├── config.rs               # MinerConfig persistence
│   ├── prover/
│   │   ├── mod.rs
│   │   └── zk_prover.rs        # ZK proof generation/verification
│   ├── rpc/
│   │   ├── mod.rs
│   │   ├── client.rs           # WebSocket Borsh RPC client
│   │   └── messages.rs         # Borsh message definitions
│   ├── trainer/
│   │   ├── mod.rs
│   │   ├── cpu_trainer.rs      # Candle CPU trainer
│   │   └── mock_trainer.rs     # Deterministic mock trainer
│   └── wallet/
│       ├── mod.rs
│       └── manager.rs          # BIP39 wallet + secp256k1 signing
└── tests/
    └── integration_tests.rs    # Mock RPC server + end-to-end pipeline
```

## Environment Variables

- `XENOM_WALLET_PASSWORD` - default password for the encrypted wallet file.

## License

ISC / MIT / Apache-2.0 (choose your preferred license).
