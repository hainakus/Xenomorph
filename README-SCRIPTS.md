# Xenomorph AI Devnet Scripts

This directory contains bash scripts to deploy, test, and manage the Xenomorph AI devnet using Docker Compose.

## Quick Start

```bash
# 1. Copy and edit environment
cp .env.example .env

# 2. One-command setup
./scripts/quick-devnet.sh

# 3. Verify health
./scripts/check-devnet-health.sh

# 4. Run tests
./scripts/test-governance-flow.sh --proposal add-model --model-id test-v1
./scripts/test-payment-flow.sh --amount 0.1

# 5. Monitor
./scripts/monitor-dashboard.sh

# 6. Clean up
./scripts/cleanup-devnet.sh --volumes
```

## Requirements

- Docker Engine >= 20.10
- Docker Compose plugin or `docker-compose` >= 2.0
- Git
- `jq` (for JSON processing in some scripts)
- `cast` (Foundry) for governance and payment tests
- Linux (primary), macOS (secondary), Windows via WSL

## Scripts Reference

| Script | Purpose | Example |
|--------|---------|---------|
| `quick-devnet.sh` | One-command clone/build/start | `./scripts/quick-devnet.sh` |
| `build-devnet.sh` | Build all Docker images | `./scripts/build-devnet.sh --no-cache --parallel` |
| `check-devnet-health.sh` | Verify all services | `./scripts/check-devnet-health.sh --json` |
| `test-checkpoint-sync.sh` | Test checkpoint fast sync | `./scripts/test-checkpoint-sync.sh --from-block 1000 --verify` |
| `test-dnabert2-devnet.sh` | End-to-end DNABERT-2 training test | `./scripts/test-dnabert2-devnet.sh --duration 300` |
| `test-governance-flow.sh` | Governance lifecycle test | `./scripts/test-governance-flow.sh --proposal add-model --model-id test-v1` |
| `test-payment-flow.sh` | USDT payment test | `./scripts/test-payment-flow.sh --amount 0.1` |
| `stress-test.sh` | Load test | `./scripts/stress-test.sh --miners 5 --duration 30m` |
| `logs-collector.sh` | Collect logs | `./scripts/logs-collector.sh --since 1h` |
| `cleanup-devnet.sh` | Tear down | `./scripts/cleanup-devnet.sh --volumes --images` |
| `monitor-dashboard.sh` | Live monitoring | `./scripts/monitor-dashboard.sh` |

## Environment Variables

All scripts read from `.env`. See `.env.example` for the full list.

Key variables:

- `XENO_VERSION` — image tag version
- `XENO_NODE_RPC_PORT` / `XENO_NODE_P2P_PORT` — node ports
- `XENO_SEED_GRPC_PORT` — seed gRPC port
- `XENO_MINER_RPC_PORT` — seed WebSocket port used by miners (default 17110)
- `XENO_ANVIL_PORT` — local EVM devnet port
- `XENO_MINER_TRAINER` — trainer backend for the miner: `mock` (default, fast CPU-less), `cpu` (legacy MLP trainer), or `dnabert2` (real DNABERT-2 training from seed-node checkpoint)
- `XENO_STRESS_MINERS` / `XENO_STRESS_DURATION` — stress defaults

## Makefile Targets

```bash
make setup    # quick-devnet.sh
make build    # build-devnet.sh
make test     # run health + governance + payment tests
make stress   # stress-test.sh
make logs     # logs-collector.sh
make clean    # cleanup-devnet.sh
```

## Notes

- The first build downloads/copies a Rust toolchain and compiles `xenom`, `seed-node`, and `xenom-miner`; this may take several minutes.
- On Linux, `build-devnet.sh` compiles binaries locally and copies them into images.
- On macOS / non-Linux hosts, `build-devnet.sh` defaults to `--docker-build` and compiles the binaries inside the Docker image (slower, but no cross-toolchain needed).
- Use `./scripts/build-devnet.sh --local --target aarch64-unknown-linux-gnu` to cross-compile on Apple Silicon if you have a suitable linker.
- `--trainer=mock` is enabled by default for fast devnet testing; use `--trainer=dnabert2` on CPU/GPU-capable hosts to run real DNABERT-2 training against a seed-node checkpoint. `--mock-mode` is still accepted as a hidden alias for `--trainer=mock`.
- The `test-dnabert2-devnet.sh` script runs a full HITL end-to-end test: it starts the devnet with `--trainer=dnabert2`, waits for the model download, and validates real `loss_before`/`loss_after` values and block submission. It requires ~8 GB RAM and several CPU cores.
