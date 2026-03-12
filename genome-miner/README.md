# genome-miner

GPU miner for the **Xenom** network (Genome PoW).  
Supports both **pool mining** (Stratum v1) and **solo mining** (direct gRPC node).

---

## Requirements

- **GPU**: any Vulkan-capable GPU (NVIDIA / AMD / Intel) on Linux/Windows; Metal on macOS
- **Genome file**: `grch38.xenom` (~739 MB) — required for Genome PoW
- **Xenom address**: to receive rewards

---

## Download

Grab the latest release for your platform from [Releases](https://github.com/hainakus/Xenomorph/releases):

| Platform | Archive |
|---|---|
| Linux x86_64 | `genome-miner-linux-x86_64.tar.gz` |
| Windows x86_64 | `genome-miner-windows-x86_64.zip` |
| macOS (Apple Silicon) | `genome-miner-macos-aarch64.tar.gz` |

The genome file (`grch38.xenom`) is included in the same release and must be placed at  
`~/.rusty-xenom/grch38.xenom` (auto-detected) or passed with `--genome-file`.

---

## Quick start — Pool mining (Stratum)

Connect to a public pool at `xenom.space:1444`.  
Rewards are distributed automatically by the pool based on your shares.

### Linux / macOS

```bash
tar -xzf genome-miner-linux-x86_64.tar.gz

./genome-miner gpu \
  --gpu all \
  --mainnet \
  --genome-file ~/.rusty-xenom/grch38.xenom \
  --stratum "stratum+tcp://xenom.space:1444" \
  --stratum-worker "xenom:<your-address>.$(hostname)"
```

### Windows (PowerShell)

```powershell
Expand-Archive genome-miner-windows-x86_64.zip

.\genome-miner.exe gpu `
  --gpu all `
  --mainnet `
  --genome-file "$env:USERPROFILE\.rusty-xenom\grch38.xenom" `
  --stratum "stratum+tcp://xenom.space:1444" `
  --stratum-worker "xenom:<your-address>.$env:COMPUTERNAME"
```

> The `--stratum-worker` must follow the format `xenom:<your-address>[.workername]`  
> so the pool can attribute rewards to your wallet.

---

## Quick start — Solo mining (direct node)

Connect directly to a Xenom node. All block rewards go entirely to your address.  
Requires a node running with `--utxoindex` (or use the public node below).

### Linux / macOS

```bash
tar -xzf genome-miner-linux-x86_64.tar.gz

./genome-miner gpu \
  --gpu all \
  --mainnet \
  --genome-file ~/.rusty-xenom/grch38.xenom \
  --rpcserver xenom.space:36669 \
  --mining-address xenom:<your-address>
```

### Windows (PowerShell)

```powershell
Expand-Archive genome-miner-windows-x86_64.zip

.\genome-miner.exe gpu `
  --gpu all `
  --mainnet `
  --genome-file "$env:USERPROFILE\.rusty-xenom\grch38.xenom" `
  --rpcserver xenom.space:36669 `
  --mining-address xenom:<your-address>
```

---

## Automated installer (Linux)

One-liner scripts that download, extract and start the miner automatically:

```bash
# Pool mining
bash <(curl -s https://raw.githubusercontent.com/hainakus/Xenomorph/main/integrations/stratum-bridge/install-miner.sh) xenom:<your-address>

# Solo mining
bash <(curl -s https://raw.githubusercontent.com/hainakus/Xenomorph/main/integrations/stratum-bridge/install-miner-solo.sh) xenom:<your-address>
```

---

## CLI reference (`gpu` subcommand)

| Flag | Default | Description |
|---|---|---|
| `--gpu` | `0` | GPU adapter(s): `0`, `1`, `0,1`, or `all` |
| `--mainnet` | — | Enable Genome PoW at mainnet activation DAA |
| `--genome-file` | `~/.rusty-xenom/grch38.xenom` | Path to genome dataset |
| `--stratum` | — | Pool URL, e.g. `stratum+tcp://xenom.space:1444` |
| `--stratum-worker` | — | Worker name sent to pool (use `xenom:<addr>.name`) |
| `--stratum-password` | `x` | Stratum password |
| `--rpcserver` | `localhost:36669` | Node gRPC endpoint (solo mode) |
| `--mining-address` | — | Reward address (required for solo mode) |
| `--batch-size` | `1048576` | Nonces per GPU dispatch (1M) |
| `--nonce-offset` | `0` | Instance index for multi-process setups |
| `--list-gpus` | — | Print available GPU adapters and exit |
| `--no-tui` | — | Plain log output (no dashboard) |

---

## Multi-GPU

All GPUs:
```bash
./genome-miner gpu --gpu all --mainnet --stratum "stratum+tcp://xenom.space:1444" \
  --stratum-worker "xenom:<your-address>" --genome-file ~/.rusty-xenom/grch38.xenom
```

Specific GPUs (e.g. 0 and 2):
```bash
./genome-miner gpu --gpu 0,2 ...
```

---

## Notes

- The genome file (`grch38.xenom`) must be present **at the miner**, not at the bridge or node.
- Place the file at `~/.rusty-xenom/grch38.xenom` to avoid specifying `--genome-file` each time.
- Windows path: `%USERPROFILE%\.rusty-xenom\grch38.xenom`
