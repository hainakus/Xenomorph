# genome-miner — HiveOS Installation Guide

> **Supported OS:** HiveOS (Ubuntu 20.04 base) · Ubuntu 20.04 LTS · Ubuntu 22.04 LTS  
> **Supported GPUs:** NVIDIA (Vulkan) · AMD (Vulkan) · Intel Arc (Vulkan)  
> **Algorithm:** Genome PoW · KHeavyHash (pre-activation fallback)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Install](#2-quick-install)
3. [Manual Install](#3-manual-install)
4. [Genome Dataset](#4-genome-dataset)
5. [HiveOS Flight Sheet](#5-hiveos-flight-sheet)
6. [GPU Selection](#6-gpu-selection)
7. [Stats API](#7-stats-api)
8. [Monitoring & Logs](#8-monitoring--logs)
9. [Troubleshooting](#9-troubleshooting)
10. [Updating](#10-updating)

---

## 1. Prerequisites

### HiveOS rig

- HiveOS image based on Ubuntu 20.04 (any version released after 2022)
- NVIDIA: driver **≥ 590** required (Vulkan 1.3 + wgpu compute support). Driver 590 is the minimum; latest stable recommended.  
  Install on HiveOS: **Hive Shell → `nvidia-driver-update 590`** or select driver `590` in the Hive dashboard (Workers → Edit → Nvidia Driver)
- AMD: AMDGPU-PRO or Mesa with Vulkan (`vulkaninfo` to verify)
- At least **1 GB free RAM** per GPU during startup (genome dataset is loaded into VRAM)
- **~800 MB free disk** at `~/.rusty-xenom/` for the GRCh38 genome file
- Internet access for the first-run genome download

### Verify Vulkan (optional but recommended)

```bash
# Check NVIDIA driver version — must be ≥ 590
nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Install / upgrade to driver 590 on HiveOS
nvidia-driver-update 590          # installs 590.xx series
# Or latest available:
nvidia-driver-update latest

# Verify Vulkan after driver install
vulkaninfo --summary 2>/dev/null | head -20

# AMD
vulkaninfo --summary 2>/dev/null | head -20
```

If `vulkaninfo` is not installed:
```bash
sudo apt-get install -y vulkan-tools
```

---

## 2. Quick Install

Run the following on your HiveOS rig as **root** (via SSH or HiveOS shell):

```bash
# 1. Download the HiveOS release package
MINER_VER="0.15.2"
RELEASE_URL="https://github.com/hainakus/Xenomorph/releases/download/genome-miner-v${MINER_VER}"

mkdir -p /hive/miners/genome-miner
cd /tmp
wget -q "${RELEASE_URL}/genome-miner-${MINER_VER}-hiveos.tar.gz"
tar -xzf "genome-miner-${MINER_VER}-hiveos.tar.gz" -C /hive/miners/genome-miner/

# 2. Run the installer (downloads genome dataset ~780 MB with resume support)
bash /hive/miners/genome-miner/install.sh
```

The installer will:
- Create `~/.rusty-xenom/` and download `grch38.xenom` (~780 MB) with **automatic resume** if interrupted
- Set correct permissions on all scripts
- Create a symlink `/hive/miners/genome-miner/grch38.xenom → ~/.rusty-xenom/grch38.xenom`

> **Interrupted download?** Just re-run `bash /hive/miners/genome-miner/install.sh` — `wget` will resume from where it stopped.

---

## 3. Manual Install

If you prefer to do each step by hand:

```bash
# ── Step 1: Create miner directory ───────────────────────────────────────────
mkdir -p /hive/miners/genome-miner

# ── Step 2: Download and extract miner package ───────────────────────────────
cd /tmp
wget https://github.com/hainakus/Xenomorph/releases/download/genome-miner-v0.15.2/genome-miner-0.15.2-hiveos.tar.gz
tar -xzf genome-miner-0.15.2-hiveos.tar.gz -C /hive/miners/genome-miner/
chmod 755 /hive/miners/genome-miner/genome-miner
chmod 755 /hive/miners/genome-miner/*.sh

# ── Step 3: Download genome dataset (with resume support) ─────────────────────
mkdir -p ~/.rusty-xenom
wget -c https://github.com/hainakus/Xenomorph/releases/download/genome-grch38-v0/grch38.xenom \
  -O ~/.rusty-xenom/grch38.xenom

# ── Step 4: Verify genome file (~780 MB) ─────────────────────────────────────
ls -lh ~/.rusty-xenom/grch38.xenom
# Expected: ~780 MB
```

---

## 4. Genome Dataset

The **GRCh38 human reference genome** dataset is required for mainnet Genome PoW mining.

| Property | Value |
|----------|-------|
| File | `grch38.xenom` |
| Size | ~780 MB |
| Location | `~/.rusty-xenom/grch38.xenom` (auto-detected) |
| Download URL | `https://github.com/hainakus/Xenomorph/releases/download/genome-grch38-v0/grch38.xenom` |

### Auto-download behaviour

`h-run.sh` automatically downloads the genome file on first start if it is absent:

```
[h-run] Genome dataset not found — downloading (~780 MB)...
[h-run] Source : https://github.com/hainakus/Xenomorph/releases/download/...
[h-run] Dest   : /root/.rusty-xenom/grch38.xenom
[h-run] Download will resume automatically if interrupted.
```

If the download is interrupted (power cut, network drop), HiveOS will restart `h-run.sh` and `wget -c` will **resume from the byte offset** automatically — no data is re-downloaded.

### Custom genome path

If you store the file elsewhere, pass it via **Extra config** in the flight sheet:

```
--genome-file /data/genome/grch38.xenom --mainnet
```

---

## 5. HiveOS Flight Sheet

Go to **Hive → Flight Sheets → Add flight sheet** and fill in:

| Field | Value |
|-------|-------|
| **Coin** | XENOM *(or add as custom coin)* |
| **Wallet** | Your `xenom:` reward address |
| **Pool** | *(used as node endpoint — see below)* |
| **Miner** | Custom |
| **Miner name** | `genome-miner` |
| **Installation URL** | *(leave blank — installed manually above)* |
| **Wallet and worker template** | `%WAL%` |
| **Pool URL** | `<your-xenom-node-ip>:36669` |
| **Extra config arguments** | `--mainnet` |
| **API port** | `4000` |

> **Pool URL** is the gRPC endpoint of your Xenom node, **not** a stratum pool.  
> Default port is `36669`. For testnet use port `36660`.

### Example — single GPU, mainnet

```
Pool URL   : 192.168.1.100:36669
Extra args : --mainnet
```

### Example — all GPUs, mainnet

```
Pool URL   : 192.168.1.100:36669
Extra args : --mainnet --gpu all
```

### Example — specific GPUs (0 and 2), testnet

```
Pool URL   : 192.168.1.100:36660
Extra args : --testnet --gpu 0,2
```

### Example — custom genome file path

```
Extra args : --mainnet --gpu all --genome-file /data/genome/grch38.xenom
```

---

## 6. GPU Selection

List available GPUs without starting the miner:

```bash
/hive/miners/genome-miner/genome-miner gpu \
  --mining-address xenom:dummy \
  --list-gpus
```

Example output:
```
3 eligible GPU adapter(s):
  [0] NVIDIA GeForce RTX 3090 — DiscreteGpu (vendor: 0x10de)
  [1] NVIDIA GeForce RTX 3080 — DiscreteGpu (vendor: 0x10de)
  [2] AMD Radeon RX 6900 XT   — DiscreteGpu (vendor: 0x1002)
```

| `--gpu` value | Behaviour |
|---------------|-----------|
| `0` | Only GPU index 0 (default) |
| `0,1` | GPUs 0 and 1 |
| `all` | All eligible GPUs |

---

## 7. Stats API

genome-miner exposes a **HiveOS-compatible JSON API** on port `4000` by default.

```bash
curl http://localhost:4000/
```

Example response:
```json
{
  "hs":      [523400000, 487200000],
  "hs_units": "hs",
  "temp":    [68, 72],
  "fan":     [75, 82],
  "power":   [210.5, 195.0],
  "accepted": 14,
  "rejected": 0,
  "algo":    "genome-pow",
  "ver":     "0.15.2",
  "uptime":  3612,
  "ar":      [14, 0]
}
```

| Field | Description |
|-------|-------------|
| `hs` | Hashrate per GPU in H/s |
| `temp` | GPU temperature °C (from `nvidia-smi` / `rocm-smi` / sysfs) |
| `fan` | Fan speed % |
| `power` | Power draw W |
| `accepted` / `rejected` | Block submission counters |
| `algo` | `genome-pow` or `kheavyhash` |
| `uptime` | Seconds since miner started |

### Change API port

```
Extra config : --mainnet --api-port 4001
```

### Disable API

```
Extra config : --mainnet --api-port 0
```

---

## 8. Monitoring & Logs

### Live log (HiveOS shell)

```bash
tail -f /var/log/miner/genome-miner/genome-miner.log
```

### Check miner process

```bash
pgrep -a genome-miner
```

### Manual stats check

```bash
curl -s http://localhost:4000/ | python3 -m json.tool
```

### GPU hardware check

```bash
# NVIDIA
nvidia-smi --query-gpu=index,name,temperature.gpu,fan.speed,power.draw \
  --format=csv,noheader

# AMD
rocm-smi --showtemp --showfan --showpower
```

---

## 9. Troubleshooting

### Miner doesn't start — "No eligible GPU adapters found"

```bash
# Check Vulkan is working
vulkaninfo --summary 2>&1 | grep -i "gpu\|driver\|error"

# NVIDIA: ensure the Vulkan ICD is visible
ls /usr/share/vulkan/icd.d/
# Should contain: nvidia_icd.json

# If missing, set manually:
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### Genome download stalls or fails

```bash
# Resume the download manually
wget -c https://github.com/hainakus/Xenomorph/releases/download/genome-grch38-v0/grch38.xenom \
  -O ~/.rusty-xenom/grch38.xenom

# Check file size (expect ~780 MB)
ls -lh ~/.rusty-xenom/grch38.xenom
```

### "vm.max_map_count too low" warning (NVIDIA multi-GPU)

NVIDIA Vulkan uses ~500–1000 mmap regions per GPU. With many GPUs the kernel may kill the process silently.

```bash
# Fix immediately (survives until reboot)
sudo sysctl -w vm.max_map_count=1048576

# Fix permanently
echo 'vm.max_map_count=1048576' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Temperature / fan / power shows "—" in TUI

Hardware monitoring requires `nvidia-smi` (NVIDIA) or `rocm-smi` / sysfs (AMD).

```bash
# NVIDIA
nvidia-smi --query-gpu=temperature.gpu,fan.speed,power.draw --format=csv,noheader

# AMD (ROCm)
rocm-smi --showtemp --showfan --showpower
```

If neither is available, the miner still works — only the hardware metrics are unavailable.

### Node not synced

```
WARN Node not synced — waiting for IBD to complete
```

Your Xenom node is still syncing. Mining will start automatically once IBD completes. Wait or switch to a fully-synced node.

### KHeavyHash false-positive warnings

```
WARN [GPU0] KHeavyHash false-positive nonce=0x... — skipping
```

Normal — GPU false-positives are CPU-verified and discarded. Not a problem unless the rate is very high (> 1 per second).

---

## 10. Updating

```bash
# Download new version
MINER_VER="<new-version>"
cd /tmp
wget -q "https://github.com/hainakus/Xenomorph/releases/download/genome-miner-v${MINER_VER}/genome-miner-${MINER_VER}-hiveos.tar.gz"

# Stop the miner via HiveOS dashboard or:
# hive miner stop

# Replace files (genome dataset is NOT touched)
tar -xzf "genome-miner-${MINER_VER}-hiveos.tar.gz" -C /hive/miners/genome-miner/
chmod 755 /hive/miners/genome-miner/genome-miner
chmod 755 /hive/miners/genome-miner/*.sh

# Restart via HiveOS dashboard or:
# hive miner start
```

> The `grch38.xenom` genome dataset does **not** need to be re-downloaded on updates.

---

## File locations reference

| Path | Description |
|------|-------------|
| `/hive/miners/genome-miner/genome-miner` | Miner binary |
| `/hive/miners/genome-miner/h-run.sh` | HiveOS start script |
| `/hive/miners/genome-miner/h-stats.sh` | HiveOS stats script |
| `/hive/miners/genome-miner/h-config.sh` | HiveOS config generator |
| `/hive/miners/genome-miner/genome-miner.conf` | Generated config (flight-sheet args) |
| `/hive/miners/genome-miner/hive-manifest.conf` | HiveOS miner manifest |
| `~/.rusty-xenom/grch38.xenom` | GRCh38 genome dataset (~780 MB) |
| `/var/log/miner/genome-miner/genome-miner.log` | Miner log |
| `http://localhost:4000/` | Stats API (JSON) |
