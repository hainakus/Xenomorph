# Implementation Status — branch `xenom-AI-v2`

Report generated from the current repository state and the current development session.

---

## Recent commits

```text
aa8419c Add bounded-staleness checkpoint cache to seed-node
b26b735 Speed up seed-node FedAvg by avoiding temp file and extra weight reads
1babf08 Add MultiGpuTrainer phase timing logs
0e94b82 Avoid CPU round-trip when averaging multi-GPU gradients
e67ef66 Add gradient compression CLI and 10-BPS simnet stress script
5ecff33 Fix remote miner WebSocket timeouts on gradient submission
421ae48 Reject FedAvg gradients for stale checkpoints
e5182fd Add weighted FedAvg and top-k gradient compression
06bab2d Keep persistent AdamW state in FedAvg aggregator
39a84d8 Wire FedAvg gradient aggregation end-to-end (#60)
```

Working tree: **clean** (no uncommitted changes).

---

## Features implemented in this session

### 1. Multi-GPU gradient averaging on the master GPU

- **File:** `xenom-miner/src/trainer/multi_gpu.rs`
- **Change:** gradients no longer need a CPU round-trip; they are moved to the master GPU and averaged/accumulated there.
- **Impact:** removes the CPU-GPU round-trip bottleneck during training.

### 2. Phase timing logs in `MultiGpuTrainer`

- **File:** `xenom-miner/src/trainer/multi_gpu.rs`
- **Change:** emits `INFO` per block with a breakdown of time spent in:
  - `compute`, `gather`, `overflow_check`, `avg`, `add`, `apply`, `loss_after`, `commitment`
  - and `Gradient update build time`.
- **Impact:** makes it easy to see where time is spent (CPU vs GPU, training vs overhead).

### 3. `submit_gradients` optimization in the seed-node

- **File:** `seed-node/src/model/manager.rs`
- **Changes:**
  - Added `DnaBert2Trainer::save_weights_to_bytes()` for in-memory serialization.
  - Added `ModelStorage::load_model_metadata()` to read only `config.enc` and `tokenizer.enc`.
  - FedAvg no longer writes a temporary `safetensors` file and no longer re-reads the old `weights.enc`.
- **Impact:** reduces disk I/O during aggregation, speeding up new checkpoint production.

### 4. Bounded-staleness checkpoint cache

- **File:** `seed-node/src/model/manager.rs`
- **Configuration:**
  - `XENO_CHECKPOINT_HISTORY_SIZE=8` (default)
  - `FEDAVG_MIN_PARTICIPANTS=4` (default for devnet)
- **Behavior:**
  - The seed-node keeps up to 8 checkpoint lineages in memory.
  - It accepts gradients whose `base_checkpoint` is inside that window.
  - Each cache entry has its own `DnaBert2Trainer` (CPU) and `FedAvgAggregator`.
  - When the number of participants for a given base reaches `FEDAVG_MIN_PARTICIPANTS`, it averages and advances that lineage's `head_hash`.
  - It only promotes the new hash to `active_checkpoint` if the previous `head_hash` was the active one.
  - It inserts a new key for the new head so the lineage can continue.
- **Impact:** the miner no longer needs to wait for aggregation after every block; it can submit several gradients on the same `base_checkpoint` before receiving a new active checkpoint.

### 5. Devnet configuration

- **Files:** `.env.example`, `scripts/run-native-devnet.sh`
- **Added:**
  - `XENO_CHECKPOINT_HISTORY_SIZE=8`
  - `FEDAVG_MIN_PARTICIPANTS=4`

---

## Test results

```bash
cargo test -p seed-node -p xenom-miner
```

- `seed-node`: 41 tests passed
- `xenom-miner`: 34 tests passed
- `integration_tests`: 2 tests passed

`cargo fmt --all -- --check`: OK  
`cargo clippy -p seed-node`: no new warnings in `manager.rs` (pre-existing warnings in other files remain).

---

## Blockchain state in this branch

| Aspect | State |
|---|---|
| **Difficulty / DAA** | Unchanged. Devnet keeps `target_time_per_block=1000 ms` and `max_difficulty_target=2^255-1`. With one 3080 producing well below 1 BPS, difficulty stays at the minimum floor. |
| **Block production** | Now mostly limited by `training + aggregation`. With `FEDAVG_MIN_PARTICIPANTS=4`, the miner submits 4 gradients on the same `base_checkpoint` before waiting for a new active checkpoint. |
| **GPU utilization** | Still not continuous. `nvidia-smi dmon` shows bursts of 38-85% `sm` followed by long idle periods. To reach high occupancy, you need: (a) larger `--genome-batch-size` and `--gradient-accumulation` so training lasts longer; (b) `--gradient-top-k-ratio 0.1` to reduce aggregation time. |
| **Convergence / FedAvg** | Supports multiple submissions per base and stale gradients inside the cache. The active lineage advances correctly; gradients outside the cache are rejected. |
| **Finality** | `finality_depth=86400` blocks. For devnet this remains high if block time increases. |

---

## Recommended run

To test high GPU occupancy and continuous flow, start the unified `xenom` node (which now includes the miner WebSocket and gRPC inference):

```bash
# unified xeno-node
FEDAVG_MIN_PARTICIPANTS=4 XENO_CHECKPOINT_HISTORY_SIZE=8 \
  ./target/release/xenom --devnet --utxoindex \
  --miner-ws-listen=0.0.0.0:17110 \
  --inference-grpc-listen=0.0.0.0:50051 \
  --models-dir=./devnet-data-native/models

# miner
./target/release/xenom-miner --trainer dnabert2 \
  --rpc-url ws://127.0.0.1:17110 \
  --gpus 0,1,2 \
  --genome-batch-size 64 \
  --gradient-accumulation 3 \
  --gradient-top-k-ratio 0.1
```

Expected estimate with ~30 s training and ~10 s aggregation:

```text
round time = 4 × 30 s + 10 s = 130 s
GPU active time = 4 × 30 s = 120 s
GPU utilization ≈ 92%
```

---

## Notes

- The `XENO_TARGET_BLOCK_TIME_MS` override is not in this branch; it was implemented and later lost during the session. If you want difficulty to be sensitive to a single 3080, it needs to be reintroduced.
- CPU usage on the unified `xenom` node (where gradient aggregation runs) remains the limiting factor when `--gradient-top-k-ratio` is 1.0 (dense 440 MB payload). The standalone `seed-node` is no longer required in the unified devnet.
