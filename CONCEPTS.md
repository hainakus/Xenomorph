# XEN, PoW and Model Training in Xenomorph

Conceptual report covering the native coin, the proof-of-work algorithm, and how the DNABERT-2 model is trained.

---

## 1. The coin

The native token of the network is **XEN** (also referred to as Xenom/XEN in the miner messages and emission contract).

### Emission

- New coins are minted through the **coinbase** of blocks, following the same deflationary structure as Kaspa:
  - Pre-deflationary phase: a base subsidy per block (`pre_deflationary_phase_base_subsidy`).
  - Deflationary phase: a monthly subsidy table (`SUBSIDY_BY_MONTH_TABLE`) that decreases over 426 months.
  - Implementation: `consensus/src/processes/coinbase.rs` (`calc_block_subsidy`, `SUBSIDY_BY_MONTH_TABLE`).

- The miner CLI tracks accumulated rewards using `BLOCK_REWARD = 100` per submitted block.
  - See `xenom-miner/src/main.rs`, lines 29-32.

- Per-model rewards and staking:
  - The `ModelGovernance.sol` contract records, for each active model, `reward_per_block` and `min_stake_to_train`.
  - The seed-node reads these values in `seed-node/src/consensus/dynamic_registry.rs`.

### Token utility

- **Mining**: reward for producing blocks with a valid training proof.
- **Governance**: proposals and votes on which models are active.
- **Inference**: the `api-gateway` may require payments in **USDT** for inference calls (`InferencePayments`), while miner rewards remain in XEN.

---

## 2. The Proof-of-Work algorithm

Xenomorph uses **two layers** of PoW:

### 2.1 Block header PoW (consensus)

The `xenom` full node selects the algorithm based on `daa_score`:

- **Classic KHeavyHash** — before `genome_pow_activation_daa_score`.
- **Genome PoW** — after that point, based on the human GRCh38 genome.

#### Genome PoW, step by step

1. **Fragment selection**: `fragment_index(epoch_seed, nonce, fragment_size)` picks a 1 MB fragment from the genome (~3 GB).
2. **Deterministic mutations**: `apply_mutations(fragment, epoch_seed)` applies 4 to 16 rounds of operations (swap, insert, rotate, XOR, shift). The number and parameters depend only on `epoch_seed`, so they are reproducible.
3. **Fitness score**: `compute_fitness_with_seed` combines Shannon entropy, GC content (~50% ideal), and 4-gram complexity.
4. **Final hash**: `genome_final_hash = blake3(blake3(mutated_fragment) ‖ pre_pow_hash ‖ nonce_le)`.
5. **Validation**: the result is compared against `target`; if lower, the block is valid.

#### Memory-hard variant

`genome_mix_hash` performs 8 rounds of random 32-byte reads across the packed genome (~739 MB). This forces the entire dataset to reside in fast RAM/storage, making cheap ASICs impractical.

- See `consensus/pow/src/genome_pow.rs`.
- See `xenom/src/training/coordinator.rs`, lines 345-415, for the mining loop inside the node.

### 2.2 Useful PoW — model training proof

This is the main innovation: instead of random hashes, the miner **trains a DNABERT-2 batch** and generates a `TrainingProof`:

```rust
pub struct TrainingProof {
    pub model_id: ModelId,
    pub base_checkpoint: Hash,
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: Hash,
    pub zk_proof: ZKProof,
    pub batch_indices: Vec<u64>,
}
```

#### Training difficulty

A proof is accepted when:

- `loss_before - loss_after > min_improvement` (e.g., 0.001)
- `loss_after < max_loss_after` (e.g., 1.0)
- `ZKProof` verifies (currently a placeholder: checks structure and `loss_after < loss_before`)

- See `consensus/core/src/pow/training_proof.rs`.

#### Validation cycle in the node

`xenom/src/training/coordinator.rs`:

1. Verifies the address and network prefix.
2. Confirms `model_id` is the active one.
3. Compares `base_checkpoint` with `active_weights_hash`.
4. Builds `ConsensusTrainingProof` and calls `verify(&difficulty)`.
5. Only then mines the block header (KHeavyHash or Genome PoW) and submits the block.

---

## 3. Advantage over traditional PoW

| Traditional PoW | Xenomorph UsefulPoW |
|---|---|
| Energy spent computing hashes with no external utility | Energy spent training an AI model |
| Security = hashes per second | Security = training compute per second |
| No useful output | Output: continuous improvement of DNABERT-2 |
| ASICs optimize hash grinding | GPUs/TPUs optimize transformer training |

In other words, **network security and scientific/AI progress are the same operation**. The more difficulty, the more training is performed, and the global model improves through FedAvg gradient aggregation.

---

## 4. How model training works

### 4.1 The model

- **DNABERT-2** (`multimolecule/dnabert2`):
  - ~110 M parameters.
  - BERT-like transformer with DNA k-mer embeddings.
  - ALiBi positional bias (`alibi_starting_size`) instead of classic positional embeddings.
  - Tokenizer with `<pad>`, `<mask>`, `A`, `C`, `G`, `T` tokens.
- The seed-node downloads the model from Hugging Face, encrypts it locally with AES-256-GCM, and serves the checkpoint to the miner.
- The miner loads it into memory for training/inference without persisting locally.

### 4.2 Batch generation

1. The miner requests a `TrainingBatch` from the seed-node over WebSocket.
2. The batch contains:
   - `model_id`
   - `base_checkpoint` (hash of the initial weights)
   - `data_indices` (indices of the data to train)
   - `target_improvement` and `learning_rate`
3. If the miner uses `--genome-merkle`, the `GenomeBatchGenerator` extracts DNA sequences from a `.xenom` archive and produces an MLM batch.
4. Otherwise, `DnaBert2Trainer` generates a synthetic DNA batch.

### 4.3 Training step (`DnaBert2Trainer::train_mlm_batch`)

1. **Forward + backward** (`compute_gradients`):
   - Passes the batch through the model.
   - Computes masked-language modeling loss (cross-entropy).
   - Backpropagates and collects per-layer gradients.
   - Returns `loss_before` and the gradients.
2. **Optimization** (`apply_gradients`):
   - Applies `ManualAdamW` (AdamW manually implemented over Candle's `VarMap`).
   - Learning rate is clamped to `1e-5` for the real trainer.
3. **Forward after optimization** (`compute_loss_scalar`):
   - Measures `loss_after` to prove the loss improved.
4. **Gradient commitment** (`gradient_commitment_from_named_tensors`):
   - Computes `blake3` over the gradients, producing `gradients_commitment`.
   - In multi-GPU mode, gradients are compressed (`top-k`) and averaged on the master GPU before the commitment.

### 4.4 Multi-GPU

`MultiGpuTrainer`:

- Creates a model replica on each GPU.
- Trains micro-batches in parallel.
- Collects gradients on the master GPU.
- Averages gradients directly on the GPU (no CPU round-trip).
- Returns a single `GradientUpdate`.

### 4.5 Submission and aggregation

1. The miner signs the block (`wallet.sign_block`) and submits the `TrainingBlock` to the `xenom` node.
2. At the same time, it submits the encrypted `GradientUpdate` to the seed-node (`submit_gradients`).
3. The seed-node:
   - Checks whether `base_checkpoint` is in the cache (bounded staleness).
   - Gathers `FEDAVG_MIN_PARTICIPANTS` gradients and computes their average (FedAvg).
   - Applies the average to a trainable replica of the model (`DnaBert2Trainer` on CPU).
   - Stores the new checkpoint and promotes it to `active_checkpoint` if it was the active head.

- See `seed-node/src/model/manager.rs` for cache and FedAvg implementation.
- See `xenom-miner/src/trainer/dnabert2_trainer.rs` for the actual training code.
- See `xenom-miner/src/main.rs` for the full miner submission loop.

---

## 5. Visual summary

```text
Miner                              Seed-node / Xenom-node
  │                                        │
  ├─ requests TrainingBatch ──────────────►│
  │◄───────────────────────────────────────┤
  │                                        │
  ├─ trains DNABERT-2 batch ──────────────►│
  │  (produces loss_before/loss_after + grads)
  │                                        │
  ├─ sends TrainingBlock ────────────────► │ validates proof
  │  (signed, with ZK proof)               │ mines block header (KHeavyHash/Genome PoW)
  │                                        │ submits block → XEN reward
  │                                        │
  ├─ sends GradientUpdate ───────────────► │ aggregates K gradients (FedAvg)
  │                                        │ applies to model → new checkpoint
  │                                        │
  ◄─ receives new base_checkpoint ─────────┤
```

---

## Key files

- `consensus/pow/src/genome_pow.rs` — Genome PoW.
- `consensus/core/src/pow/training_proof.rs` — TrainingProof and difficulty rules.
- `xenom/src/training/coordinator.rs` — block validation and mining inside the node.
- `xenom-miner/src/trainer/dnabert2_trainer.rs` — real DNABERT-2 training.
- `xenom-miner/src/trainer/multi_gpu.rs` — multi-GPU training and top-k compression.
- `xenom-miner/src/prover/zk_prover.rs` — placeholder ZK proof generation/verification.
- `xenom-miner/src/main.rs` — main miner loop.
- `seed-node/src/model/manager.rs` — checkpoint cache and FedAvg aggregation.
- `seed-node/src/consensus/dynamic_registry.rs` — model registry and reward reading.
- `consensus/src/processes/coinbase.rs` — XEN emission and subsidies.
