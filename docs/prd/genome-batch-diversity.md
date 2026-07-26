
# PRD: Genome Batch Diversity for DNABERT-2 Training

## Status

Implemented.

## Problem Statement

The `.xenom` genome archive's **merkle root** (`577126c448d24d132ba77436517a7db2203d6fce0cd81e2b84db39875d43ee80`) is a 32-byte file identifier and integrity check. It is **not** DNA. The real genome data — ~3 billion bases packed 2-bit per base (A=00, C=01, G=10, T=11) — lives inside the `.xenom` archive.

`DNABERT-2` needs the actual DNA sequences (`ACGT...`) for training, and it needs **variety**. The previous code extracted real DNA correctly (`extract_sequences`), but it always re-created the `GenomeBatchGenerator` with a seed derived only from the static merkle root (and, in the unified node, the `current_epoch`, which only advanced when a block was accepted). The result was the same ~88 slices being served repeatedly, so the model trained on a tiny repeating window instead of the full genome.

## Goal

Every `GetGenomeTrainingBatch` request must return a different set of genomic windows sampled from the real DNA inside the `.xenom` archive identified by the merkle root.

## Non-Goal

- Changing the merkle root semantics or verification.
- Replacing `GenomeBatchGenerator` with a different sampler.
- Adding a timestamp-based seed (timestamps are non-deterministic and make replay/debugging harder; a monotonic atomic counter is sufficient).

## Solution

Derive the `GenomeBatchGenerator` seed with `blake3` over the merkle root plus a per-request nonce. The merkle root still selects the correct archive; the nonce makes the seed change on every request, advancing the RNG and producing different slices.

The unified `xenom` node also mixes in the `current_epoch` so accepted training blocks additionally rotate the starting distribution.

### Code

#### Unified `xenom` node (`xenom/src/training/coordinator.rs`)

```rust
let epoch = self.inner.current_epoch.load(Ordering::Relaxed);
let batch_nonce = self.inner.batch_counter.fetch_add(1, Ordering::Relaxed);
let mut seed_input = Vec::with_capacity(48);
seed_input.extend_from_slice(&request.genome_merkle_root);
seed_input.extend_from_slice(&epoch.to_le_bytes());
seed_input.extend_from_slice(&batch_nonce.to_le_bytes());
let seed = *blake3::hash(&seed_input).as_bytes();

let mut generator = GenomeBatchGenerator::new(archive, seed);
let seq_len_bases = 512usize.saturating_mul(4);
let mut batch = generator.generate_batch(request.preferred_batch_size.max(1), seq_len_bases);
batch.batch_id = batch_nonce;
batch.model_id = request.model_id;
```

#### Standalone `seed-node` (`seed-node/src/rpc/server.rs`)

```rust
let batch_nonce = batch_counter.fetch_add(1, Ordering::Relaxed);
let mut seed_input = Vec::with_capacity(40);
seed_input.extend_from_slice(&request.genome_merkle_root);
seed_input.extend_from_slice(&batch_nonce.to_le_bytes());
let seed = *blake3::hash(&seed_input).as_bytes();

let mut generator = GenomeBatchGenerator::new(archive, seed);
let seq_len_bases = 512usize.saturating_mul(4);
let mut batch = generator.generate_batch(request.preferred_batch_size.max(1), seq_len_bases);
batch.batch_id = batch_nonce;
batch.model_id = request.model_id.clone();
```

## Data Flow

1. Miner requests `GetGenomeTrainingBatch` with `genome_merkle_root`.
2. Node loads the `.xenom` archive identified by that merkle root.
3. Node fetches/increments a monotonic `batch_counter` (and reads `current_epoch` in the unified node).
4. Node computes `seed = blake3(merkle_root || epoch || batch_counter)`.
5. `GenomeBatchGenerator::new(archive, seed)` is created, `generate_batch` advances its `StdRng`, and `extract_sequences` returns real `ACGT...` strings.
6. The returned `batch_id` equals the nonce so both miner and node can correlate logs.

## Acceptance Criteria

- [x] Two consecutive `GetGenomeTrainingBatch` requests for the same merkle root return different `data_indices`.
- [x] The extracted `sequences` are real DNA (`ACGT`) from the `.xenom` archive.
- [x] The unified node and standalone `seed-node` both use the new seed derivation.
- [x] Sequence length is `512 * 4 = 2048` bases to match DNABERT-2's ~4:1 BPE token compression.
- [x] `cargo fmt --all`, `cargo clippy`, `cargo test -p xenom-miner -p seed-node -p xenom --lib`, and `cargo test --test integration` all pass.

## Related Work

- `GenomeBatchGenerator` contiguous-window improvement: `seed-node/src/genome/batch_generator.rs`
- K-mer span masking and reverse-complement augmentation in `xenom-miner/src/data.rs`
- Canonical human genome merkle root enforced for all DNA trainers: `xenom-miner/src/main.rs`

## Notes

- `blake3` was chosen because it is already a dependency of both `xenom` and `seed-node` and produces a clean 32-byte seed.
- The nonce is monotonic and starts at `1`; `0` is reserved so an uninitialized counter is distinguishable.
