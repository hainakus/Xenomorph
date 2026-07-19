# Issue 5: DnaBert2 training step

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Implement `DnaBert2Trainer`, a new implementation of the existing `Trainer` trait. It loads the model and tokenizer, generates an MLM batch from `TrainingBatch.data_indices` as a seed, runs a forward pass, computes the masked cross-entropy loss, performs one backward pass, applies an optimizer step (SGD or AdamW), and returns a `TrainingResult` with `loss_before`, `loss_after`, `gradients_commitment` (blake3 hash of the flattened gradient tensors), and `compute_time_ms`. This slice must not block the devnet when the model is still downloading.

## Acceptance criteria

- [ ] `DnaBert2Trainer` implements `Trainer` and is selectable by CLI in a later slice.
- [ ] Training a batch reduces the loss (`loss_after < loss_before`).
- [ ] `gradients_commitment` is non-zero and deterministic for the same input/weights.
- [ ] Unit tests use a tiny synthetic model to verify the training step without downloading 468 MB.

## Blocked by

- `docs/issues/02-tokenizer-mlm-batch.md`
- `docs/issues/04-dnabert2-forward-pass.md`
