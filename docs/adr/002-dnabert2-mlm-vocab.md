# ADR 002: DNABERT-2 MLM must use the full BPE/k-mer vocabulary

## Status

Proposed.

## Context

`multimolecule/dnabert2` is a BPE-tokenised masked language model. Its published
vocabulary contains thousands of k-mer/BPE tokens (roughly 4096 entries including
special tokens), not the four DNA bases.  A proper DNABERT-2 MLM benchmark
therefore masks *tokens* and asks the model to predict the original token id over
the full vocabulary.

The current Xenomorph stack treats the output head as if it only has four classes
(A, C, G, T).  Specifically:

- The legacy/minimal tests for `DnaBert2Trainer`, `MlmBatchGenerator` and
  `DnaTokenizer` construct a fake vocabulary with only A/T/C/G plus a few
  special tokens.
- The masked-LM inference path for MGM-1 (`seed-node/src/serving/inference_engine.rs`)
  explicitly `narrow`s the logits to the first four positions and reports
  `logits_vocab_size: 4`.
- Synthetic devnet batches are generated as random A/T/C/G strings and then
  tokenised, so even when the real BPE tokenizer is loaded the *input* is still
  single-base tokens.

The result is not MLM in the literature sense; it is a base-reconstruction task.
That is a legitimate benchmark, but it is not the benchmark the project intends
to run for DNABERT-2.

## Decision

Use the full model tokenizer throughout the DNABERT-2 lifecycle:

1. **Training** (`xenom-miner`): `DnaBert2Trainer` and `MlmBatchGenerator` must
   use the real `tokenizer.json` shipped with the checkpoint.  Synthetic devnet
   sequences must be *sampled at the token level* (or drawn from real genomic
   sequence that tokenises into proper BPE tokens), not as random single bases.
2. **Inference** (`seed-node` / `xenom`): the masked-LM endpoint must return the
   full vocabulary logits for each masked token, with `logits_vocab_size` equal
   to the tokenizer's true `vocab_size`.
3. **Benchmarking** (`xenom_benchmark`): keep token-level masking (`MaskingStrategy`)
   and use `BenchmarkTokenizer` loaded from Hugging Face.  Metrics should be
   reported in terms of token cross-entropy / perplexity, with base-level metrics
   kept only as a secondary/legacy view.

MGM-1 may keep its 4-base vocabulary because that is its own design; the issue
here is specifically DNABERT-2 and any other BPE/k-mer model being evaluated with
a 4-class head.

## Consequences

- **Positive**: the benchmark becomes comparable with published DNABERT-2 numbers.
- **Positive**: the model is exercised over its full output distribution, not just
  the first four logits.
- **Negative**: the synthetic devnet batch generator can no longer produce
  realistic BPE inputs from random single bases; it must either sample from the
  tokenizer vocabulary or use real genome slices.
- **Negative**: `seed-node` inference must be changed to stop narrowing logits to
  four classes and must decode with `decode_to_sequence` rather than character
  decoding.

## Implementation slices

1. Update `xenom-miner` tests and `MlmBatchGenerator` to use real BPE tokenizers
   and token-level synthetic sequences.
2. Remove the `logits.narrow(..., 0, 4)` path in `seed-node` masked-LM inference;
   return the full vocabulary logits.
3. Update `xenom_benchmark` to always use token-level evaluation for
   `multimolecule/dnabert2` and report vocab-correct metrics.
4. Add integration tests that verify `logits_vocab_size` equals the real
   `tokenizer.vocab_size`.

## References

- `docs/issues/08-dnabert2-mlm-vocab.md`
- `docs/PRD-DNABERT2-CANDLE.md`
- `seed-node/src/serving/inference_engine.rs`
- `xenom-miner/src/data.rs`
- `xenom-miner/src/trainer/dnabert2_trainer.rs`
- `xenom-miner/src/tokenizer.rs`
- `xenom_benchmark/masking.py`
- `xenom_benchmark/evaluator.py`
