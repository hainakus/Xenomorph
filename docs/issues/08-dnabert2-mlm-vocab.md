# Issue 8: Fix DNABERT-2 benchmark to use the full BPE/k-mer vocabulary

## Parent

ADR: `docs/adr/002-dnabert2-mlm-vocab.md`

## Problem

The current DNABERT-2 evaluation and training path collapses the model's output
head to four classes (A, C, G, T).  This turns the intended MLM benchmark into a
base-reconstruction benchmark, which is not what is used in the DNABERT-2
literature.  The `multimolecule/dnabert2` tokenizer has a BPE/k-mer vocabulary of
thousands of tokens, and the benchmark must predict over that full vocabulary.

## What to fix

### `seed-node` masked-LM inference

- `seed-node/src/serving/inference_engine.rs` currently narrows the DNABERT-2
  logits to the first four positions for the MGM-1 path and reports
  `logits_vocab_size: 4`.  The general DNABERT-2 path (`evaluate_masked_lm_dnabert`)
  already returns full logits, but the codebase still contains the 4-logit fallback
  and tests that assume a 4-token vocabulary.
- Ensure all masked-LM paths that serve DNABERT-2 return logits shaped
  `[n_masks, tokenizer.vocab_size]` and set `logits_vocab_size` accordingly.

### `xenom-miner` training and batch generation

- `xenom-miner/src/data.rs` generates random A/T/C/G strings for the synthetic
  devnet batch.  When these are tokenised by the real BPE tokenizer they produce
  one token per base, which keeps the task at base level.
- Change the synthetic generator to sample tokens (or real genomic sequence)
  instead of single bases, so the masked positions are BPE/k-mer tokens.
- The `MlmBatchGenerator` must mask at the token level and produce labels that
  are real token ids in the model's vocabulary.

### `xenom-miner` trainer and tests

- `xenom-miner/src/trainer/dnabert2_trainer.rs` and `xenom-miner/src/tokenizer.rs`
  contain unit tests that build a fake 4-base tokenizer.  Replace or augment these
  with tests that load a real DNABERT-2 `tokenizer.json` (or a realistic minimal
  BPE tokenizer) and verify that training produces a full-vocab loss.

### `xenom_benchmark`

- `xenom_benchmark/masking.py` and `xenom_benchmark/evaluator.py` already support
  token-level masking and full-vocab logits.  Ensure the default evaluator
  configuration for `multimolecule/dnabert2` uses `use_token_level=True` and
  computes token cross-entropy / perplexity over `vocab_size`.
- Remove or clearly label any report/metric named as DNABERT-2 MLM that is
  implicitly a 4-base reconstruction score.

## Acceptance criteria

- [ ] `seed-node` masked-LM returns `logits_vocab_size == tokenizer.vocab_size` for
      DNABERT-2, not 4.
- [ ] `xenom-miner` `MlmBatchGenerator` can generate token-level batches using the
      real DNABERT-2 tokenizer and mask whole BPE/k-mer tokens.
- [ ] `xenom-miner` DNABERT-2 unit tests use a tokenizer whose vocabulary is not
      limited to A/T/C/G.
- [ ] `xenom_benchmark` reports token-level cross-entropy and perplexity for
      `multimolecule/dnabert2`.
- [ ] A note is added to `AGENTS.md` or the benchmark docs explaining that
      base-level reconstruction is a separate, legacy metric.

## Blocked by

- `docs/issues/02-tokenizer-mlm-batch.md` (tokenizer loading and MLM batch
  generation must be in place first).

## Related

- `docs/PRD-DNABERT2-CANDLE.md`
- `docs/adr/002-dnabert2-mlm-vocab.md`
