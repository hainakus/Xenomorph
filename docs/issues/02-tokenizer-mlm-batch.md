# Issue 2: Tokenizer + MLM batch generator

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Build a tokenizer module that loads the downloaded `tokenizer.json` using the `tokenizers` crate and exposes `encode(text) -> Vec<u32>` and `decode(ids) -> String`. Then build a synthetic DNA batch generator: it creates random `A/T/C/G` sequences, encodes them, masks 15% of the tokens with the `[MASK]` id, and returns input tensors (`input_ids`, `token_type_ids`, `attention_mask`, `labels`). This generator will feed the real training step in a later slice.

## Acceptance criteria

- [ ] The tokenizer can encode `"ATCG..."` and decode back to the original string (allowing for added `[BOS]/[EOS]` tokens if present).
- [ ] The MLM batch generator produces tensors of shape `[batch_size, seq_len]`.
- [ ] Exactly 15% of non-special tokens are masked on average.
- [ ] Unit tests cover encoding roundtrip and masking logic.

## Blocked by

- `docs/issues/01-upgrade-candle-model-downloader.md`
