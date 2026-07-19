# Issue 3: DnaBert2 config and safetensors weight loading

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Create a Rust representation of the DNABERT-2 `config.json` (hidden size, num layers, heads, vocab size, ALiBi settings, etc.). Implement a weight loader that reads the downloaded `model.safetensors` into a `VarBuilder` and handles the `dnabert2.*` weight prefix used by the `multimolecule` checkpoint. For this slice, it is enough to parse the config and load all tensors successfully; the actual model architecture is built in the next slice.

## Acceptance criteria

- [ ] `config.json` is parsed into a typed struct.
- [ ] All tensors from `model.safetensors` are loaded into a `VarBuilder`.
- [ ] A sanity check reports any missing expected keys (embeddings, encoder layers, LM head).
- [ ] Tests run without network by using a tiny locally-generated safetensors file.

## Blocked by

- `docs/issues/01-upgrade-candle-model-downloader.md`
