# Issue 1: Upgrade Candle + model client for seed-node checkpoint

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

This first vertical slice prepares the `xenom-miner` to support the DNABERT-2 training backend. Upgrade the `candle` stack from 0.3 to a recent stable release (e.g. 0.8.x), add `candle-transformers` and `tokenizers`, and fix any API drift in the existing `CpuTrainer`/`MockTrainer`. Instead of downloading from Hugging Face, implement a `model_client` module that fetches the model checkpoint (`config.json`, `tokenizer.json`, `model.safetensors`) from the seed-node over the existing WebSocket connection using a new `GetModelCheckpoint` request. The miner keeps the received bytes in memory only; it must not persist model weights to disk.

## Acceptance criteria

- [ ] `xenom-miner` compiles and tests pass after the Candle upgrade.
- [ ] Existing `mock` and `cpu` trainers continue to work.
- [ ] `model_client` can send `GetModelCheckpoint { model_id }` and receive a `ModelCheckpoint` response from the seed-node.
- [ ] Received checkpoint bytes are kept in memory and can be passed to a loader.

## Blocked by

None - can start immediately.
