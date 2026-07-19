# Issue 1: Upgrade Candle + model downloader

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

This first vertical slice prepares the `xenom-miner` to support a new ML backend. Upgrade the `candle` stack from 0.3 to a recent stable release (e.g. 0.8.x), add `candle-transformers` and `tokenizers`, and fix any API drift in the existing `CpuTrainer`/`MockTrainer`. Then implement a `model_hub` module that downloads `config.json`, `tokenizer.json`, and `model.safetensors` for a Hugging Face model id into the miner's local data directory (`data_dir/models/<safe_id>`). The downloader should skip re-downloading files that already exist and respect timeouts.

## Acceptance criteria

- [ ] `xenom-miner` compiles and tests pass after the Candle upgrade.
- [ ] Running `xenom-miner` with a model id downloads the three required files to the cache directory.
- [ ] Re-running does not re-download existing files.
- [ ] The existing `mock` and `cpu` trainers still work unchanged.

## Blocked by

None - can start immediately.
