# Issue 6: CLI `--trainer` selection and seed-node checkpoint

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Add a `--trainer` CLI argument to `xenom-miner` with values `mock` (default for devnet), `cpu`, and `dnabert2`. Wire the main loop to instantiate the correct trainer. Also update the seed-node `GetTrainingBatch` WebSocket handler so it returns the real `weights_hash` from the downloaded model as `base_checkpoint` once the model is available. While the model is still downloading, it may return zeros for backward compatibility.

## Acceptance criteria

- [ ] `--trainer=dnabert2` starts the real DNABERT-2 trainer.
- [ ] Default devnet remains `mock` so low-resource hosts keep working.
- [ ] Seed-node returns non-zero `base_checkpoint` after the model download completes.
- [ ] `AGENTS.md` and `README-SCRIPTS.md` document the new flag.

## Blocked by

- `docs/issues/05-dnabert2-training-step.md`
