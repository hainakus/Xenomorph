# Issue 7: End-to-end devnet test with real training

## Parent

PRD: `docs/PRD-DNABERT2-CANDLE.md`

## What to build

Run a full devnet with `xenom-miner` configured to use `--trainer=dnabert2` on hardware capable of holding the 110M model in memory and running one MLM step per block. Validate that the miner connects to `xeno-seed`, downloads the model, receives training batches, performs real training, and submits blocks. Because this requires significant compute, it is marked HITL and should be scheduled/coordinated with the team.

## Acceptance criteria

- [ ] The devnet starts with `--trainer=dnabert2` and all containers are healthy.
- [ ] Miner logs show real `loss_before`/`loss_after` values.
- [ ] Blocks are submitted and accepted by the seed-node.
- [ ] Performance metrics (time per block, memory) are documented.

## Blocked by

- `docs/issues/06-cli-trainer-seed-checkpoint.md`
