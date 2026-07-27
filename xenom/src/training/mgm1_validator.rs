//! Full re-execution validator for MGM-1 gradient updates.
//!
//! The full node does not trust the miner's reported `loss_before`/`loss_after`.
//! For every genome-backed MGM-1 `GradientUpdate`, it loads the same genome
//! archive, extracts the exact slices, instantiates the same `Mgm1Trainer`,
//! and re-runs `train_genome_with_gradients` on the CPU. The update is only
//! accepted if the recomputed losses match the claimed ones (within a small
//! tolerance).

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use seed_node::genome::{GenomeSlice as NodeGenomeSlice, GenomeStorage, GenomeTrainingBatch};
use seed_node::model::RawModelFiles;
use seed_node::rpc::messages::GradientUpdate;
use xenom_miner::rpc::messages::{GenomeSlice as MinerGenomeSlice, GenomeTrainingBatchMsg};
use xenom_miner::trainer::{Mgm1Trainer, Trainer};

/// Tolerance for comparing floating-point loss values between miner and validator.
const LOSS_TOLERANCE: f64 = 1e-3;

/// Re-run the training step that produced `update` and verify the reported losses.
///
/// `files` must contain the raw MGM-1 checkpoint (config JSON + safetensors weights)
/// that corresponds to `update.base_checkpoint`. `genome_storage` is used to load the
/// genome archive identified by `update.genome_merkle_root`.
pub async fn validate_mgm1_update(
    update: &GradientUpdate,
    files: &RawModelFiles,
    genome_storage: &Arc<tokio::sync::RwLock<GenomeStorage>>,
) -> Result<()> {
    if update.genome_slices.is_empty() {
        bail!("MGM-1 gradient update has no genome slices; cannot re-execute training");
    }

    let archive = {
        let mut storage = genome_storage.write().await;
        storage
            .get_or_load(update.genome_merkle_root, "")
            .await
            .with_context(|| format!("Failed to load genome archive for validation: {}", hex::encode(update.genome_merkle_root)))?
    };

    let sequences: Vec<String> = update
        .genome_slices
        .iter()
        .filter_map(|slice| {
            archive
                .extract_sequence(slice.chunk_idx, slice.start_base, slice.length)
                .map_err(|e| {
                    tracing::warn!("Failed to extract genome slice {:?}: {}", slice, e);
                    e
                })
                .ok()
        })
        .collect();

    if sequences.is_empty() {
        bail!("No DNA sequences could be extracted for validation");
    }

    // CPU-bound validation runs on the blocking pool so it does not stall the
    // async runtime.
    let update = update.clone();
    let files = files.clone();
    tokio::task::spawn_blocking(move || validate_on_cpu(&update, &files, sequences))
        .await
        .map_err(|e| anyhow::anyhow!("MGM-1 validation task panicked: {}", e))?
}

fn validate_on_cpu(update: &GradientUpdate, files: &RawModelFiles, sequences: Vec<String>) -> Result<()> {
    // The miner ignores the tokenizer bytes and uses the fixed DnaTokenizer, so we
    // pass an empty tokenizer buffer here as well.
    let trainer = Mgm1Trainer::new(
        &update.model_id,
        &files.config,
        &files.tokenizer,
        files.weights.clone(),
        update.base_checkpoint,
        Device::Cpu,
        update.learning_rate as f64,
        1.0,
    )
    .with_context(|| format!("Failed to build MGM-1 validator for {}", update.model_id))?;

    let batch = GenomeTrainingBatch {
        batch_id: update.batch_id,
        model_id: update.model_id.clone(),
        genome_merkle_root: update.genome_merkle_root,
        data_indices: update.genome_slices.clone(),
        mask_ratio: 0.15,
        seq_length: 512,
    };

    let msg = GenomeTrainingBatchMsg { batch: convert_batch(batch), sequences, base_checkpoint: update.base_checkpoint };

    let (recomputed, _) = trainer.train_genome_with_gradients(&msg).with_context(|| "MGM-1 validation training step failed")?;

    if !approx_eq(update.loss_before, recomputed.loss_before, LOSS_TOLERANCE) {
        bail!(
            "MGM-1 validation failed: claimed loss_before {} != recomputed {} (tolerance {})",
            update.loss_before,
            recomputed.loss_before,
            LOSS_TOLERANCE
        );
    }

    if !approx_eq(update.loss_after, recomputed.loss_after, LOSS_TOLERANCE) {
        bail!(
            "MGM-1 validation failed: claimed loss_after {} != recomputed {} (tolerance {})",
            update.loss_after,
            recomputed.loss_after,
            LOSS_TOLERANCE
        );
    }

    Ok(())
}

fn convert_batch(batch: GenomeTrainingBatch) -> xenom_miner::rpc::messages::GenomeTrainingBatch {
    xenom_miner::rpc::messages::GenomeTrainingBatch {
        batch_id: batch.batch_id,
        model_id: batch.model_id,
        genome_merkle_root: batch.genome_merkle_root,
        data_indices: batch.data_indices.into_iter().map(convert_slice).collect(),
        mask_ratio: batch.mask_ratio,
        seq_length: batch.seq_length,
    }
}

fn convert_slice(slice: NodeGenomeSlice) -> MinerGenomeSlice {
    MinerGenomeSlice { chunk_idx: slice.chunk_idx, start_base: slice.start_base, length: slice.length }
}

fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= tolerance
}
