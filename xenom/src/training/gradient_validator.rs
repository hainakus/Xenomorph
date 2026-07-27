//! Generic gradient-update validation framework.
//!
//! The full node does not blindly trust the miner's reported metrics. For model
//! families where re-execution is feasible (currently `xeno/mgm-1`), the validator
//! reloads the exact genome slices and checkpoint and re-runs the training step,
//! comparing the recomputed loss with the values claimed in the `GradientUpdate`.
//! Unknown or unsupported models are accepted by default, leaving commitment and
//! proof-of-work checks to the caller.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use candle_core::Device;
use seed_node::genome::{GenomeSlice as NodeGenomeSlice, GenomeStorage, GenomeTrainingBatch};
use seed_node::model::RawModelFiles;
use seed_node::rpc::messages::GradientUpdate;
use xenom_miner::rpc::messages::{GenomeSlice as MinerGenomeSlice, GenomeTrainingBatchMsg};
use xenom_miner::trainer::{Mgm1Trainer, Trainer};

/// Tolerance for comparing floating-point loss values between miner and validator.
const LOSS_TOLERANCE: f64 = 1e-3;

/// Interface implemented by every model-specific validator.
#[async_trait]
pub trait ModelValidator: Send + Sync {
    /// Return `true` if this validator knows how to re-execute the given update.
    fn can_validate(&self, update: &GradientUpdate) -> bool;

    /// Re-execute the training step described by `update` and verify the reported
    /// metrics.
    async fn validate(
        &self,
        update: &GradientUpdate,
        files: &RawModelFiles,
        genome_storage: Arc<tokio::sync::RwLock<GenomeStorage>>,
    ) -> Result<()>;
}

/// Model-agnostic dispatcher. It routes each `GradientUpdate` to the first
/// `ModelValidator` that accepts it.
pub struct GradientValidator {
    validators: Vec<Box<dyn ModelValidator>>,
}

impl GradientValidator {
    /// Build the default validator set (currently only `xeno/mgm-1`).
    pub fn new() -> Self {
        Self { validators: vec![Box::new(Mgm1ModelValidator)] }
    }

    /// Validate `update` using the first validator that claims it, or accept it
    /// if no validator is registered for this model type.
    pub async fn validate(
        &self,
        update: &GradientUpdate,
        files: &RawModelFiles,
        genome_storage: Arc<tokio::sync::RwLock<GenomeStorage>>,
    ) -> Result<()> {
        for validator in &self.validators {
            if validator.can_validate(update) {
                return validator.validate(update, files, genome_storage).await;
            }
        }
        tracing::debug!("No re-execution validator for model {}; accepting metrics", update.model_id);
        Ok(())
    }
}

impl Default for GradientValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Full re-execution validator for the `xeno/mgm-1` model.
pub struct Mgm1ModelValidator;

impl Mgm1ModelValidator {
    fn model_ids() -> &'static [&'static str] {
        &["xeno/mgm-1", "xenom/mgm-1"]
    }
}

#[async_trait]
impl ModelValidator for Mgm1ModelValidator {
    fn can_validate(&self, update: &GradientUpdate) -> bool {
        Self::model_ids().contains(&update.model_id.as_str())
    }

    async fn validate(
        &self,
        update: &GradientUpdate,
        files: &RawModelFiles,
        genome_storage: Arc<tokio::sync::RwLock<GenomeStorage>>,
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
        tokio::task::spawn_blocking(move || validate_mgm1_on_cpu(&update, &files, sequences))
            .await
            .map_err(|e| anyhow::anyhow!("MGM-1 validation task panicked: {}", e))?
    }
}

fn validate_mgm1_on_cpu(update: &GradientUpdate, files: &RawModelFiles, sequences: Vec<String>) -> Result<()> {
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

    let msg = GenomeTrainingBatchMsg {
        batch: convert_batch(batch),
        sequences,
        base_checkpoint: update.base_checkpoint,
    };

    let (recomputed, _) = trainer
        .train_genome_with_gradients(&msg)
        .with_context(|| "MGM-1 validation training step failed")?;

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
