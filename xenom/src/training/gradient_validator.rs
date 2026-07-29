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
use borsh::from_slice as borsh_from_slice;
use candle_core::Device;
use seed_node::genome::GenomeStorage;
use seed_node::model::RawModelFiles;
use seed_node::rpc::messages::GradientUpdate;
use xenom_miner::rpc::messages::GradientPayload;
use xenom_miner::trainer::gradient::{gradient_commitment, reconstruct_from_layers};
use xenom_miner::trainer::{Mgm1Trainer, MultiGpuConfig};

/// Tolerance for comparing floating-point loss values between miner and validator.
/// Multi-GPU forward/backward and cross-device tensor movement introduce
/// small numerical differences, so a 5e-3 tolerance is used while still
/// preventing miners from claiming arbitrary loss improvements.
const LOSS_TOLERANCE: f64 = 5e-3;

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
        1e-4,
        &MultiGpuConfig::default(),
    )
    .with_context(|| format!("Failed to build MGM-1 validator for {}", update.model_id))?;

    // Build the same random seed the miner uses: base checkpoint with the first
    // 8 bytes overwritten by the batch id.
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&update.base_checkpoint);
    seed[..8].copy_from_slice(&update.batch_id.to_le_bytes());

    let mask_ratio = 0.15f64;
    let seq_length = 512usize;
    let (input_ids, labels) = trainer
        .prepare_sequences(&sequences, seed, mask_ratio, seq_length)
        .with_context(|| "MGM-1 validator failed to prepare genome sequences")?;

    // Loss on the base checkpoint: a single forward pass, so CPU/GPU drift is small.
    let (loss_before, _) =
        trainer.compute_loss_and_accuracy(&input_ids, &labels).with_context(|| "MGM-1 validator failed to compute loss_before")?;
    if !approx_eq(update.loss_before, loss_before, LOSS_TOLERANCE) {
        bail!(
            "MGM-1 validation failed: claimed loss_before {} != recomputed {} (tolerance {})",
            update.loss_before,
            loss_before,
            LOSS_TOLERANCE
        );
    }

    // Decrypt and reconstruct the weight-space delta, then verify the commitment.
    let payload_bytes = model_crypto::decrypt(&update.encrypted_payload, &model_crypto::derive_encryption_key())
        .context("Failed to decrypt MGM-1 gradient payload")?;
    let payload: GradientPayload = borsh_from_slice(&payload_bytes).context("Failed to deserialize MGM-1 gradient payload")?;
    let weight_delta = reconstruct_from_layers(&payload.layer_gradients).context("Failed to reconstruct MGM-1 gradient layers")?;
    let reconstructed_commitment =
        gradient_commitment(&weight_delta).context("Failed to compute gradient commitment over reconstructed payload")?;
    if reconstructed_commitment != update.gradients_commitment {
        bail!(
            "MGM-1 validation failed: payload commitment mismatch {:?} != {:?}",
            reconstructed_commitment,
            update.gradients_commitment
        );
    }

    // Apply the exact delta the seed-node will apply and measure the resulting loss.
    // This removes the cross-device drift that made re-running 8+ local AdamW steps
    // on CPU fail against a Metal/CUDA miner.
    trainer.apply_weight_delta(&weight_delta).context("MGM-1 validator failed to apply weight delta")?;
    let (loss_after, _) =
        trainer.compute_loss_and_accuracy(&input_ids, &labels).with_context(|| "MGM-1 validator failed to compute loss_after")?;
    if !approx_eq(update.loss_after, loss_after, LOSS_TOLERANCE) {
        bail!(
            "MGM-1 validation failed: claimed loss_after {} != recomputed {} (tolerance {})",
            update.loss_after,
            loss_after,
            LOSS_TOLERANCE
        );
    }

    Ok(())
}

fn approx_eq(a: f64, b: f64, tolerance: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    (a - b).abs() <= tolerance
}
