//! Single-device trainer for the Mini Genome Model (MGM-1).
//!
//! Supports masked-language-model training, gradient extraction for FedAvg, and
//! multi-GPU data-parallel via `Mgm1MultiGpuTrainer`.

use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{D as TensorD, DType, Device, Tensor};
use candle_nn::VarMap;
use mini_genome_model::{DnaTokenizer, MiniGenomeConfig, MiniGenomeModel};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::info;

use crate::rpc::messages::{GenomeTrainingBatchMsg, GradientUpdate, TrainingBatch};
use crate::trainer::gradient::{build_gradient_update, gradient_commitment};
use crate::trainer::{DeviceInfo, DeviceType, ManualAdamW, Trainer, TrainingResult};

const MASK_TOKEN_ID: usize = 4;
const MASK_RATIO: f64 = 0.15;
pub(crate) const MAX_LEARNING_RATE: f32 = 1e-4;

/// Trainer for the `xenom/mgm-1` model.
pub struct Mgm1Trainer {
    model: MiniGenomeModel,
    varmap: Mutex<VarMap>,
    optimizer: Mutex<ManualAdamW>,
    config: MiniGenomeConfig,
    tokenizer: DnaTokenizer,
    device: Device,
    base_checkpoint: Mutex<[u8; 32]>,
    model_id: String,
    gradient_top_k_ratio: f32,
    learning_rate: f64,
}

impl Mgm1Trainer {
    /// Load or initialize an MGM-1 model from raw checkpoint files.
    ///
    /// `config` and `weights` are required; `tokenizer` is currently ignored
    /// because the DNA tokenizer mapping is fixed. `weights` must be a valid
    /// safetensors buffer; an empty `weights` vector initializes a fresh model.
    pub fn new(
        model_id: impl Into<String>,
        config_bytes: &[u8],
        _tokenizer_bytes: &[u8],
        weights: Vec<u8>,
        base_checkpoint: [u8; 32],
        device: Device,
        lr: f64,
        gradient_top_k_ratio: f32,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let config: MiniGenomeConfig =
            serde_json::from_slice(config_bytes).with_context(|| format!("Failed to parse MGM-1 config for {model_id}"))?;

        let varmap = Mutex::new(VarMap::new());
        let model = {
            let locked = varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
            let vb = candle_nn::VarBuilder::from_varmap(&locked, DType::F32, &device);
            MiniGenomeModel::new(vb, config.clone()).with_context(|| format!("Failed to build MGM-1 model {model_id}"))?
        };

        if !weights.is_empty() {
            load_varmap_weights(&varmap, &weights, &device)?;
        }

        let optimizer = ManualAdamW::new(lr);

        Ok(Self {
            model,
            varmap,
            optimizer: Mutex::new(optimizer),
            config,
            tokenizer: DnaTokenizer::new(),
            device,
            base_checkpoint: Mutex::new(base_checkpoint),
            model_id,
            gradient_top_k_ratio,
            learning_rate: lr,
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Tokenize a batch of DNA sequences, padding/truncating to `max_seq_len`.
    fn tokenize_batch(&self, sequences: &[String]) -> Vec<Vec<usize>> {
        sequences
            .iter()
            .map(|seq| {
                let mut tokens = self.tokenizer.encode(seq);
                tokens.truncate(self.config.max_seq_len);
                while tokens.len() < self.config.max_seq_len {
                    tokens.push(MASK_TOKEN_ID);
                }
                tokens
            })
            .collect()
    }

    /// Build masked-language-modeling input and label tensors from token IDs.
    fn build_mlm_tensors(&self, token_ids: &[Vec<usize>], rng: &mut ChaCha8Rng) -> Result<(Tensor, Tensor)> {
        let batch = token_ids.len();
        let seq_len = token_ids[0].len();
        let mut input_data = Vec::with_capacity(batch * seq_len);
        let mut label_data = Vec::with_capacity(batch * seq_len);

        for row in token_ids {
            for &tok in row {
                let mask = rng.gen_bool(MASK_RATIO);
                if mask {
                    input_data.push(MASK_TOKEN_ID as i64);
                    label_data.push(tok as i64);
                } else {
                    input_data.push(tok as i64);
                    label_data.push(tok as i64);
                }
            }
        }

        let input_ids = Tensor::new(input_data, &self.device)?.reshape((batch, seq_len))?;
        let labels = Tensor::new(label_data, &self.device)?.reshape((batch, seq_len))?;
        Ok((input_ids, labels))
    }

    /// Prepare a random synthetic batch on the trainer's device.
    pub(crate) fn prepare_random(&self, batch_size: usize, seed: [u8; 32]) -> Result<(Tensor, Tensor)> {
        let mut rng = ChaCha8Rng::from_seed(seed);
        let mut token_ids = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            let mut row = Vec::with_capacity(self.config.max_seq_len);
            for _ in 0..self.config.max_seq_len {
                row.push(rng.gen_range(0..self.config.vocab_size));
            }
            token_ids.push(row);
        }
        self.build_mlm_tensors(&token_ids, &mut rng)
    }

    /// Prepare a genome-backed batch on the trainer's device.
    pub(crate) fn prepare_sequences(&self, sequences: &[String], seed: [u8; 32]) -> Result<(Tensor, Tensor)> {
        let token_ids = self.tokenize_batch(sequences);
        let mut rng = ChaCha8Rng::from_seed(seed);
        self.build_mlm_tensors(&token_ids, &mut rng)
    }

    /// Compute smoothed inverse-frequency class weights from the masked positions
    /// of a batch.  The weights are clipped to [0.5, 2.0] so a small number of
    /// minority examples cannot dominate the gradient and collapse predictions to
    /// a single class.
    pub(crate) fn class_weights_for_batch(&self, input_ids: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let mask = input_ids.ne(labels)?.to_dtype(DType::F32)?;
        let labels_u32 = labels.to_dtype(DType::U32)?;
        let flat = labels_u32.reshape((labels.elem_count(),))?;
        let mask_flat = mask.reshape((labels.elem_count(),))?;

        let mut counts = vec![0.0f32; self.config.vocab_size];
        let label_vec = flat.to_vec1::<u32>()?;
        let mask_vec = mask_flat.to_vec1::<f32>()?;
        for (id, m) in label_vec.iter().zip(mask_vec.iter()) {
            let idx = *id as usize;
            if idx < self.config.vocab_size {
                counts[idx] += *m;
            }
        }

        let total_masked: f32 = counts.iter().take(4).sum::<f32>().max(1.0);
        let active_classes = 4usize;
        let smoothing = total_masked / active_classes as f32;
        let numerator = total_masked + active_classes as f32 * smoothing;

        let mut weights = vec![1.0f32; self.config.vocab_size];
        for i in 0..self.config.vocab_size {
            let effective = counts[i] + smoothing;
            let w = numerator / (active_classes as f32 * effective);
            weights[i] = w.clamp(0.5, 2.0);
        }

        Ok(Tensor::new(weights, &self.device)?)
    }

    /// Compute loss and accuracy for a batch without taking gradients.
    pub(crate) fn compute_loss_and_accuracy(&self, input_ids: &Tensor, labels: &Tensor) -> Result<(f64, f32)> {
        let class_weights = self.class_weights_for_batch(input_ids, labels).ok();
        let (loss, accuracy) = self.model.compute_mlm_loss(input_ids, labels, class_weights.as_ref())?;
        let loss_scalar = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        Ok((loss_scalar, accuracy))
    }

    /// Run a forward/backward pass and return the unscaled loss, accuracy, and
    /// per-variable gradients moved to the CPU (as F32).
    ///
    /// If `class_weights` is `None` they are computed from `(input_ids, labels)`
    /// so that a single-device trainer and a multi-device trainer can share the
    /// same per-batch weights and produce identical weight deltas.
    pub(crate) fn compute_gradients(
        &self,
        input_ids: &Tensor,
        labels: &Tensor,
        loss_scale: f32,
        class_weights: Option<&Tensor>,
    ) -> Result<(f64, f32, HashMap<String, Tensor>)> {
        let cw = class_weights
            .cloned()
            .or_else(|| self.class_weights_for_batch(input_ids, labels).ok());
        let (loss, accuracy) = self.model.compute_mlm_loss(input_ids, labels, cw.as_ref())?;
        let loss_scalar = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        if !loss_scalar.is_finite() {
            anyhow::bail!("Loss is not finite ({}) before backward", loss_scalar);
        }

        let scaled_loss = if (loss_scale - 1.0).abs() > f32::EPSILON { (&loss * (loss_scale as f64))? } else { loss };
        let scaled_loss_scalar = scaled_loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        if !scaled_loss_scalar.is_finite() {
            anyhow::bail!("Scaled loss is not finite ({}); loss scale {} is too large", scaled_loss_scalar, loss_scale);
        }

        let grads = scaled_loss.backward().context("Backward pass failed")?;
        let named_grads = self.grad_store_to_map(&grads)?;
        if named_grads.is_empty() {
            anyhow::bail!("Backward produced no named gradients; likely no trainable variables in the graph");
        }
        Ok((loss_scalar, accuracy, named_grads))
    }

    /// Apply named gradients to this trainer using its AdamW optimizer.
    pub(crate) fn apply_gradients(&self, named_grads: &HashMap<String, Tensor>, learning_rate: f32) -> Result<()> {
        let mut optimizer = self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {e}"))?;
        let effective_lr = learning_rate.min(MAX_LEARNING_RATE) as f64;
        optimizer.set_learning_rate(effective_lr);
        let varmap = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        optimizer.step(&varmap, named_grads).context("Optimizer step failed")?;
        Ok(())
    }

    fn grad_store_to_map(&self, grads: &candle_core::backprop::GradStore) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let locked = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        let data = locked.data().lock().map_err(|e: std::sync::PoisonError<_>| anyhow::anyhow!("VarMap data poisoned: {e}"))?;
        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                out.insert(name.clone(), grad.to_dtype(DType::F32)?);
            }
        }
        Ok(out)
    }

    /// Run one training step on an (input_ids, labels) pair.
    ///
    /// Returns the `TrainingResult` and, if requested, the weight-space delta that
    /// will be sent to the seed-node for FedAvg aggregation.
    fn train_step(
        &self,
        input_ids: &Tensor,
        labels: &Tensor,
        batch_indices: Vec<u64>,
        learning_rate: f32,
        return_update: bool,
    ) -> Result<(TrainingResult, Option<HashMap<String, Tensor>>)> {
        let start = Instant::now();
        let effective_lr = learning_rate.min(MAX_LEARNING_RATE);

        // Snapshot the base weights so we can (a) compute the weight-space delta and
        // (b) restore the model after producing the update. In gradient-update mode the
        // trainer must always return to the shared base checkpoint; in plain `train` mode
        // the updated weights are kept.
        let base_weights = self.varmap_snapshot()?;

        if return_update {
            // Reset Adam moment estimates so the local weight delta is computed from the
            // same clean state the node will use during validation.
            self.reset_optimizer()?;
        }

        let (loss_before, accuracy_before, grads) = self.compute_gradients(input_ids, labels, 1.0, None)?;
        let grad_norm = gradient_norm(&grads)?;
        let label_dist = masked_label_distribution(input_ids, labels)?;

        self.apply_gradients(&grads, learning_rate)?;
        let updated_weights = self.varmap_snapshot()?;
        let weight_delta = if return_update {
            Some(Self::compute_weight_delta(&base_weights, &updated_weights)?)
        } else {
            None
        };

        let (loss_after, accuracy_after) = self.compute_loss_and_accuracy(input_ids, labels)?;
        let logits = self.model.forward(input_ids)?;
        let pred_dist = masked_prediction_distribution(input_ids, labels, &logits)?;

        // Commitment is over the payload that will be sent to the seed-node.
        let gradients_commitment = gradient_commitment(weight_delta.as_ref().unwrap_or(&grads))?;
        info!(
            "MGM-1 block: lr={:.3e} loss={:.4} -> {:.4} acc={:.2}% -> {:.2}% grad_norm={:.4}\n  labels A={:>3} C={:>3} G={:>3} T={:>3}\n  preds  A={:>3} C={:>3} G={:>3} T={:>3}",
            effective_lr, loss_before, loss_after,
            accuracy_before * 100.0, accuracy_after * 100.0, grad_norm,
            label_dist[0], label_dist[1], label_dist[2], label_dist[3],
            pred_dist[0], pred_dist[1], pred_dist[2], pred_dist[3],
        );

        if return_update {
            self.restore_varmap(&base_weights)?;
        }

        let result = TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices,
            base_checkpoint: *self.base_checkpoint.lock().unwrap(),
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        };
        Ok((result, weight_delta))
    }

    /// Reset optimizer moment estimates (used when the base checkpoint changes).
    pub(crate) fn reset_optimizer(&self) -> Result<()> {
        self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {e}"))?.reset()?;
        Ok(())
    }

    /// Load trainable weights from an in-memory safetensors buffer.
    pub(crate) fn load_weights_from_bytes(&self, weights: &[u8]) -> Result<()> {
        let loaded =
            candle_core::safetensors::load_buffer(weights, &self.device).context("Failed to load MGM-1 safetensors buffer")?;
        let locked = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        let data = locked.data().lock().map_err(|e: std::sync::PoisonError<_>| anyhow::anyhow!("VarMap data poisoned: {e}"))?;
        for (name, var) in data.iter() {
            let loaded_var = loaded.get(name).ok_or_else(|| anyhow::anyhow!("Missing weight {name} in checkpoint"))?;
            let loaded_var = loaded_var.to_device(&self.device)?.to_dtype(var.as_tensor().dtype())?;
            var.set(&loaded_var)?;
        }
        Ok(())
    }

    /// Snapshot the current trainable weights as a deep copy in F32.
    pub(crate) fn varmap_snapshot(&self) -> Result<HashMap<String, Tensor>> {
        let locked = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        let data = locked.data().lock().map_err(|e: std::sync::PoisonError<_>| anyhow::anyhow!("VarMap data poisoned: {e}"))?;
        let mut snap = HashMap::new();
        for (name, var) in data.iter() {
            let t = var.as_tensor();
            let t_f32 = t.to_dtype(DType::F32)?.copy().context("Failed to copy snapshot tensor")?;
            snap.insert(name.clone(), t_f32);
        }
        Ok(snap)
    }

    /// Compute `updated - base` for a pair of weight snapshots.
    /// Used to turn a local AdamW step into the weight-space delta that is sent
    /// to the seed-node for FedAvg aggregation.
    pub(crate) fn compute_weight_delta(
        base: &HashMap<String, Tensor>,
        updated: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut delta = HashMap::with_capacity(base.len());
        for (name, base_t) in base {
            let updated_t = updated.get(name).ok_or_else(|| anyhow::anyhow!("Updated weights missing variable {}", name))?;
            delta.insert(name.clone(), (updated_t.sub(base_t))?);
        }
        Ok(delta)
    }

    /// Restore the trainable weights from a snapshot.
    pub(crate) fn restore_varmap(&self, snapshot: &HashMap<String, Tensor>) -> Result<()> {
        let locked = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        let data = locked.data().lock().map_err(|e: std::sync::PoisonError<_>| anyhow::anyhow!("VarMap data poisoned: {e}"))?;
        for (name, var) in data.iter() {
            if let Some(snap) = snapshot.get(name) {
                let t = var.as_tensor();
                let snap = snap.to_dtype(t.dtype())?.to_device(t.device())?;
                var.set(&snap)?;
            }
        }
        Ok(())
    }
}

impl Trainer for Mgm1Trainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.prepare_random(n, seed)?;
        self.train_step(&input_ids, &labels, batch.data_indices.clone(), batch.learning_rate, false).map(|(r, _)| r)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.prepare_random(n, seed)?;
        let participant_weight = (input_ids.dim(0)? * input_ids.dim(1)?) as f32;

        let (mut result, weight_delta) =
            self.train_step(&input_ids, &labels, batch.data_indices.clone(), batch.learning_rate, true)?;
        let weight_delta = weight_delta.ok_or_else(|| anyhow::anyhow!("Weight delta was not produced"))?;
        let update = build_gradient_update(
            &self.model_id,
            batch.base_checkpoint,
            weight_delta,
            participant_weight,
            self.gradient_top_k_ratio,
            &result,
            batch.batch_id,
            batch.learning_rate,
            [0u8; 32],
            Vec::new(),
        )?;
        result.gradients_commitment = update.gradients_commitment;
        Ok((result, Some(update)))
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        if msg.sequences.is_empty() {
            anyhow::bail!("Genome batch contains no sequences");
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&msg.base_checkpoint);
        seed[..8].copy_from_slice(&msg.batch.batch_id.to_le_bytes());
        let batch_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();

        let (input_ids, labels) = self.prepare_sequences(&msg.sequences, seed)?;
        self.train_step(&input_ids, &labels, batch_indices, self.learning_rate as f32, false).map(|(r, _)| r)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        if msg.sequences.is_empty() {
            anyhow::bail!("Genome batch contains no sequences");
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&msg.base_checkpoint);
        seed[..8].copy_from_slice(&msg.batch.batch_id.to_le_bytes());
        let batch_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();

        let (input_ids, labels) = self.prepare_sequences(&msg.sequences, seed)?;
        let participant_weight = (input_ids.dim(0)? * input_ids.dim(1)?) as f32;

        let (mut result, weight_delta) =
            self.train_step(&input_ids, &labels, batch_indices, self.learning_rate as f32, true)?;
        let weight_delta = weight_delta.ok_or_else(|| anyhow::anyhow!("Weight delta was not produced"))?;
        let update = build_gradient_update(
            &self.model_id,
            msg.base_checkpoint,
            weight_delta,
            participant_weight,
            self.gradient_top_k_ratio,
            &result,
            msg.batch.batch_id,
            self.learning_rate as f32,
            msg.batch.genome_merkle_root,
            msg.batch.data_indices.clone(),
        )?;
        result.gradients_commitment = update.gradients_commitment;
        Ok((result, Some(update)))
    }

    fn device_info(&self) -> DeviceInfo {
        let device_type = match self.device {
            Device::Cpu => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Metal(_) => DeviceType::Metal,
        };
        DeviceInfo { device_type, name: format!("MGM-1 on {:?}", self.device), threads: 1, ..Default::default() }
    }

    fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        Some(*self.base_checkpoint.lock().unwrap())
    }

    fn load_base_checkpoint(&self, base_checkpoint: [u8; 32], weights: &[u8]) -> Result<()> {
        self.load_weights_from_bytes(weights)?;
        self.reset_optimizer()?;
        *self.base_checkpoint.lock().unwrap() = base_checkpoint;
        Ok(())
    }
}

/// Compute the L2 norm of a collection of named gradients.
pub(crate) fn gradient_norm(named_grads: &HashMap<String, Tensor>) -> Result<f64> {
    let mut sum_sq = 0.0f64;
    for grad in named_grads.values() {
        let t = grad.to_dtype(DType::F32)?;
        let v = t.sqr()?.sum_all()?.to_scalar::<f32>()? as f64;
        sum_sq += v;
    }
    Ok(sum_sq.sqrt())
}

/// Count how many masked labels belong to each DNA base (A, C, G, T).
pub(crate) fn masked_label_distribution(input_ids: &Tensor, labels: &Tensor) -> Result<[usize; 4]> {
    let mask = input_ids.ne(labels)?.to_dtype(DType::F32)?;
    let labels_u32 = labels.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let mask_f = mask.to_vec2::<f32>()?;
    let mut dist = [0usize; 4];
    for b in 0..labels_u32.len() {
        for t in 0..labels_u32[b].len() {
            if mask_f[b][t] > 0.5 {
                let id = labels_u32[b][t] as usize;
                if id < 4 {
                    dist[id] += 1;
                }
            }
        }
    }
    Ok(dist)
}

/// Count how many masked positions are predicted as each DNA base (A, C, G, T).
pub(crate) fn masked_prediction_distribution(input_ids: &Tensor, labels: &Tensor, logits: &Tensor) -> Result<[usize; 4]> {
    let mask = input_ids.ne(labels)?.to_dtype(DType::F32)?;
    let logits_dna = logits.narrow(TensorD::Minus1, 0, 4)?;
    let pred_ids = logits_dna.argmax(TensorD::Minus1)?.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let mask_f = mask.to_vec2::<f32>()?;
    let mut dist = [0usize; 4];
    for b in 0..pred_ids.len() {
        for t in 0..pred_ids[b].len() {
            if mask_f[b][t] > 0.5 {
                let id = pred_ids[b][t] as usize;
                if id < 4 {
                    dist[id] += 1;
                }
            }
        }
    }
    Ok(dist)
}

/// Load safetensor weights into an existing `VarMap`.
fn load_varmap_weights(varmap: &Mutex<VarMap>, weights: &[u8], _device: &Device) -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("mgm1_load_{}.safetensors", rand::random::<u64>()));
    fs::write(&tmp, weights)?;
    let result = {
        let mut locked = varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        locked.load(&tmp).map_err(|e| anyhow::anyhow!("Failed to load MGM-1 weights: {}", e))
    };
    let _ = fs::remove_file(&tmp);
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::messages::{GenomeSlice, GenomeTrainingBatch, GenomeTrainingBatchMsg};

    fn default_config() -> Vec<u8> {
        serde_json::to_vec(&MiniGenomeConfig::default()).unwrap()
    }

    #[test]
    fn test_mgm1_trainer_loads_and_trains() {
        let config = default_config();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, 1.0).unwrap();

        let batch = TrainingBatch {
            batch_id: 1,
            model_id: "xeno/mgm-1".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: (0..4).collect(),
            target_improvement: 0.01,
            learning_rate: 0.01,
        };
        let result = trainer.train(&batch).unwrap();
        assert!(result.loss_before.is_finite());
        assert!(result.loss_after.is_finite());
    }

    #[test]
    fn test_mgm1_trainer_extracts_gradient_update() {
        let config = default_config();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, 1.0).unwrap();

        let batch = TrainingBatch {
            batch_id: 1,
            model_id: "xeno/mgm-1".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: (0..4).collect(),
            target_improvement: 0.01,
            learning_rate: 0.01,
        };
        let (result, update) = trainer.train_with_gradients(&batch).unwrap();
        assert!(result.loss_before.is_finite());
        assert!(result.loss_after.is_finite());
        assert!(
            result.loss_after < result.loss_before,
            "MGM-1 training should reduce loss: {} -> {}",
            result.loss_before,
            result.loss_after
        );
        assert!(update.is_some(), "Expected a GradientUpdate for FedAvg");
        let update = update.unwrap();
        assert_eq!(update.model_id, "xeno/mgm-1");
        assert_eq!(update.base_checkpoint, [0u8; 32]);
        assert!(!update.encrypted_payload.is_empty());
    }

    #[test]
    fn test_mgm1_trainer_genome_batch() {
        let config = default_config();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, 1.0).unwrap();

        let sequences = vec!["ACGTACGTACGT".to_string(), "TGCATGCATGCA".to_string(), "AAAACCCCGGGGTTTT".to_string()];
        let batch = GenomeTrainingBatch {
            batch_id: 1,
            model_id: "xeno/mgm-1".to_string(),
            genome_merkle_root: [0u8; 32],
            data_indices: sequences
                .iter()
                .enumerate()
                .map(|(i, _)| GenomeSlice { chunk_idx: i as u64, start_base: 0, length: 12 })
                .collect(),
            mask_ratio: 0.15,
            seq_length: 12,
        };
        let msg = GenomeTrainingBatchMsg { batch, sequences, base_checkpoint: [0u8; 32] };
        let (result, update) = trainer.train_genome_with_gradients(&msg).unwrap();
        assert!(result.loss_before.is_finite());
        assert!(result.loss_after.is_finite());
        assert!(
            result.loss_after < result.loss_before,
            "MGM-1 training should reduce loss: {} -> {}",
            result.loss_before,
            result.loss_after
        );
        assert!(update.is_some());
    }

    #[test]
    #[ignore = "diagnostic helper; run manually with -- --ignored --nocapture"]
    fn diagnose_mgm1_first_batch() {
        use candle_nn::ops::softmax;
        use candle_core::D;

        let config_json = serde_json::json!({
            "vocab_size": 8,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "d_ff": 512,
            "max_seq_len": 64,
            "dropout": 0.1,
        });
        let config_bytes = config_json.to_string().into_bytes();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config_bytes, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, 1.0).unwrap();

        // Balanced synthetic batch (already tested) plus an imbalanced batch
        let sequences = vec![
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCAA".to_string(),
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCAA".to_string(),
            "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTTCCCCCCCCCCCCAA".to_string(),
        ];
        let msg = GenomeTrainingBatchMsg {
            batch: GenomeTrainingBatch {
                batch_id: 1,
                model_id: "xeno/mgm-1".to_string(),
                genome_merkle_root: [0u8; 32],
                data_indices: sequences.iter().enumerate().map(|(i, s)| GenomeSlice { chunk_idx: i as u64, start_base: 0, length: s.len() as u32 }).collect(),
                mask_ratio: 0.15,
                seq_length: 64,
            },
            sequences,
            base_checkpoint: [0u8; 32],
        };

        let (input_ids, labels) = trainer.prepare_sequences(&msg.sequences, [0u8; 32]).unwrap();
        let input_vec = input_ids.to_vec2::<i64>().unwrap();
        let label_vec = labels.to_vec2::<i64>().unwrap();

        let mut label_counts = HashMap::<i64, usize>::new();
        let mut mask_count = 0usize;
        for b in 0..input_vec.len() {
            for t in 0..input_vec[b].len() {
                if input_vec[b][t] != label_vec[b][t] {
                    mask_count += 1;
                    *label_counts.entry(label_vec[b][t]).or_insert(0) += 1;
                }
            }
        }
        println!("\n[DIAGNOSE] Masked positions: {}", mask_count);
        println!("[DIAGNOSE] Label distribution among masked positions:");
        let base_name = |id: i64| -> char {
            match id {
                0 => 'A', 1 => 'C', 2 => 'G', 3 => 'T', 4 => '[', 5 => ' ', 6 => ']', 7 => '|', _ => '?',
            }
        };
        let mut counts: Vec<_> = label_counts.iter().collect();
        counts.sort_by(|a, b| a.0.cmp(b.0));
        for (id, c) in counts {
            println!("  {} (id={}): {} ({:.1}%)", base_name(*id), id, c, 100.0 * (*c as f32) / mask_count.max(1) as f32);
        }

        let logits = trainer.forward(&input_ids).unwrap();
        let logits_4 = logits.narrow(D::Minus1, 0, 4).unwrap();
        let probs = softmax(&logits_4, D::Minus1).unwrap();
        let probs_vec = probs.to_vec3::<f32>().unwrap();
        let pred_ids = logits_4.argmax(D::Minus1).unwrap().to_vec2::<u32>().unwrap();

        println!("\n[DIAGNOSE] Top-4 logits / probabilities for first 20 masked positions (fresh model):");
        let mut printed = 0usize;
        'outer: for b in 0..input_vec.len() {
            for t in 0..input_vec[b].len() {
                if input_vec[b][t] != label_vec[b][t] {
                    let true_id = label_vec[b][t];
                    println!(
                        "  batch[{}][{}] true={} pred={} | A={:.3} C={:.3} G={:.3} T={:.3}",
                        b, t, base_name(true_id), base_name(pred_ids[b][t] as i64),
                        probs_vec[b][t][0], probs_vec[b][t][1], probs_vec[b][t][2], probs_vec[b][t][3]
                    );
                    printed += 1;
                    if printed >= 20 {
                        break 'outer;
                    }
                }
            }
        }

        let mut all_predictions = HashMap::<char, usize>::new();
        for b in 0..input_vec.len() {
            for t in 0..input_vec[b].len() {
                if input_vec[b][t] != label_vec[b][t] {
                    let pred = base_name(pred_ids[b][t] as i64);
                    *all_predictions.entry(pred).or_insert(0) += 1;
                }
            }
        }
        println!("\n[DIAGNOSE] Prediction distribution on fresh model masked positions:");
        for c in ['A', 'C', 'G', 'T'] {
            let n = all_predictions.get(&c).copied().unwrap_or(0);
            println!("  {}: {} ({:.1}%)", c, n, 100.0 * (n as f32) / mask_count.max(1) as f32);
        }

        // Inspect class-weights that the trainer would use for this batch.
        let class_weights = trainer.class_weights_for_batch(&input_ids, &labels).unwrap();
        let cw_vec = class_weights.to_vec1::<f32>().unwrap();
        println!("\n[DIAGNOSE] Class weights for this batch: A={:.3} C={:.3} G={:.3} T={:.3}", cw_vec[0], cw_vec[1], cw_vec[2], cw_vec[3]);

        // Train for a few steps on the same batch and observe whether loss decreases
        // and whether the prediction distribution stays balanced or collapses.
        println!("\n[DIAGNOSE] Training on same batch for 50 steps (no class weights)...");
        for step in 0..50 {
            let (loss, _) = trainer.model.compute_mlm_loss(&input_ids, &labels, None).unwrap();
            let loss_scalar = loss.to_dtype(DType::F32).unwrap().to_vec0::<f32>().unwrap();
            let grads = loss.backward().unwrap();
            let named_grads = trainer.grad_store_to_map(&grads).unwrap();
            trainer.apply_gradients(&named_grads, 1e-4).unwrap();
            if step % 10 == 0 {
                println!("  step {:>3}: loss = {:.6}", step, loss_scalar);
            }
        }

        let logits_after = trainer.forward(&input_ids).unwrap();
        let logits_4_after = logits_after.narrow(D::Minus1, 0, 4).unwrap();
        let pred_ids_after = logits_4_after.argmax(D::Minus1).unwrap().to_vec2::<u32>().unwrap();
        let mut predictions_after = HashMap::<char, usize>::new();
        let mut correct = 0usize;
        for b in 0..input_vec.len() {
            for t in 0..input_vec[b].len() {
                if input_vec[b][t] != label_vec[b][t] {
                    let pred = base_name(pred_ids_after[b][t] as i64);
                    *predictions_after.entry(pred).or_insert(0) += 1;
                    if pred == base_name(label_vec[b][t]) {
                        correct += 1;
                    }
                }
            }
        }
        println!("\n[DIAGNOSE] Prediction distribution after 50 steps (same batch):");
        for c in ['A', 'C', 'G', 'T'] {
            let n = predictions_after.get(&c).copied().unwrap_or(0);
            println!("  {}: {} ({:.1}%)", c, n, 100.0 * (n as f32) / mask_count.max(1) as f32);
        }
        println!("[DIAGNOSE] Accuracy after 50 steps: {} / {} ({:.1}%)", correct, mask_count, 100.0 * (correct as f32) / mask_count.max(1) as f32);

        let (final_loss, final_acc) = trainer.model.compute_mlm_loss(&input_ids, &labels, None).unwrap();
        println!(
            "[DIAGNOSE] Final loss = {:.6}, final accuracy = {:.2}%",
            final_loss.to_dtype(DType::F32).unwrap().to_vec0::<f32>().unwrap(),
            final_acc * 100.0
        );
    }
}
