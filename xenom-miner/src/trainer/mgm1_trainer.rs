//! Single-device trainer for the Mini Genome Model (MGM-1).
//!
//! Supports masked-language-model training, gradient extraction for FedAvg, and
//! multi-GPU data-parallel via `Mgm1MultiGpuTrainer`.

use std::collections::HashMap;
use std::fs;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarMap;
use mini_genome_model::{DnaTokenizer, MiniGenomeConfig, MiniGenomeModel};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::rpc::messages::{GenomeTrainingBatchMsg, GradientUpdate, TrainingBatch};
use crate::trainer::gradient::{build_gradient_update, gradient_commitment};
use crate::trainer::{DeviceInfo, DeviceType, ManualAdamW, Trainer, TrainingResult};

const MASK_TOKEN_ID: usize = 4;
const MASK_RATIO: f64 = 0.15;
const MAX_LEARNING_RATE: f32 = 1e-4;

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

    /// Run a forward/backward pass and return the unscaled loss plus per-variable
    /// gradients moved to the CPU (as F32).
    pub(crate) fn compute_gradients(
        &self,
        input_ids: &Tensor,
        labels: &Tensor,
        loss_scale: f32,
    ) -> Result<(f64, HashMap<String, Tensor>)> {
        let (loss, _) = self.model.compute_mlm_loss(input_ids, labels)?;
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
        Ok((loss_scalar, named_grads))
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

    /// Compute scalar loss for a batch without taking gradients.
    pub(crate) fn compute_loss_scalar(&self, input_ids: &Tensor, labels: &Tensor) -> Result<f64> {
        let (loss, _) = self.model.compute_mlm_loss(input_ids, labels)?;
        Ok(loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64)
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

    /// Run one training step on an (input_ids, labels) pair and return the result.
    fn train_step(&self, input_ids: &Tensor, labels: &Tensor, batch_indices: Vec<u64>, learning_rate: f32) -> Result<TrainingResult> {
        let start = Instant::now();
        let (loss_before, grads) = self.compute_gradients(input_ids, labels, 1.0)?;
        self.apply_gradients(&grads, learning_rate)?;
        let loss_after = self.compute_loss_scalar(input_ids, labels)?;
        let gradients_commitment = gradient_commitment(&grads)?;
        Ok(TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices,
            base_checkpoint: *self.base_checkpoint.lock().unwrap(),
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
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
        self.train_step(&input_ids, &labels, batch.data_indices.clone(), batch.learning_rate)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.prepare_random(n, seed)?;
        let participant_weight = (input_ids.dim(0)? * input_ids.dim(1)?) as f32;

        let start = Instant::now();
        let (loss_before, grads) = self.compute_gradients(&input_ids, &labels, 1.0)?;
        let update = build_gradient_update(
            &self.model_id,
            batch.base_checkpoint,
            grads.clone(),
            participant_weight,
            self.gradient_top_k_ratio,
        )?;
        self.apply_gradients(&grads, batch.learning_rate)?;
        let loss_after = self.compute_loss_scalar(&input_ids, &labels)?;
        let gradients_commitment = gradient_commitment(&grads)?;

        let result = TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices: batch.data_indices.clone(),
            base_checkpoint: batch.base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        };
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
        self.train_step(&input_ids, &labels, batch_indices, self.learning_rate as f32)
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

        let start = Instant::now();
        let (loss_before, grads) = self.compute_gradients(&input_ids, &labels, 1.0)?;
        let update =
            build_gradient_update(&self.model_id, msg.base_checkpoint, grads.clone(), participant_weight, self.gradient_top_k_ratio)?;
        self.apply_gradients(&grads, self.learning_rate as f32)?;
        let loss_after = self.compute_loss_scalar(&input_ids, &labels)?;
        let gradients_commitment = gradient_commitment(&grads)?;

        let result = TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices,
            base_checkpoint: msg.base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        };
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
        assert!(update.is_some());
    }
}
