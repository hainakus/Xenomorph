use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::loss;

use crate::data::{MlmBatch, MlmBatchGenerator};
use crate::dnabert2::DnaBert2ForMaskedLM;
use crate::lora::LoraConfig;
use crate::model::DnaBert2Config;
use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

/// AdamW learning rate cap for DNABERT-2 sized models.
/// The seed-node sends `0.01` for all trainers, but that is far too aggressive for
/// fine-tuning a pre-trained transformer on a single batch and can make the loss
/// increase after one step.
const MAX_LEARNING_RATE: f32 = 1e-5;

/// DNABERT-2 trainer that runs one SGD/AdamW step on a masked language modelling batch.
pub struct DnaBert2Trainer {
    pub(crate) model: DnaBert2ForMaskedLM,
    pub(crate) varmap: candle_nn::VarMap,
    /// Frozen base weights used when merging LoRA adapters for serialization.
    pub(crate) base_weights: Arc<HashMap<String, Tensor>>,
    pub(crate) generator: MlmBatchGenerator,
    pub(crate) device: Device,
    pub(crate) threads: usize,
    pub(crate) optimizer: Mutex<ManualAdamW>,
}

impl DnaBert2Trainer {
    /// Load a trainable DNABERT-2 model and build the MLM batch generator.
    pub fn new(
        config: DnaBert2Config,
        weights: Vec<u8>,
        tokenizer: DnaTokenizer,
        device: Device,
        threads: usize,
        dtype: DType,
        lora_config: Option<LoraConfig>,
    ) -> Result<Self> {
        let (model, varmap, base_weights) =
            DnaBert2ForMaskedLM::load_for_training(config.clone(), weights, dtype, &device, lora_config.as_ref())
                .context("Failed to load DNABERT-2 model for training")?;
        // Only LoRA training needs the frozen base weights for merging the adapter
        // back into a full checkpoint.
        let base_weights = if lora_config.is_some() { base_weights } else { Arc::new(HashMap::new()) };
        let seq_len = config.max_position_embeddings.min(512);
        let generator = MlmBatchGenerator::new(tokenizer, seq_len);
        let optimizer = ManualAdamW::new(0.0);
        let trainer = Self { model, varmap, base_weights, generator, device, threads, optimizer: Mutex::new(optimizer) };
        trainer.sanity_check().context("Loaded checkpoint failed sanity check; weights may contain NaN/Inf")?;
        Ok(trainer)
    }

    /// Run a tiny forward pass on a synthetic batch to verify the loaded weights do
    /// not immediately produce NaN/Inf losses. This catches corrupted checkpoints
    /// before the mining loop starts.
    fn sanity_check(&self) -> Result<()> {
        let batch = self
            .generator
            .generate(&TrainingBatch {
                batch_id: 0,
                model_id: "sanity".to_string(),
                base_checkpoint: [0u8; 32],
                data_indices: vec![0],
                target_improvement: 0.01,
                learning_rate: 0.01,
            })
            .context("Failed to generate sanity batch")?;
        let loss = self.compute_loss_scalar(&batch).context("Sanity-check forward failed")?;
        if !loss.is_finite() {
            bail!("Sanity-check loss is not finite ({}); checkpoint weights may be corrupted", loss);
        }
        Ok(())
    }

    fn build_tensors(&self, batch: &MlmBatch) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let input_ids = Tensor::from_vec(batch.input_ids.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create input_ids tensor")?;

        let attention_mask = Tensor::from_vec(batch.attention_mask.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create attention_mask tensor")?;

        let labels = Tensor::from_vec(batch.labels.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create labels tensor")?;

        let mask = Tensor::from_vec(batch.mask.clone(), (batch.batch_size, batch.seq_len), &self.device)
            .context("Failed to create mask tensor")?;

        Ok((input_ids, attention_mask, labels, mask))
    }

    fn compute_loss(&self, logits: &Tensor, labels: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let dims = logits.dims();
        let (batch, seq, vocab) = (dims[0], dims[1], dims[2]);

        let logits_flat = logits.reshape((batch * seq, vocab))?;
        let labels_flat = labels.reshape((batch * seq,))?;
        let mask_flat = mask.flatten_all()?;

        let mask_vec = mask_flat.to_vec1::<u8>()?;
        let mut positions = Vec::new();
        for (i, &m) in mask_vec.iter().enumerate() {
            if m != 0 {
                positions.push(i as u32);
            }
        }

        // If nothing is masked, the batch cannot produce useful gradients. Bail so the
        // multi-GPU loop can skip this micro-batch instead of returning a constant zero
        // tensor that yields an empty GradStore and breaks gradient averaging.
        if positions.is_empty() {
            bail!("No masked positions in micro-batch; cannot compute MLM loss");
        }

        let positions_t = Tensor::new(positions.as_slice(), &self.device)?;
        let masked_logits = logits_flat.index_select(&positions_t, 0)?;

        // Metal does not implement index_select on U32 source tensors, so gather the
        // masked labels on the CPU and copy them to the device.
        let labels_vec = labels_flat.to_vec1::<u32>()?;
        let masked_labels: Vec<u32> = positions.iter().map(|&i| labels_vec[i as usize]).collect();
        let masked_labels = Tensor::new(masked_labels.as_slice(), &self.device)?;

        // Run cross-entropy / log-softmax in F32 to avoid FP16 overflow/NaN in the loss.
        let masked_logits_f32 = masked_logits.to_dtype(DType::F32)?;
        loss::cross_entropy(&masked_logits_f32, &masked_labels).context("Failed to compute cross-entropy loss")
    }

    /// Compute a deterministic gradient commitment hash from a name -> tensor map.
    pub(crate) fn gradient_commitment_from_named_tensors(&self, grads: &HashMap<String, Tensor>) -> Result<[u8; 32]> {
        let mut hasher = blake3::Hasher::new();
        let mut names: Vec<_> = grads.keys().cloned().collect();
        names.sort();
        for name in names {
            let grad = &grads[&name];
            // Normalize gradients to F32 for a deterministic, device-agnostic commitment.
            let grad_f32 = grad.to_dtype(DType::F32)?;
            let values = grad_f32.flatten_all()?.to_vec1::<f32>()?;
            hasher.update(name.as_bytes());
            for value in values {
                hasher.update(&value.to_le_bytes());
            }
        }
        Ok(*hasher.finalize().as_bytes())
    }

    /// Run a forward/backward pass and return the unscaled loss plus per-variable
    /// gradients moved to the CPU. `loss_scale` can be used for FP16 mixed precision.
    pub(crate) fn compute_gradients(&self, mlm_batch: &MlmBatch, loss_scale: f32) -> Result<(f64, HashMap<String, Tensor>)> {
        let (input_ids, attention_mask, labels, mask) = self.build_tensors(mlm_batch)?;

        let logits_before =
            self.model.forward(&input_ids, None, Some(&attention_mask)).map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;
        let loss_before = self.compute_loss(&logits_before, &labels, &mask)?;
        let loss_before_scalar = loss_before.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        if !loss_before_scalar.is_finite() {
            bail!("Loss is not finite ({}) before backward", loss_before_scalar);
        }

        let scaled_loss = if (loss_scale - 1.0).abs() > f32::EPSILON { (&loss_before * (loss_scale as f64))? } else { loss_before };

        let scaled_loss_scalar = scaled_loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        if !scaled_loss_scalar.is_finite() {
            bail!("Scaled loss is not finite ({}); loss scale {} is too large", scaled_loss_scalar, loss_scale);
        }

        let grads = scaled_loss.backward().context("Backward pass failed")?;
        let named_grads = Self::grad_store_to_map(&grads, &self.varmap)?;
        if named_grads.is_empty() {
            bail!("Backward produced no named gradients; likely no trainable variables in the graph");
        }
        Ok((loss_before_scalar, named_grads))
    }

    /// Apply named gradients to this trainer using its AdamW optimizer.
    pub fn apply_gradients(&self, named_grads: &HashMap<String, Tensor>, learning_rate: f32) -> Result<()> {
        let mut optimizer = self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {}", e))?;
        let effective_lr = learning_rate.min(MAX_LEARNING_RATE);
        optimizer.set_learning_rate(effective_lr as f64);
        optimizer.step(&self.varmap, named_grads).context("Optimizer step failed")?;
        Ok(())
    }

    /// Apply a simple SGD update from a map of named gradients. This is used by
    /// the seed-node's FedAvg aggregator where no per-miner optimizer state is
    /// available; the gradient is already the averaged update across participants.
    pub fn apply_sgd_gradients(&self, named_grads: &HashMap<String, Tensor>, learning_rate: f32) -> Result<()> {
        let effective_lr = learning_rate.min(MAX_LEARNING_RATE) as f64;
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        for (name, var) in data.iter() {
            if let Some(grad) = named_grads.get(name) {
                let theta = var.as_tensor();
                let updated = (theta - &(grad * effective_lr)?)?;
                var.set(&updated)?;
            }
        }
        Ok(())
    }

    /// Persist the current trainable weights to `path` as a SafeTensors file.
    pub fn save_weights_to_path<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        let tensors: HashMap<String, Tensor> = data.iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect();
        candle_core::safetensors::save(&tensors, path).context("Failed to save model weights")
    }

    /// Serialize the current trainable weights into an in-memory SafeTensors buffer.
    ///
    /// For LoRA, this merges the frozen base weights with the trainable adapter so the
    /// result is a full checkpoint compatible with existing storage and inference paths.
    pub fn save_weights_to_bytes(&self) -> Result<Vec<u8>> {
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (k, v) in self.base_weights.iter() {
            tensors.insert(k.clone(), v.clone());
        }
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        for (k, v) in data.iter() {
            tensors.insert(k.clone(), v.as_tensor().clone());
        }
        let tensors: Vec<(String, &Tensor)> = tensors.iter().map(|(k, v)| (k.clone(), v)).collect();
        safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow::anyhow!("Failed to serialize model weights: {}", e))
    }

    /// Serialize only the trainable adapter weights (LoRA A/B) into a SafeTensors buffer.
    pub fn save_adapter_to_bytes(&self) -> Result<Vec<u8>> {
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        let tensors: Vec<(String, &Tensor)> = data.iter().map(|(k, v)| (k.clone(), v.as_tensor())).collect();
        safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow::anyhow!("Failed to serialize adapter weights: {}", e))
    }

    /// Serialize the frozen base weights into a SafeTensors buffer.
    ///
    /// For non-LoRA models this is the full checkpoint; for LoRA it is the base
    /// model without the adapter tensors.
    pub fn save_base_weights_to_bytes(&self) -> Result<Vec<u8>> {
        if self.base_weights.is_empty() {
            return self.save_weights_to_bytes();
        }
        let tensors: Vec<(String, &Tensor)> = self.base_weights.iter().map(|(k, v)| (k.clone(), v)).collect();
        safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow::anyhow!("Failed to serialize base weights: {}", e))
    }

    /// Load trainable weights from an in-memory SafeTensors buffer into the live VarMap.
    ///
    /// For LoRA training, missing `*.lora_a` / `*.lora_b` keys are ignored so that a
    /// base (non-LoRA) checkpoint can be hot-reloaded; the LoRA matrices were already
    /// initialized to zero when the model was built.
    pub fn load_weights_from_bytes(&self, weights: &[u8]) -> Result<()> {
        let loaded = candle_core::safetensors::load_buffer(weights, &self.device).context("Failed to load safetensors weights")?;
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        let is_lora = !self.base_weights.is_empty();
        for (name, var) in data.iter() {
            let loaded_var = match loaded.get(name) {
                Some(v) => v,
                None => {
                    if is_lora && (name.ends_with(".lora_a") || name.ends_with(".lora_b")) {
                        continue;
                    }
                    anyhow::bail!("Missing weight {} in checkpoint buffer", name);
                }
            };
            let loaded_var = loaded_var.to_device(&self.device)?.to_dtype(var.as_tensor().dtype())?;
            var.set(&loaded_var)?;
        }
        Ok(())
    }

    /// Reset the AdamW optimizer state (step counter and moments).
    pub fn reset_optimizer(&self) -> Result<()> {
        self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {}", e))?.reset()
    }

    /// Access the underlying trainable variables. Used by the seed-node FedAvg
    /// aggregator to reconstruct gradient tensors with the correct shapes.
    pub fn varmap(&self) -> &candle_nn::VarMap {
        &self.varmap
    }

    /// Return a copy of the current trainable weights keyed by variable name.
    pub fn trainable_weights(&self) -> Result<HashMap<String, Tensor>> {
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        Ok(data.iter().map(|(k, v)| (k.clone(), v.as_tensor().clone())).collect())
    }

    /// Return a clone of the AdamW optimizer state.
    pub fn clone_optimizer(&self) -> Result<ManualAdamW> {
        let optimizer = self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {}", e))?;
        Ok(optimizer.clone())
    }

    /// Apply a raw weight-space delta to the current trainable variables.
    /// This is used for delta rebase: B' = B + delta.
    pub fn apply_weight_delta(&self, delta: &HashMap<String, Tensor>) -> Result<()> {
        let data = self.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        for (name, var) in data.iter() {
            if let Some(d) = delta.get(name) {
                let current = var.as_tensor();
                let updated = current.broadcast_add(d).with_context(|| format!("Failed to apply delta to {}", name))?;
                var.set(&updated).with_context(|| format!("Failed to set updated tensor for {}", name))?;
            }
        }
        Ok(())
    }

    /// Compute the weight-space delta `A' - A` that results from applying the
    /// averaged gradients to a snapshot of base `A` using `A`'s optimizer state.
    /// The returned delta can be added to the active checkpoint `B` to produce `B'`.
    pub fn compute_delta_from_snapshot(
        &self,
        snapshot_weights: &HashMap<String, Tensor>,
        snapshot_optimizer: &ManualAdamW,
        named_grads: &HashMap<String, Tensor>,
        learning_rate: f32,
    ) -> Result<HashMap<String, Tensor>> {
        let temp_varmap = candle_nn::VarMap::new();
        {
            let mut data = temp_varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
            for (name, tensor) in snapshot_weights {
                let var = Var::from_tensor(tensor).with_context(|| format!("Failed to create Var for {}", name))?;
                data.insert(name.clone(), var);
            }
        }

        let mut optimizer = snapshot_optimizer.clone();
        optimizer.set_learning_rate(learning_rate.min(MAX_LEARNING_RATE) as f64);
        optimizer.step(&temp_varmap, named_grads).context("Failed to apply gradients to snapshot")?;

        let updated = temp_varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        let mut delta = HashMap::with_capacity(snapshot_weights.len());
        for (name, var) in updated.iter() {
            let snapshot_tensor = snapshot_weights.get(name).ok_or_else(|| anyhow::anyhow!("Missing snapshot weight for {}", name))?;
            let d = var.as_tensor().sub(snapshot_tensor).with_context(|| format!("Failed to compute delta for {}", name))?;
            delta.insert(name.clone(), d);
        }
        Ok(delta)
    }

    /// Compute the scalar loss for a batch without taking gradients.
    pub(crate) fn compute_loss_scalar(&self, mlm_batch: &MlmBatch) -> Result<f64> {
        let (input_ids, attention_mask, labels, mask) = self.build_tensors(mlm_batch)?;
        let logits =
            self.model.forward(&input_ids, None, Some(&attention_mask)).map_err(|e| anyhow::anyhow!("Forward pass failed: {}", e))?;
        let loss = self.compute_loss(&logits, &labels, &mask)?;
        Ok(loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64)
    }

    /// Convert a `GradStore` into a `HashMap` keyed by variable name, with gradients
    /// cast to F32 on their original device for stable averaging. The caller is
    /// responsible for moving gradients to a common device before averaging.
    fn grad_store_to_map(grads: &candle_core::backprop::GradStore, varmap: &candle_nn::VarMap) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        let data = varmap.data().lock().map_err(|e: PoisonError<_>| anyhow::anyhow!("VarMap poisoned: {}", e))?;
        for (name, var) in data.iter() {
            if let Some(grad) = grads.get(var.as_tensor()) {
                let grad = grad.to_dtype(DType::F32)?;
                out.insert(name.clone(), grad);
            }
        }
        Ok(out)
    }
}

impl DnaBert2Trainer {
    /// Shared training step: forward, compute loss, backward, optimize, forward again.
    fn train_mlm_batch(
        &self,
        mlm_batch: &MlmBatch,
        model_id: &str,
        base_checkpoint: [u8; 32],
        batch_indices: Vec<u64>,
        learning_rate: f32,
    ) -> Result<TrainingResult> {
        let start = Instant::now();

        let (loss_before_scalar, grads) = self.compute_gradients(mlm_batch, 1.0)?;
        self.apply_gradients(&grads, learning_rate)?;
        let loss_after_scalar = self.compute_loss_scalar(mlm_batch)?;
        let gradients_commitment = self.gradient_commitment_from_named_tensors(&grads)?;

        Ok(TrainingResult {
            model_id: model_id.to_string(),
            batch_indices,
            base_checkpoint,
            loss_before: loss_before_scalar,
            loss_after: loss_after_scalar,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
    }
}

impl Trainer for DnaBert2Trainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mlm_batch = self.generator.generate(batch).context("Failed to generate MLM batch")?;
        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        let batch = &msg.batch;
        let mlm_batch = self
            .generator
            .generate_from_sequences(&msg.sequences, &batch.genome_merkle_root, batch.batch_id)
            .context("Failed to generate MLM batch from genome sequences")?;

        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();

        self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, batch_indices, 0.01)
    }

    fn device_info(&self) -> DeviceInfo {
        let device_type = if self.device.is_cuda() {
            DeviceType::Cuda
        } else if self.device.is_metal() {
            DeviceType::Metal
        } else {
            DeviceType::Cpu
        };

        let name = match device_type {
            DeviceType::Cuda => format!("DNABERT-2 CUDA trainer ({} threads)", self.threads),
            DeviceType::Metal => format!("DNABERT-2 Metal trainer ({} threads)", self.threads),
            _ => format!("DNABERT-2 CPU trainer ({} threads)", self.threads),
        };

        DeviceInfo { device_type, name, threads: self.threads, ..Default::default() }
    }
}

/// A minimal AdamW optimizer that operates on a `VarMap` using a name-indexed
/// gradient map. This exists because `candle_core::backprop::GradStore` cannot
/// be constructed from outside the crate, which prevents feeding averaged
/// multi-GPU gradients into `candle_nn::AdamW`.
pub struct ManualAdamW {
    step_t: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    /// First/second moment estimates keyed by variable name.
    moments: Mutex<HashMap<String, (Tensor, Tensor)>>,
}

impl Clone for ManualAdamW {
    fn clone(&self) -> Self {
        let moments = self
            .moments
            .lock()
            .expect("ManualAdamW moments mutex poisoned during clone")
            .iter()
            .map(|(k, (m, v))| (k.clone(), (m.clone(), v.clone())))
            .collect();
        Self {
            step_t: self.step_t,
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
            moments: Mutex::new(moments),
        }
    }
}

impl ManualAdamW {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            step_t: 0,
            lr: learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            moments: Mutex::new(HashMap::new()),
        }
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Reset the AdamW step counter and first/second moment buffers.
    pub fn reset(&mut self) -> Result<()> {
        self.step_t = 0;
        let mut moments = self.moments.lock().map_err(|e| anyhow::anyhow!("Moments mutex poisoned: {}", e))?;
        moments.clear();
        Ok(())
    }

    /// Apply named gradients (multi-GPU path where gradients are already on CPU and in F32).
    pub fn step(&mut self, varmap: &candle_nn::VarMap, named_grads: &HashMap<String, Tensor>) -> Result<()> {
        self.step_t += 1;
        let lr = self.lr;
        let lambda = self.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let scale_m = 1.0 / (1.0 - beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - beta2.powi(self.step_t as i32));

        let mut moments = self.moments.lock().map_err(|e| anyhow::anyhow!("Moments mutex poisoned: {}", e))?;
        let data = varmap.data().lock().map_err(|e: PoisonError<_>| anyhow::anyhow!("VarMap poisoned: {}", e))?;

        for (name, var) in data.iter() {
            let grad = match named_grads.get(name) {
                Some(g) => g,
                None => continue,
            };

            let theta = var.as_tensor();
            let device = theta.device();
            let grad = grad.to_device(device)?;

            let (m, v) = moments.entry(name.clone()).or_insert_with(|| {
                // Keep moment estimates in F32 for numerical stability, even when the model is F16.
                let m = Tensor::zeros(theta.shape().clone(), DType::F32, device).unwrap();
                let v = Tensor::zeros(theta.shape().clone(), DType::F32, device).unwrap();
                (m, v)
            });

            let next_m = ((&*m * beta1)? + (&grad * (1.0 - beta1))?)?;
            let next_v = ((&*v * beta2)? + (&grad.sqr()? * (1.0 - beta2))?)?;
            let m_hat = (&next_m * scale_m)?;
            let v_hat = (&next_v * scale_v)?;
            let next_theta = (theta * (1.0 - lr_lambda))?;
            let adjusted_grad = (&m_hat / (&v_hat.sqrt()? + self.eps)?)?;
            // Cast the update back to the parameter's dtype (F16 or F32) before applying it.
            let adjusted_grad = adjusted_grad.to_dtype(theta.dtype())?;
            let next_theta = (&next_theta - (&adjusted_grad * lr)?)?;

            var.set(&next_theta)?;
            *m = next_m;
            *v = next_v;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;
    use std::collections::HashMap;
    use tokenizers::models::bpe::Vocab;

    fn build_tiny_tokenizer() -> DnaTokenizer {
        let mut vocab: Vocab = Vocab::new();
        vocab.insert("<pad>".to_string(), 0);
        vocab.insert("A".to_string(), 1);
        vocab.insert("T".to_string(), 2);
        vocab.insert("C".to_string(), 3);
        vocab.insert("G".to_string(), 4);
        vocab.insert("<mask>".to_string(), 5);

        let bpe = tokenizers::models::bpe::BPE::new(vocab, vec![]);
        let mut tokenizer = tokenizers::Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[
            tokenizers::tokenizer::AddedToken::from("<mask>", true),
            tokenizers::tokenizer::AddedToken::from("<pad>", true),
        ]);

        let bytes = serde_json::to_vec(&tokenizer).unwrap();
        DnaTokenizer::from_bytes(&bytes).unwrap()
    }

    fn insert_weight(map: &mut HashMap<String, Tensor>, name: &str, shape: &[usize], device: &Device) {
        let n = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.01).sin() + 0.001).collect();
        let t = Tensor::from_vec(data, shape, device).unwrap();
        map.insert(name.to_string(), t);
    }

    fn build_tiny_safetensors() -> (DnaBert2Config, Vec<u8>) {
        let device = Device::Cpu;
        let config = DnaBert2Config {
            vocab_size: 8,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            type_vocab_size: 2,
            hidden_dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".to_string(),
            position_embedding_type: "alibi".to_string(),
            alibi_starting_size: Some(16),
            tie_word_embeddings: true,
            pad_token_id: 0,
            mask_token_id: 5,
            bos_token_id: 1,
            eos_token_id: 2,
            num_labels: None,
        };

        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        insert_weight(&mut tensors, "model.embeddings.word_embeddings.weight", &[config.vocab_size, config.hidden_size], &device);
        insert_weight(
            &mut tensors,
            "model.embeddings.token_type_embeddings.weight",
            &[config.type_vocab_size, config.hidden_size],
            &device,
        );
        insert_weight(&mut tensors, "model.embeddings.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "model.embeddings.layer_norm.bias", &[config.hidden_size], &device);

        for i in 0..config.num_hidden_layers {
            let prefix = format!("model.encoder.layer.{}", i);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.query.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.query.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.key.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.key.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.self.value.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.self.value.bias", prefix), &[config.hidden_size], &device);
            insert_weight(
                &mut tensors,
                &format!("{}.attention.output.dense.weight", prefix),
                &[config.hidden_size, config.hidden_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.attention.output.dense.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.attention.output.layer_norm.bias", prefix), &[config.hidden_size], &device);

            insert_weight(
                &mut tensors,
                &format!("{}.mlp.up_proj.weight", prefix),
                &[config.intermediate_size * 2, config.hidden_size],
                &device,
            );
            insert_weight(
                &mut tensors,
                &format!("{}.mlp.down_proj.weight", prefix),
                &[config.hidden_size, config.intermediate_size],
                &device,
            );
            insert_weight(&mut tensors, &format!("{}.mlp.down_proj.bias", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.weight", prefix), &[config.hidden_size], &device);
            insert_weight(&mut tensors, &format!("{}.mlp.layer_norm.bias", prefix), &[config.hidden_size], &device);
        }

        insert_weight(&mut tensors, "lm_head.transform.dense.weight", &[config.hidden_size, config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.dense.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.weight", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.transform.layer_norm.bias", &[config.hidden_size], &device);
        insert_weight(&mut tensors, "lm_head.bias", &[config.vocab_size], &device);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.safetensors");
        candle_core::safetensors::save(&tensors, &path).unwrap();
        let weights = std::fs::read(&path).unwrap();
        (config, weights)
    }

    fn dummy_batch() -> TrainingBatch {
        TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: vec![0, 1, 2, 3],
            target_improvement: 0.01,
            learning_rate: 0.01,
        }
    }

    #[test]
    fn test_dna_bert2_trainer_runs_and_improves() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32, None).unwrap();

        let result = trainer.train(&dummy_batch()).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        // The tiny model should usually reduce the loss after one AdamW step.
        // We allow equality in the very rare case where the random seed gives no improvement.
        assert!(result.loss_after <= result.loss_before);

        // A second step on the same batch should start from a lower loss.
        let result2 = trainer.train(&dummy_batch()).unwrap();
        assert!(
            result2.loss_before <= result.loss_after,
            "model state did not persist: {} > {}",
            result2.loss_before,
            result.loss_after
        );
    }

    #[test]
    fn test_dna_bert2_trainer_lora_runs_and_improves() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let lora_config = crate::lora::LoraConfig {
            rank: 2,
            alpha: 4.0,
            dropout: 0.0,
            target_modules: crate::lora::LoraConfig::default_target_modules(),
        };
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32, Some(lora_config)).unwrap();

        let result = trainer.train(&dummy_batch()).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        assert!(result.loss_after <= result.loss_before);

        // The adapter checkpoint should only contain LoRA A/B tensors.
        let adapter_bytes = trainer.save_adapter_to_bytes().unwrap();
        let adapter = candle_core::safetensors::load_buffer(&adapter_bytes, &Device::Cpu).unwrap();
        assert!(adapter.keys().all(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")));

        // The merged checkpoint must contain both base and LoRA keys.
        let merged_bytes = trainer.save_weights_to_bytes().unwrap();
        let merged = candle_core::safetensors::load_buffer(&merged_bytes, &Device::Cpu).unwrap();
        assert!(merged.keys().any(|k| k.ends_with(".lora_a") || k.ends_with(".lora_b")));
        assert!(merged.keys().any(|k| k.ends_with(".weight") && !k.ends_with(".lora_a") && !k.ends_with(".lora_b")));
    }

    #[test]
    fn test_dna_bert2_trainer_train_genome() {
        use crate::rpc::messages::{GenomeSlice, GenomeTrainingBatch};

        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32, None).unwrap();

        let msg = GenomeTrainingBatchMsg {
            batch: GenomeTrainingBatch {
                batch_id: 1,
                model_id: "dnabert2".to_string(),
                genome_merkle_root: [1u8; 32],
                data_indices: vec![
                    GenomeSlice { chunk_idx: 0, start_base: 0, length: 4 },
                    GenomeSlice { chunk_idx: 1, start_base: 0, length: 4 },
                ],
                mask_ratio: 0.15,
                seq_length: 8,
            },
            sequences: vec!["ATCG".to_string(), "GCTA".to_string()],
            base_checkpoint: [1u8; 32],
        };

        let result = trainer.train_genome(&msg).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert_eq!(result.batch_indices, vec![0, 1]);
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
        assert!(result.loss_after <= result.loss_before);
    }

    #[test]
    fn test_delta_rebase_matches_direct_training() {
        let (config, weights) = build_tiny_safetensors();
        let tokenizer = build_tiny_tokenizer();
        let base_trainer =
            DnaBert2Trainer::new(config.clone(), weights.clone(), tokenizer.clone(), Device::Cpu, 2, DType::F32, None).unwrap();

        let batch = base_trainer.generator.generate(&dummy_batch()).unwrap();
        let (_, named_grads) = base_trainer.compute_gradients(&batch, 1.0).unwrap();

        // Snapshot the base before any update.
        let snapshot_weights = base_trainer.trainable_weights().unwrap();
        let snapshot_optimizer = base_trainer.clone_optimizer().unwrap();

        // Directly train a copy of the base and capture its new weights.
        let direct_trainer =
            DnaBert2Trainer::new(config.clone(), weights.clone(), tokenizer.clone(), Device::Cpu, 2, DType::F32, None).unwrap();
        direct_trainer.apply_gradients(&named_grads, 1e-5f32).unwrap();
        let direct_weights = direct_trainer.trainable_weights().unwrap();

        // Rebase: compute delta from the base snapshot and apply it to another copy.
        let rebased_trainer = DnaBert2Trainer::new(config, weights, tokenizer, Device::Cpu, 2, DType::F32, None).unwrap();
        let delta =
            rebased_trainer.compute_delta_from_snapshot(&snapshot_weights, &snapshot_optimizer, &named_grads, 1e-5f32).unwrap();
        rebased_trainer.apply_weight_delta(&delta).unwrap();
        let rebased_weights = rebased_trainer.trainable_weights().unwrap();

        // The rebased weights should match the directly trained weights.
        for (name, direct) in direct_weights {
            let rebased = rebased_weights.get(&name).expect("rebased weights missing variable");
            let diff = direct.sub(rebased).unwrap().abs().unwrap().mean_all().unwrap().to_vec0::<f32>().unwrap();
            assert!(diff < 1e-6, "delta rebase mismatch for {}: {}", name, diff);
        }
    }
}
