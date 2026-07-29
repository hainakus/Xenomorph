//! Single-device trainer for the Mini Genome Model (MGM-1).
//!
//! Supports masked-language-model training, gradient extraction for FedAvg, and
//! multi-GPU data-parallel via `Mgm1MultiGpuTrainer`.

use std::collections::HashMap;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, D as TensorD};
use candle_nn::VarMap;
use mini_genome_model::{DnaTokenizer, MiniGenomeConfig, MiniGenomeModel, MLM_IGNORE_INDEX};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tracing::info;

use crate::rpc::messages::{GenomeTrainingBatchMsg, GradientUpdate, TrainingBatch};
use crate::trainer::gradient::{add_grad_maps, build_gradient_update, clip_grad_norm, gradient_commitment, scale_grad_map};
use crate::trainer::{DeviceInfo, DeviceType, ManualAdamW, MultiGpuConfig, Trainer, TrainingResult};

const MASK_TOKEN_ID: usize = 4;
const MASK_RATIO: f64 = 0.15;

/// Contiguous span of masked bases.  Longer spans make the model learn context
/// beyond single-base prediction and reduce the chance of collapse to a single
/// nucleotide.  Mean length 5 keeps the expected mask ratio close to 15%.
const MIN_SPAN_LEN: usize = 3;
const MAX_SPAN_LEN: usize = 7;
/// Number of local gradient steps taken on each genome batch.  A single AdamW
/// step on a masked language-modeling task only learns a marginal class bias
/// (argmax collapses to the majority base); several steps are needed for the
/// transformer to learn context and predict minority bases correctly.  With
/// gradient accumulation the effective batch per step is much larger, so four
/// local steps are enough to fit local context without overfitting the same
/// batch.
pub(crate) const MGM1_LOCAL_STEPS: usize = 4;
pub(crate) const MAX_LEARNING_RATE: f32 = 1e-3;

/// Trainer for the `xenom/mgm-1` model.
pub struct Mgm1Trainer {
    model: MiniGenomeModel,
    varmap: Mutex<VarMap>,
    optimizer: Mutex<ManualAdamW>,
    config: MiniGenomeConfig,
    class_weights: Option<Tensor>,
    tokenizer: DnaTokenizer,
    device: Device,
    base_checkpoint: Mutex<[u8; 32]>,
    model_id: String,
    gradient_top_k_ratio: f32,
    learning_rate: f64,
    /// Global optimizer step counter used for the warmup/cosine LR schedule.
    step_count: AtomicUsize,
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
        gpu_config: &MultiGpuConfig,
    ) -> Result<Self> {
        let model_id = model_id.into();
        let mut config: MiniGenomeConfig =
            serde_json::from_slice(config_bytes).with_context(|| format!("Failed to parse MGM-1 config for {model_id}"))?;

        // The CLI/JSON config may not reflect the user's selected micro-batch and
        // accumulation settings.  Override the model defaults with the multi-GPU
        // config so single-GPU MGM-1 respects the same --micro-batch-size and
        // --gradient-accumulation flags as DNABERT-2.
        config.micro_batch_size = gpu_config.micro_batch_size.max(1);
        config.gradient_accumulation_steps = gpu_config.gradient_accumulation_steps.max(1);

        let varmap = Mutex::new(VarMap::new());
        let model = {
            let locked = varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
            let vb = candle_nn::VarBuilder::from_varmap(&locked, DType::F32, &device);
            MiniGenomeModel::new(vb, config.clone()).with_context(|| format!("Failed to build MGM-1 model {model_id}"))?
        };

        if !weights.is_empty() {
            load_varmap_weights(&varmap, &weights, &device)?;
        }

        let class_weights = config
            .class_weights
            .as_ref()
            .map(|cw| {
                let cw_4: Vec<f32> = cw.iter().take(4).copied().collect();
                Tensor::new(cw_4.as_slice(), &device).with_context(|| "Failed to build MGM-1 class weights tensor")
            })
            .transpose()?;

        let mut optimizer = ManualAdamW::new(lr);
        optimizer.set_weight_decay(config.weight_decay);

        Ok(Self {
            model,
            varmap,
            optimizer: Mutex::new(optimizer),
            config,
            class_weights,
            tokenizer: DnaTokenizer::new(),
            device,
            base_checkpoint: Mutex::new(base_checkpoint),
            model_id,
            gradient_top_k_ratio: gpu_config.gradient_top_k_ratio,
            learning_rate: lr,
            step_count: AtomicUsize::new(0),
        })
    }

    pub(crate) fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.model.forward(input_ids).map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Tokenize a batch of DNA sequences, padding/truncating to `target_len`.
    fn tokenize_batch(&self, sequences: &[String], target_len: usize) -> Vec<Vec<usize>> {
        sequences
            .iter()
            .map(|seq| {
                let mut tokens = self.tokenizer.encode(seq);
                tokens.truncate(target_len);
                while tokens.len() < target_len {
                    tokens.push(MASK_TOKEN_ID);
                }
                tokens
            })
            .collect()
    }

    /// Build masked-language-modeling input and label tensors from token IDs.
    ///
    /// Uses span-level masking and the BERT-style 80/10/10 split for the selected
    /// MLM positions:
    ///   - 80% replaced with [MASK]
    ///   - 10% replaced with a random DNA base
    ///   - 10% left unchanged
    ///
    /// Unselected positions (including padding [MASK] tokens) get the ignore
    /// label so `compute_mlm_loss` does not train on them.
    fn build_mlm_tensors(&self, token_ids: &[Vec<usize>], rng: &mut ChaCha8Rng, mask_ratio: f64) -> Result<(Tensor, Tensor)> {
        let batch = token_ids.len();
        let seq_len = token_ids[0].len();
        let mean_span_len = (MIN_SPAN_LEN + MAX_SPAN_LEN) as f64 / 2.0;
        let start_prob = mask_ratio / mean_span_len;

        let mut input_data = Vec::with_capacity(batch * seq_len);
        let mut label_data = Vec::with_capacity(batch * seq_len);

        for row in token_ids {
            // Build a boolean mask indicating which positions are selected.  Spans
            // never start on [MASK] padding and never overflow into padding.
            let mut selected = vec![false; seq_len];
            let mut i = 0;
            while i < seq_len {
                if row[i] == MASK_TOKEN_ID || i + MIN_SPAN_LEN > seq_len {
                    i += 1;
                    continue;
                }
                if !rng.gen_bool(start_prob) {
                    i += 1;
                    continue;
                }
                let max_len = (MAX_SPAN_LEN).min(seq_len - i);
                // Pick a span length and shrink it if it would hit padding.
                let mut span_len = if max_len >= MIN_SPAN_LEN {
                    rng.gen_range(MIN_SPAN_LEN..=max_len)
                } else {
                    max_len
                };
                while span_len > 0 && i + span_len <= seq_len && row[i + span_len - 1] == MASK_TOKEN_ID {
                    span_len -= 1;
                }
                if span_len >= MIN_SPAN_LEN {
                    for j in i..i + span_len {
                        selected[j] = true;
                    }
                    i += span_len;
                } else {
                    i += 1;
                }
            }

            for (t, &tok) in row.iter().enumerate() {
                if selected[t] {
                    let roll: f64 = rng.gen();
                    let input_tok = if roll < 0.8 {
                        MASK_TOKEN_ID
                    } else if roll < 0.9 {
                        rng.gen_range(0..4)
                    } else {
                        tok
                    };
                    input_data.push(input_tok as i64);
                    label_data.push(tok as i64);
                } else if tok == MASK_TOKEN_ID {
                    input_data.push(tok as i64);
                    label_data.push(MLM_IGNORE_INDEX);
                } else {
                    input_data.push(tok as i64);
                    label_data.push(MLM_IGNORE_INDEX);
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
                // Synthetic MLM batches should only contain the four DNA bases;
                // special tokens are learned by the input/padding paths, not as MLM
                // targets, so training on them would corrupt the genomic objective.
                row.push(rng.gen_range(0..4));
            }
            token_ids.push(row);
        }
        self.build_mlm_tensors(&token_ids, &mut rng, MASK_RATIO)
    }

    /// Prepare a genome-backed batch on the trainer's device.
    ///
    /// `mask_ratio` and `target_len` are taken from the seed-node's batch message;
    /// they are clamped to safe ranges so the node can control masking/length without
    /// breaking the model's tensor assumptions.
    pub fn prepare_sequences(
        &self,
        sequences: &[String],
        seed: [u8; 32],
        mask_ratio: f64,
        target_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        let target_len = target_len.clamp(1, self.config.max_seq_len);
        let mask_ratio = mask_ratio.clamp(0.0, 0.5);
        let token_ids = self.tokenize_batch(sequences, target_len);
        let mut rng = ChaCha8Rng::from_seed(seed);
        self.build_mlm_tensors(&token_ids, &mut rng, mask_ratio)
    }

    /// Compute loss and accuracy for a batch without taking gradients.
    pub fn compute_loss_and_accuracy(&self, input_ids: &Tensor, labels: &Tensor) -> Result<(f64, f32)> {
        // Use the model config's stable global class weights.  Per-batch
        // reweighting is disabled because it amplifies sampling noise.
        let (loss, accuracy) = self.model.compute_mlm_loss(input_ids, labels, self.class_weights.as_ref())?;
        let loss_scalar = loss.to_dtype(DType::F32)?.to_vec0::<f32>()? as f64;
        Ok((loss_scalar, accuracy))
    }

    /// Run a forward/backward pass and return the unscaled loss, accuracy, and
    /// per-variable gradients moved to the CPU (as F32).
    ///
    /// `class_weights` is intentionally not auto-computed from the batch;
    /// per-batch reweighting amplifies sampling noise and causes single-base
    /// collapse. Callers may pass a stable global weight tensor, or `None`.
    pub(crate) fn compute_gradients(
        &self,
        input_ids: &Tensor,
        labels: &Tensor,
        loss_scale: f32,
        class_weights: Option<&Tensor>,
    ) -> Result<(f64, f32, HashMap<String, Tensor>)> {
        let (loss, accuracy) = self.model.compute_mlm_loss(input_ids, labels, class_weights)?;
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

    /// Compute the scheduled learning rate for the current optimizer step.
    ///
    /// Linear warmup from 0 to `peak_lr` over `warmup_steps`, then cosine decay
    /// from `peak_lr` down to `min_learning_rate` over the remaining steps until
    /// `training_steps`, after which it stays at the floor.
    fn scheduled_lr(&self, peak_lr: f64) -> f64 {
        let step = self.step_count.fetch_add(1, Ordering::Relaxed) + 1;
        let warmup = self.config.warmup_steps;
        let total = self.config.training_steps.max(warmup + 1);
        let floor = self.config.min_learning_rate;

        if step >= total {
            return floor;
        }
        if warmup > 0 && step <= warmup {
            // Linear warmup starting from 0 (not the floor) so very early steps
            // are conservative and do not shock a fresh random model.
            return peak_lr * (step as f64 / warmup as f64);
        }

        // Cosine annealing from peak to floor.
        let progress = (step - warmup) as f64 / (total - warmup) as f64;
        let cosine = 0.5 * (1.0 + (progress * std::f64::consts::PI).cos());
        floor + (peak_lr - floor) * cosine
    }

    /// Apply named gradients to this trainer using its AdamW optimizer.
    pub(crate) fn apply_gradients(&self, named_grads: &HashMap<String, Tensor>, learning_rate: f32) -> Result<()> {
        // Clip global gradient norm to prevent a single noisy batch from pushing
        // logits into a saturated softmax and collapsing predictions to one base.
        let clipped = clip_grad_norm(named_grads, self.config.grad_clip_norm as f64).context("Gradient clipping failed")?;
        let mut optimizer = self.optimizer.lock().map_err(|e| anyhow::anyhow!("Optimizer mutex poisoned: {e}"))?;
        let peak_lr = (learning_rate as f64).min(MAX_LEARNING_RATE as f64);
        let effective_lr = self.scheduled_lr(peak_lr);
        optimizer.set_learning_rate(effective_lr);
        let varmap = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        optimizer.step(&varmap, &clipped).context("Optimizer step failed")?;
        Ok(())
    }

    /// Add a raw weight-space delta to the current weights.  This is used by the
    /// validator to apply the decrypted update directly instead of re-running the
    /// full local training loop, which avoids cross-device numerical drift.
    pub fn apply_weight_delta(&self, named_deltas: &HashMap<String, Tensor>) -> Result<()> {
        let varmap = self.varmap.lock().map_err(|e| anyhow::anyhow!("VarMap mutex poisoned: {e}"))?;
        let data = varmap.data().lock().map_err(|e: std::sync::PoisonError<_>| anyhow::anyhow!("VarMap data poisoned: {e}"))?;
        for (name, var) in data.iter() {
            let delta = named_deltas.get(name).with_context(|| format!("Missing weight delta for {name}"))?;
            let updated = (var.as_tensor() + delta)?;
            var.set(&updated).with_context(|| format!("Failed to apply weight delta to {name}"))?;
        }
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

    /// Run one or more local training steps on an (input_ids, labels) pair.
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
        local_steps: usize,
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

        let (loss_before, accuracy_before) = self.compute_loss_and_accuracy(input_ids, labels)?;
        let label_dist = masked_label_distribution(input_ids, labels)?;

        let mut grads_for_commitment = HashMap::new();
        let mut grad_norm = 0.0;
        let micro_batch_size = self.config.micro_batch_size.max(1);
        let accumulation_steps = self.config.gradient_accumulation_steps.max(1);
        let batch_size = input_ids.dim(0)?;
        let effective_batch = micro_batch_size.saturating_mul(accumulation_steps);

        for _ in 0..local_steps.max(1) {
            let mut offset = 0usize;
            while offset < batch_size {
                let mut accumulated_grads: Option<HashMap<String, Tensor>> = None;
                let mut total_masked: f64 = 0.0;
                let mut micros = 0usize;

                while micros < accumulation_steps && offset + micros * micro_batch_size < batch_size {
                    let start = offset + micros * micro_batch_size;
                    let end = (start + micro_batch_size).min(batch_size);
                    let ids = input_ids.narrow(0, start, end - start)?;
                    let lbls = labels.narrow(0, start, end - start)?;

                    let (_, _, grads) = self.compute_gradients(&ids, &lbls, 1.0, self.class_weights.as_ref())?;

                    // Weight gradients by the number of masked positions in this micro-batch
                    // so that accumulating and then dividing by total_masked gives the
                    // average gradient over the accumulated chunk.
                    let mask = ids.ne(&lbls)?.to_dtype(DType::F32)?;
                    let masked_count = mask.sum_all()?.to_vec0::<f32>()? as f64;
                    let scaled = scale_grad_map(grads, masked_count).context("Failed to scale micro-batch gradients")?;

                    accumulated_grads = Some(match accumulated_grads {
                        None => scaled,
                        Some(acc) => add_grad_maps(acc, scaled)?,
                    });
                    total_masked += masked_count;
                    micros += 1;
                }

                let final_grads = accumulated_grads
                    .ok_or_else(|| anyhow::anyhow!("No gradients were produced by any micro-batch"))?;
                let final_grads = if total_masked > 0.0 {
                    scale_grad_map(final_grads, 1.0 / total_masked).context("Failed to scale accumulated gradient")?
                } else {
                    final_grads
                };

                grad_norm = gradient_norm(&final_grads)?;
                grads_for_commitment = final_grads.clone();
                self.apply_gradients(&final_grads, learning_rate)?;

                offset += effective_batch;
            }
        }

        let (loss_after, accuracy_after) = self.compute_loss_and_accuracy(input_ids, labels)?;
        let logits = self.model.forward(input_ids)?;
        let pred_dist = masked_prediction_distribution(input_ids, labels, &logits)?;

        let updated_weights = self.varmap_snapshot()?;
        let weight_delta = if return_update { Some(Self::compute_weight_delta(&base_weights, &updated_weights)?) } else { None };

        // Commitment is over the payload that will be sent to the seed-node.
        let gradients_commitment = gradient_commitment(weight_delta.as_ref().unwrap_or(&grads_for_commitment))?;
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
        if weights.is_empty() {
            return Ok(());
        }
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
        self.train_step(&input_ids, &labels, batch.data_indices.clone(), batch.learning_rate, false, 1).map(|(r, _)| r)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.prepare_random(n, seed)?;
        let participant_weight = (input_ids.dim(0)? * input_ids.dim(1)?) as f32;

        let (mut result, weight_delta) =
            self.train_step(&input_ids, &labels, batch.data_indices.clone(), batch.learning_rate, true, 1)?;
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

        let mask_ratio = msg.batch.mask_ratio as f64;
        let target_len = msg.batch.seq_length;
        let (input_ids, labels) = self.prepare_sequences(&msg.sequences, seed, mask_ratio, target_len)?;
        self.train_step(&input_ids, &labels, batch_indices, self.learning_rate as f32, false, MGM1_LOCAL_STEPS).map(|(r, _)| r)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        if msg.sequences.is_empty() {
            anyhow::bail!("Genome batch contains no sequences");
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&msg.base_checkpoint);
        seed[..8].copy_from_slice(&msg.batch.batch_id.to_le_bytes());
        let batch_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();

        let mask_ratio = msg.batch.mask_ratio as f64;
        let target_len = msg.batch.seq_length;
        let (input_ids, labels) = self.prepare_sequences(&msg.sequences, seed, mask_ratio, target_len)?;
        let participant_weight = (input_ids.dim(0)? * input_ids.dim(1)?) as f32;

        let (mut result, weight_delta) =
            self.train_step(&input_ids, &labels, batch_indices, self.learning_rate as f32, true, MGM1_LOCAL_STEPS)?;
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
pub(crate) fn masked_label_distribution(_input_ids: &Tensor, labels: &Tensor) -> Result<[usize; 4]> {
    let ignore = Tensor::new(MLM_IGNORE_INDEX, _input_ids.device())?.broadcast_as(labels.shape())?;
    let mask = labels.ne(&ignore)?.to_dtype(DType::F32)?;
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
pub(crate) fn masked_prediction_distribution(_input_ids: &Tensor, labels: &Tensor, logits: &Tensor) -> Result<[usize; 4]> {
    let ignore = Tensor::new(MLM_IGNORE_INDEX, _input_ids.device())?.broadcast_as(labels.shape())?;
    let mask = labels.ne(&ignore)?.to_dtype(DType::F32)?;
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
    use crate::trainer::MultiGpuConfig;
    use rand::seq::SliceRandom;

    fn default_config() -> Vec<u8> {
        serde_json::to_vec(&MiniGenomeConfig::default()).unwrap()
    }

    #[test]
    fn test_mgm1_balanced_local_training() {
        let config_json = serde_json::json!({
            "vocab_size": 8,
            "d_model": 64,
            "n_heads": 2,
            "n_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.0,
            "label_smoothing": 0.1,
        });
        let config = config_json.to_string().into_bytes();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-3, &MultiGpuConfig::default()).unwrap();

        // Simulate a balanced GC-stratified batch (25% each base) so the model
        // must learn to predict all four nucleotides, not collapse to the majority.
        let seq_len = 128;
        let batch_size = 16;
        let total = seq_len * batch_size;
        let per_base = total / 4;
        let mut raw = String::with_capacity(total);
        raw.extend(std::iter::repeat_n('A', per_base));
        raw.extend(std::iter::repeat_n('C', per_base));
        raw.extend(std::iter::repeat_n('G', per_base));
        raw.extend(std::iter::repeat_n('T', total - raw.len()));

        // Shuffle so the batch is not sorted by base; this avoids a positional
        // collapse where the model learns that late positions are always T.
        let mut chars: Vec<char> = raw.chars().collect();
        let mut shuffle_rng = ChaCha8Rng::from_seed([5u8; 32]);
        chars.shuffle(&mut shuffle_rng);
        let raw: String = chars.into_iter().collect();

        let sequences: Vec<String> = raw.as_bytes().chunks(seq_len).map(|c| String::from_utf8_lossy(c).to_string()).collect();

        let (input_ids, labels) = trainer.prepare_sequences(&sequences, [0u8; 32], MASK_RATIO, seq_len).unwrap();
        // Span masking makes the task harder; give the tiny test model more
        // local steps than production so the unit test stays stable.
        let (result, _) = trainer.train_step(&input_ids, &labels, vec![1], 1e-3, false, 16).unwrap();

        let (_, acc_after) = trainer.compute_loss_and_accuracy(&input_ids, &labels).unwrap();
        let logits = trainer.forward(&input_ids).unwrap();
        let pred_dist = masked_prediction_distribution(&input_ids, &labels, &logits).unwrap();
        let label_dist = masked_label_distribution(&input_ids, &labels).unwrap();

        println!("AT-rich local training: loss {:.4} -> {:.4}, acc {:.2}%", result.loss_before, result.loss_after, acc_after * 100.0);
        println!("labels A={:>3} C={:>3} G={:>3} T={:>3}", label_dist[0], label_dist[1], label_dist[2], label_dist[3]);
        println!("preds  A={:>3} C={:>3} G={:>3} T={:>3}", pred_dist[0], pred_dist[1], pred_dist[2], pred_dist[3]);

        assert!(result.loss_after < result.loss_before, "training did not reduce loss");
        assert!(pred_dist[1] > 0, "C predictions collapsed to zero");
        assert!(pred_dist[2] > 0, "G predictions collapsed to zero");
    }

    #[test]
    fn test_mgm1_trainer_loads_and_trains() {
        let config = default_config();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();

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
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();

        let batch = TrainingBatch {
            batch_id: 1,
            model_id: "xeno/mgm-1".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: (0..4).collect(),
            target_improvement: 0.01,
            learning_rate: 1e-4,
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
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();

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
        use candle_core::D;
        use candle_nn::ops::softmax;

        let config_json = serde_json::json!({
            "vocab_size": 8,
            "d_model": 128,
            "n_heads": 4,
            "n_layers": 4,
            "d_ff": 512,
            "max_seq_len": 256,
            "dropout": 0.1,
        });
        let config_bytes = config_json.to_string().into_bytes();
        let trainer = Mgm1Trainer::new("xeno/mgm-1", &config_bytes, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();

        // Synthetic batch matching the distribution from the user's log:
        // A=72 C=70 G=55 T=117 across ~314 masked positions.
        let mut raw = String::with_capacity(2048);
        let total_bases = 2048usize;
        let a_count = (total_bases as f32 * 72.0 / 314.0).round() as usize;
        let c_count = (total_bases as f32 * 70.0 / 314.0).round() as usize;
        let g_count = (total_bases as f32 * 55.0 / 314.0).round() as usize;
        let t_count = total_bases - a_count - c_count - g_count;
        raw.extend(std::iter::repeat_n('A', a_count));
        raw.extend(std::iter::repeat_n('C', c_count));
        raw.extend(std::iter::repeat_n('G', g_count));
        raw.extend(std::iter::repeat_n('T', t_count));

        let sequences: Vec<String> = raw.as_bytes().chunks(256).map(|c| String::from_utf8_lossy(c).to_string()).collect();
        let msg = GenomeTrainingBatchMsg {
            batch: GenomeTrainingBatch {
                batch_id: 1,
                model_id: "xeno/mgm-1".to_string(),
                genome_merkle_root: [0u8; 32],
                data_indices: sequences
                    .iter()
                    .enumerate()
                    .map(|(i, s)| GenomeSlice { chunk_idx: i as u64, start_base: 0, length: s.len() as u32 })
                    .collect(),
                mask_ratio: 0.15,
                seq_length: 256,
            },
            sequences,
            base_checkpoint: [0u8; 32],
        };

        let (input_ids, labels) =
            trainer.prepare_sequences(&msg.sequences, [0u8; 32], MASK_RATIO, trainer.config.max_seq_len).unwrap();
        let input_vec = input_ids.to_vec2::<i64>().unwrap();
        let label_vec = labels.to_vec2::<i64>().unwrap();

        let mut label_counts = HashMap::<i64, usize>::new();
        let mut mask_count = 0usize;
        for b in 0..input_vec.len() {
            for t in 0..input_vec[b].len() {
                if label_vec[b][t] != MLM_IGNORE_INDEX {
                    mask_count += 1;
                    *label_counts.entry(label_vec[b][t]).or_insert(0) += 1;
                }
            }
        }
        println!("\n[DIAGNOSE] Masked positions: {}", mask_count);
        println!("[DIAGNOSE] Label distribution among masked positions:");
        let base_name = |id: i64| -> char {
            match id {
                0 => 'A',
                1 => 'C',
                2 => 'G',
                3 => 'T',
                4 => '[',
                5 => ' ',
                6 => ']',
                7 => '|',
                _ => '?',
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
                if label_vec[b][t] != MLM_IGNORE_INDEX {
                    let true_id = label_vec[b][t];
                    println!(
                        "  batch[{}][{}] true={} pred={} | A={:.3} C={:.3} G={:.3} T={:.3}",
                        b,
                        t,
                        base_name(true_id),
                        base_name(pred_ids[b][t] as i64),
                        probs_vec[b][t][0],
                        probs_vec[b][t][1],
                        probs_vec[b][t][2],
                        probs_vec[b][t][3]
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
                if label_vec[b][t] != MLM_IGNORE_INDEX {
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

        // Train for a few steps on the same batch and observe whether loss decreases
        // and whether the prediction distribution stays balanced or collapses.
        println!("\n[DIAGNOSE] Training on same batch for 30 steps (no per-batch class weights)...");
        for step in 0..30 {
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
                if label_vec[b][t] != MLM_IGNORE_INDEX {
                    let pred = base_name(pred_ids_after[b][t] as i64);
                    *predictions_after.entry(pred).or_insert(0) += 1;
                    if pred == base_name(label_vec[b][t]) {
                        correct += 1;
                    }
                }
            }
        }
        println!("\n[DIAGNOSE] Prediction distribution after 30 steps (same batch):");
        for c in ['A', 'C', 'G', 'T'] {
            let n = predictions_after.get(&c).copied().unwrap_or(0);
            println!("  {}: {} ({:.1}%)", c, n, 100.0 * (n as f32) / mask_count.max(1) as f32);
        }
        println!(
            "[DIAGNOSE] Accuracy after 50 steps: {} / {} ({:.1}%)",
            correct,
            mask_count,
            100.0 * (correct as f32) / mask_count.max(1) as f32
        );

        let (final_loss, final_acc) = trainer.model.compute_mlm_loss(&input_ids, &labels, None).unwrap();
        println!(
            "[DIAGNOSE] Final loss = {:.6}, final accuracy = {:.2}%",
            final_loss.to_dtype(DType::F32).unwrap().to_vec0::<f32>().unwrap(),
            final_acc * 100.0
        );
    }

    #[test]
    #[ignore = "audit: reproduce MGM-1 collapse; run with -- --ignored --nocapture"]
    fn audit_mgm1_collapse_reproduction() {
        use candle_core::D as TensorD;
        use candle_nn::ops::softmax;
        fn target_counts(freqs: &[f32; 4], total: usize) -> [usize; 4] {
            let mut counts = [0usize; 4];
            let mut sum = 0;
            for i in 0..3 {
                counts[i] = (total as f32 * freqs[i]).round() as usize;
                sum += counts[i];
            }
            counts[3] = total - sum;
            counts
        }

        fn chars_for_counts(counts: [usize; 4]) -> String {
            let chars = ['A', 'C', 'G', 'T'];
            let mut s = String::with_capacity(counts.iter().sum());
            for (i, &n) in counts.iter().enumerate() {
                s.extend(std::iter::repeat_n(chars[i], n));
            }
            s
        }

        fn inspect(_trainer: &Mgm1Trainer, input_ids: &Tensor, labels: &Tensor, logits: &Tensor, prefix: &str) {
            let pred_dist = masked_prediction_distribution(input_ids, labels, logits).unwrap();
            let label_dist = masked_label_distribution(input_ids, labels).unwrap();
            println!("{} labels A={:>3} C={:>3} G={:>3} T={:>3}", prefix, label_dist[0], label_dist[1], label_dist[2], label_dist[3]);
            println!("{} preds  A={:>3} C={:>3} G={:>3} T={:>3}", prefix, pred_dist[0], pred_dist[1], pred_dist[2], pred_dist[3]);
            let input_vec = input_ids.to_vec2::<i64>().unwrap();
            let label_vec = labels.to_vec2::<i64>().unwrap();
            if let Some((b, t)) = input_vec.iter().enumerate().find_map(|(bi, _row)| {
                label_vec[bi].iter().enumerate().find_map(|(ti, &v)| if v != MLM_IGNORE_INDEX { Some((bi, ti)) } else { None })
            }) {
                let logits_4 = logits.narrow(TensorD::Minus1, 0, 4).unwrap();
                let probs = softmax(&logits_4, TensorD::Minus1).unwrap();
                let l4 = logits_4.to_vec3::<f32>().unwrap();
                let p4 = probs.to_vec3::<f32>().unwrap();
                let true_id = label_vec[b][t] as usize;
                println!(
                    "{} sample masked[{}][{}] true={} | A: {:7.3}/{:.3}  C: {:7.3}/{:.3}  G: {:7.3}/{:.3}  T: {:7.3}/{:.3}",
                    prefix,
                    b,
                    t,
                    true_id,
                    l4[b][t][0],
                    p4[b][t][0],
                    l4[b][t][1],
                    p4[b][t][1],
                    l4[b][t][2],
                    p4[b][t][2],
                    l4[b][t][3],
                    p4[b][t][3],
                );
            }
        }

        let config_json = serde_json::json!({
            "vocab_size": 8,
            "d_model": 64,
            "n_heads": 2,
            "n_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.0,
            "label_smoothing": 0.1,
        });
        let config_bytes = config_json.to_string().into_bytes();

        let config_no_smooth_json = serde_json::json!({
            "vocab_size": 8,
            "d_model": 64,
            "n_heads": 2,
            "n_layers": 2,
            "d_ff": 128,
            "max_seq_len": 128,
            "dropout": 0.0,
            "label_smoothing": 0.0,
        });
        let config_no_smooth_bytes = config_no_smooth_json.to_string().into_bytes();

        // Scenario A: an imbalanced batch where C is a minority.
        let freqs_a = [0.40f32, 0.15f32, 0.25f32, 0.20f32];
        let seq_len = 128usize;
        let batch_size = 4usize;
        let total_bases = seq_len * batch_size;
        let counts_a = target_counts(&freqs_a, total_bases);
        let raw_a = chars_for_counts(counts_a);
        let sequences_a: Vec<String> = raw_a.as_bytes().chunks(seq_len).map(|c| String::from_utf8_lossy(c).to_string()).collect();

        // Scenario B: perfectly balanced.
        let counts_b = [total_bases / 4, total_bases / 4, total_bases / 4, total_bases - 3 * (total_bases / 4)];
        let raw_b = chars_for_counts(counts_b);
        let sequences_b: Vec<String> = raw_b.as_bytes().chunks(seq_len).map(|c| String::from_utf8_lossy(c).to_string()).collect();

        println!("\n========== AUDIT: IMBALANCED BATCH (label smoothing 0.1, no class weights) ==========");
        let trainer_a = Mgm1Trainer::new("xeno/mgm-1", &config_bytes, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();
        let (input_a, labels_a) = trainer_a.prepare_sequences(&sequences_a, [0u8; 32], MASK_RATIO, seq_len).unwrap();
        let logits_init_a = trainer_a.forward(&input_a).unwrap();
        inspect(&trainer_a, &input_a, &labels_a, &logits_init_a, "init");
        for step in 0..50 {
            let (loss, acc, grads) = trainer_a.compute_gradients(&input_a, &labels_a, 1.0, None).unwrap();
            assert!(loss.is_finite(), "non-finite loss at step {}", step);
            for (name, g) in &grads {
                let g_vec = g.flatten_all().unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
                assert!(g_vec.iter().all(|v| v.is_finite()), "non-finite gradient {} at step {}", name, step);
            }
            trainer_a.apply_gradients(&grads, 1e-4).unwrap();
            if step % 10 == 0 || step == 49 {
                let logits = trainer_a.forward(&input_a).unwrap();
                inspect(&trainer_a, &input_a, &labels_a, &logits, &format!("step {step:>2}"));
                println!("  step {step:>2}: loss={:.6} acc={:.2}%", loss, acc * 100.0);
            }
        }

        println!("\n========== AUDIT: IMBALANCED BATCH (no label smoothing, no class weights) ==========");
        let trainer_b =
            Mgm1Trainer::new("xeno/mgm-1", &config_no_smooth_bytes, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();
        let (input_b, labels_b) = trainer_b.prepare_sequences(&sequences_a, [1u8; 32], MASK_RATIO, seq_len).unwrap();
        let logits_init_b = trainer_b.forward(&input_b).unwrap();
        inspect(&trainer_b, &input_b, &labels_b, &logits_init_b, "init");
        for step in 0..50 {
            let (loss, acc, grads) = trainer_b.compute_gradients(&input_b, &labels_b, 1.0, None).unwrap();
            assert!(loss.is_finite(), "non-finite loss at step {}", step);
            trainer_b.apply_gradients(&grads, 1e-4).unwrap();
            if step % 10 == 0 || step == 49 {
                let logits = trainer_b.forward(&input_b).unwrap();
                inspect(&trainer_b, &input_b, &labels_b, &logits, &format!("step {step:>2}"));
                println!("  step {step:>2}: loss={:.6} acc={:.2}%", loss, acc * 100.0);
            }
        }

        println!("\n========== AUDIT: BALANCED BATCH (label smoothing 0.1, no class weights) ==========");
        let trainer_c = Mgm1Trainer::new("xeno/mgm-1", &config_bytes, &[], Vec::new(), [0u8; 32], Device::Cpu, 1e-4, &MultiGpuConfig::default()).unwrap();
        let (input_c, labels_c) = trainer_c.prepare_sequences(&sequences_b, [2u8; 32], MASK_RATIO, seq_len).unwrap();
        let logits_init_c = trainer_c.forward(&input_c).unwrap();
        inspect(&trainer_c, &input_c, &labels_c, &logits_init_c, "init");
        for step in 0..50 {
            let (loss, acc, grads) = trainer_c.compute_gradients(&input_c, &labels_c, 1.0, None).unwrap();
            assert!(loss.is_finite(), "non-finite loss at step {}", step);
            trainer_c.apply_gradients(&grads, 1e-4).unwrap();
            if step % 10 == 0 || step == 49 {
                let logits = trainer_c.forward(&input_c).unwrap();
                inspect(&trainer_c, &input_c, &labels_c, &logits, &format!("step {step:>2}"));
                println!("  step {step:>2}: loss={:.6} acc={:.2}%", loss, acc * 100.0);
            }
        }
    }
}
