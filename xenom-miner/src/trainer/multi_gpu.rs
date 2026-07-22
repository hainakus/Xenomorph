//! Multi-GPU data-parallel DNABERT-2 trainer for `xenom-miner`.
//!
//! This implementation replicates the DNABERT-2 model on every requested GPU,
//! splits the effective batch into micro-batches, computes gradients on each
//! device, averages them on the CPU, applies the result to the master replica,
//! and broadcasts the updated weights back to the other replicas.
//!
//! NCCL, ZeRO and full gradient-checkpointing are intentionally left as
//! compile-time feature stubs; the CPU-averaging path is correct and avoids
//! the single-GPU OOM by reducing activation memory per micro-batch.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use borsh::to_vec as borsh_to_vec;
use candle_core::{Device, Tensor};
use tracing::{info, warn};

use crate::data::MlmBatch;
use crate::dnabert2::DnaBert2Model;
use crate::model::DnaBert2Config;
use crate::models::checkpointed_forward::CheckpointedForward;
use crate::rpc::messages::{GenomeTrainingBatchMsg, GradientLayer, GradientPayload, GradientUpdate, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::gpu_trainer::GpuBackend;
use crate::trainer::mixed_precision::{to_grad_dtype, MixedPrecisionScaler};
use crate::trainer::DnaBert2Trainer;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

/// Learning-rate cap inherited from `DnaBert2Trainer`.
const MAX_LEARNING_RATE: f32 = 1e-5;

/// Configuration controlling how many GPUs are used and how the batch is split.
#[derive(Debug, Clone)]
pub struct MultiGpuConfig {
    /// List of GPU device ordinals to use (e.g. `[0, 1, 2]`).
    pub gpus: Vec<usize>,
    /// Micro-batch size per GPU per accumulation step.
    pub micro_batch_size: usize,
    /// Number of gradient-accumulation steps per optimizer step.
    pub gradient_accumulation_steps: usize,
    /// Enable FP16 mixed precision.
    pub use_mixed_precision: bool,
    /// Enable gradient checkpointing (currently a no-op stub).
    pub use_gradient_checkpointing: bool,
    /// ZeRO level (0 = none, higher values reserved).
    pub zero_optimization: u8,
}

impl Default for MultiGpuConfig {
    fn default() -> Self {
        Self {
            gpus: vec![0],
            micro_batch_size: 4,
            gradient_accumulation_steps: 1,
            use_mixed_precision: false,
            use_gradient_checkpointing: false,
            zero_optimization: 0,
        }
    }
}

impl MultiGpuConfig {
    pub fn validate(&self) -> Result<()> {
        if self.gpus.is_empty() {
            bail!("At least one GPU must be specified");
        }
        if self.micro_batch_size == 0 {
            bail!("micro-batch-size must be > 0");
        }
        if self.gradient_accumulation_steps == 0 {
            bail!("gradient-accumulation-steps must be > 0");
        }
        if self.zero_optimization > 0 {
            bail!("ZeRO optimization level > 0 is not yet implemented");
        }
        Ok(())
    }
}

/// A data-parallel DNABERT-2 trainer that can span multiple CUDA/Metal/CPU
/// devices.
pub struct MultiGpuTrainer {
    /// One `DnaBert2Trainer` replica per device.
    trainers: Vec<Arc<DnaBert2Trainer>>,
    /// Training configuration.
    config: MultiGpuConfig,
    /// Mixed-precision loss scaler.
    scaler: Mutex<MixedPrecisionScaler>,
    /// Cached device info for logging.
    device_info: DeviceInfo,
}

impl MultiGpuTrainer {
    /// Build one model replica on each requested device.
    pub fn new(
        _model_id: String,
        config: DnaBert2Config,
        weights: Vec<u8>,
        tokenizer: DnaTokenizer,
        gpu_config: MultiGpuConfig,
        backend: GpuBackend,
        threads: usize,
    ) -> Result<Self> {
        let devices = Self::resolve_devices(&gpu_config, backend)?;
        if devices.is_empty() {
            bail!("No usable devices for multi-GPU training");
        }

        let scaler = Mutex::new(MixedPrecisionScaler::new(gpu_config.use_mixed_precision));
        let dtype = match scaler.lock() {
            Ok(guard) => guard.compute_dtype(),
            Err(e) => {
                warn!("Mixed-precision scaler mutex poisoned: {}; defaulting to F32", e);
                e.into_inner().compute_dtype()
            }
        };

        let mut trainers = Vec::with_capacity(devices.len());
        for (idx, device) in devices.iter().enumerate() {
            // Each replica needs its own copy of the weights so it can build a
            // VarMap on its device.
            let replica_weights = weights.clone();
            let trainer = DnaBert2Trainer::new(config.clone(), replica_weights, tokenizer.clone(), device.clone(), threads, dtype)
                .with_context(|| format!("Failed to load DNABERT-2 replica on device {:?}", device))?;
            info!("Loaded DNABERT-2 replica {}/{} on device {:?}", idx + 1, devices.len(), device);
            trainers.push(Arc::new(trainer));
        }

        let device_info = Self::build_device_info(&devices, threads);

        Ok(Self { trainers, config: gpu_config, scaler, device_info })
    }

    /// Determine which physical devices to use based on the CLI config and
    /// requested backend.
    fn resolve_devices(gpu_config: &MultiGpuConfig, backend: GpuBackend) -> Result<Vec<Device>> {
        let mut devices = Vec::new();
        let _ = gpu_config.gpus.len();

        match backend {
            GpuBackend::Cuda | GpuBackend::Auto => {
                #[cfg(feature = "cuda")]
                for &idx in &gpu_config.gpus {
                    match Device::new_cuda(idx) {
                        Ok(dev) => devices.push(dev),
                        Err(e) => {
                            if backend == GpuBackend::Cuda {
                                bail!("CUDA device {} is not available: {}", idx, e);
                            }
                            warn!("Skipping CUDA device {}: {}", idx, e);
                        }
                    }
                }
                #[cfg(not(feature = "cuda"))]
                if backend == GpuBackend::Cuda {
                    bail!("CUDA support was not compiled into this binary");
                }
            }
            _ => {}
        }

        if devices.is_empty() && matches!(backend, GpuBackend::Metal | GpuBackend::Auto) {
            #[cfg(feature = "metal")]
            for &idx in &gpu_config.gpus {
                match Device::new_metal(idx) {
                    Ok(dev) => devices.push(dev),
                    Err(e) => warn!("Skipping Metal device {}: {}", idx, e),
                }
            }
            #[cfg(not(feature = "metal"))]
            warn!("Metal support was not compiled into this binary");
        }

        if devices.is_empty() {
            if backend == GpuBackend::Cuda || backend == GpuBackend::Metal {
                bail!("Requested {:?} backend but no device was usable", backend);
            }
            warn!("No GPU available; training on CPU with a single replica");
            devices.push(Device::Cpu);
        }

        Ok(devices)
    }

    fn build_device_info(devices: &[Device], threads: usize) -> DeviceInfo {
        let mut device_type = DeviceType::Cpu;
        let mut name = format!("Multi-GPU CPU trainer ({} threads)", threads);

        if devices.iter().any(|d| d.is_cuda()) {
            device_type = DeviceType::Cuda;
            name = format!("Multi-GPU CUDA trainer, {} device(s), {} threads", devices.len(), threads);
        } else if devices.iter().any(|d| d.is_metal()) {
            device_type = DeviceType::Metal;
            name = format!("Multi-GPU Metal trainer, {} device(s), {} threads", devices.len(), threads);
        }

        DeviceInfo { device_type, name, threads, ..Default::default() }
    }

    /// Train on an already-materialised `MlmBatch`.
    fn train_mlm_batch(
        &self,
        mlm_batch: &MlmBatch,
        model_id: &str,
        base_checkpoint: [u8; 32],
        batch_indices: Vec<u64>,
        learning_rate: f32,
    ) -> Result<(TrainingResult, HashMap<String, Tensor>, f32)> {
        let start = Instant::now();

        let micro_batches =
            split_mlm_batch(mlm_batch, self.trainers.len(), self.config.gradient_accumulation_steps, self.config.micro_batch_size);

        // Only the first `usable` sequences are actually used for training. Compute
        // pre/post loss on that exact slice so the values are comparable.
        let usable =
            mlm_batch.batch_size.min(self.config.micro_batch_size * self.trainers.len() * self.config.gradient_accumulation_steps);
        let used_batch = extract_mlm_batch(mlm_batch, 0, usable, mlm_batch.seq_len);
        let participant_weight = used_batch.mask.iter().filter(|&&m| m == 1).count() as f32;

        let loss_before = self.trainers[0].compute_loss_scalar(&used_batch)?;

        let mut accumulated_grads: Option<HashMap<String, Tensor>> = None;
        let mut any_overflow = false;

        for step in &micro_batches {
            let mut step_grads = Vec::new();

            for (gpu_idx, maybe_micro) in step.iter().enumerate() {
                let Some(micro) = maybe_micro else { continue };
                let trainer = &self.trainers[gpu_idx];
                let loss_scale = self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?.scale();

                let grads = match trainer.compute_gradients(micro, loss_scale) {
                    Ok((_, grads)) => grads,
                    Err(e) => {
                        warn!("GPU {} gradient computation failed: {}; skipping micro-batch", gpu_idx, e);
                        any_overflow = true;
                        continue;
                    }
                };

                // Skip micro-batches that produced no usable gradients (e.g. no masked positions).
                if grads.is_empty() {
                    warn!("GPU {} produced no gradients; skipping micro-batch", gpu_idx);
                    continue;
                }

                // Reject this micro-batch if the gradients themselves are non-finite.
                let flat_grads: Vec<(String, Tensor)> = grads.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                let step_had_overflow = self
                    .scaler
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
                    .has_overflow(&flat_grads)?;
                if step_had_overflow {
                    warn!("GPU {} produced non-finite gradients; skipping micro-batch", gpu_idx);
                    any_overflow = true;
                    continue;
                }

                step_grads.push(grads);
            }

            if step_grads.is_empty() {
                continue;
            }

            let avg = average_grad_maps(&step_grads).context("Failed to average gradients across GPUs for an accumulation step")?;
            accumulated_grads = Some(match accumulated_grads {
                None => avg,
                Some(acc) => add_grad_maps(acc, avg)?,
            });
        }

        // If every micro-batch overflowed, reduce the scale and bail so the next batch can retry.
        if accumulated_grads.is_none() {
            self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?.update_scale(true);
            bail!("All micro-batches overflowed; loss scale reduced. Will retry on next batch.");
        }

        let final_grads = accumulated_grads.unwrap();

        let flat_grads: Vec<(String, Tensor)> = final_grads.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let had_overflow =
            self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?.has_overflow(&flat_grads)?;

        let effective_lr = self
            .scaler
            .lock()
            .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
            .effective_learning_rate(learning_rate, MAX_LEARNING_RATE);

        // Only update weights when the averaged gradients are finite.
        if !had_overflow {
            self.trainers[0]
                .apply_gradients(&final_grads, effective_lr)
                .context("Failed to apply averaged gradients to master replica")?;
            self.broadcast_master_to_others().context("Failed to broadcast updated weights to replica GPUs")?;
        } else {
            warn!("Averaged gradients are non-finite; skipping optimizer step and reducing loss scale");
            any_overflow = true;
        }

        // Update the loss scale based on overflow status.
        self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?.update_scale(any_overflow);

        // Compute post-update loss on the master replica using the same
        // sequences that were actually trained.
        let loss_after = self.trainers[0].compute_loss_scalar(&used_batch)?;
        let gradients_commitment = self.trainers[0].gradient_commitment_from_named_tensors(&final_grads)?;

        let result = TrainingResult {
            model_id: model_id.to_string(),
            batch_indices,
            base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        };

        Ok((result, final_grads, participant_weight))
    }

    /// Copy updated parameters from the master (device 0) replica to all other
    /// replicas. This is a CPU-mediated broadcast; NCCL can replace it later.
    fn broadcast_master_to_others(&self) -> Result<()> {
        if self.trainers.len() <= 1 {
            return Ok(());
        }

        let master = &self.trainers[0];
        let master_data = master.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;

        for (idx, replica) in self.trainers.iter().enumerate().skip(1) {
            let replica_data = replica.varmap.data().lock().map_err(|e| anyhow::anyhow!("VarMap poisoned: {}", e))?;
            for (name, master_var) in master_data.iter() {
                let replica_var = replica_data.get(name).with_context(|| format!("Replica {} is missing variable {}", idx, name))?;
                let updated = master_var.as_tensor().to_device(&replica.device)?.contiguous()?;
                replica_var.set(&updated)?;
            }
        }

        Ok(())
    }
}

impl MultiGpuTrainer {
    /// Encrypt and package the averaged gradients as a `GradientUpdate` ready to
    /// be sent to the seed-node's FedAvg aggregator.
    fn build_gradient_update(
        &self,
        model_id: &str,
        base_checkpoint: [u8; 32],
        named_grads: HashMap<String, Tensor>,
        participant_weight: f32,
    ) -> Result<GradientUpdate> {
        let top_k_ratio = gradient_top_k_ratio();

        let mut layer_gradients = HashMap::with_capacity(named_grads.len());
        for (name, grad) in named_grads {
            let shape = grad.dims().to_vec();
            let flat = grad.flatten_all()?.to_vec1::<f32>()?;
            let (values, indices) =
                if top_k_ratio >= 1.0 { (flat, Vec::new()) } else { top_k_compress(&flat, top_k_ratio.clamp(0.0, 1.0)) };
            layer_gradients.insert(name, GradientLayer { values, shape, indices });
        }

        let payload = GradientPayload { layer_gradients };
        let payload_bytes = borsh_to_vec(&payload).context("Failed to serialize gradient payload")?;
        let encrypted_payload = model_crypto::encrypt(&payload_bytes, &model_crypto::derive_encryption_key())
            .context("Failed to encrypt gradient payload")?;

        Ok(GradientUpdate { model_id: model_id.to_string(), base_checkpoint, encrypted_payload, participant_weight })
    }
}

/// Read the global top-k gradient compression ratio. `1.0` means no compression
/// (dense gradients); `0.1` keeps the top 10 % absolute values.
fn gradient_top_k_ratio() -> f32 {
    std::env::var("XENO_GRADIENT_TOP_K_RATIO").ok().and_then(|s| s.parse::<f32>().ok()).unwrap_or(1.0).clamp(0.0, 1.0)
}

/// Keep only the `k` largest absolute values of `flat` and return them together
/// with their flattened indices (sorted ascending by index).
fn top_k_compress(flat: &[f32], ratio: f32) -> (Vec<f32>, Vec<usize>) {
    if ratio <= 0.0 || flat.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let k = ((flat.len() as f32 * ratio).ceil() as usize).clamp(1, flat.len());

    let mut indexed: Vec<(usize, f32)> = flat.iter().copied().enumerate().collect();
    // Partially sort by descending absolute value and keep the top k.
    indexed.select_nth_unstable_by(k - 1, |a, b| b.1.abs().total_cmp(&a.1.abs()).then_with(|| b.0.cmp(&a.0)));
    let mut top: Vec<(usize, f32)> = indexed.into_iter().take(k).collect();
    top.sort_by(|a, b| a.0.cmp(&b.0));

    let indices = top.iter().map(|(i, _)| *i).collect();
    let values = top.iter().map(|(_, v)| *v).collect();
    (values, indices)
}

impl Trainer for MultiGpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mlm_batch = self.trainers[0].generator.generate(batch).context("Failed to generate MLM batch")?;
        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)
            .map(|(result, _, _)| result)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mlm_batch = self.trainers[0].generator.generate(batch).context("Failed to generate MLM batch")?;
        let (result, named_grads, participant_weight) =
            self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)?;
        let update = self.build_gradient_update(&batch.model_id, batch.base_checkpoint, named_grads, participant_weight)?;
        Ok((result, Some(update)))
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        let batch = &msg.batch;
        let mlm_batch = self.trainers[0]
            .generator
            .generate_from_sequences(&msg.sequences, &batch.genome_merkle_root, batch.batch_id)
            .context("Failed to generate MLM batch from genome sequences")?;

        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();
        self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, batch_indices, 0.01).map(|(result, _, _)| result)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let batch = &msg.batch;
        let mlm_batch = self.trainers[0]
            .generator
            .generate_from_sequences(&msg.sequences, &batch.genome_merkle_root, batch.batch_id)
            .context("Failed to generate MLM batch from genome sequences")?;

        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();
        let (result, named_grads, participant_weight) =
            self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, batch_indices, 0.01)?;
        let update = self.build_gradient_update(&batch.model_id, msg.base_checkpoint, named_grads, participant_weight)?;
        Ok((result, Some(update)))
    }

    fn device_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }
}

impl CheckpointedForward for DnaBert2Model {
    fn forward_with_checkpointing(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        if self.checkpointing_state() {
            // Real gradient checkpointing is not yet implemented; delegate to
            // the standard forward pass.
            warn!("Gradient checkpointing requested but not implemented; using standard forward");
        }
        Ok(self.forward(input_ids, None, Some(attention_mask))?)
    }
}

impl DnaBert2Model {
    // Helper used by the trait impl above. Always returns false for now.
    fn checkpointing_state(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_compress_keeps_largest_absolute_values() {
        let flat = vec![1.0f32, -5.0, 2.0, 0.1, -3.0, 4.0];
        let (values, indices) = top_k_compress(&flat, 0.5);
        assert_eq!(values.len(), 3);
        assert_eq!(indices.len(), 3);

        let mut reconstructed = vec![0.0f32; flat.len()];
        for (i, idx) in indices.iter().enumerate() {
            reconstructed[*idx] = values[i];
        }
        // Top 3 absolute values are -5, -3 and 4.
        assert_eq!(reconstructed, vec![0.0, -5.0, 0.0, 0.0, -3.0, 4.0]);
    }

    #[test]
    fn test_top_k_compress_full_ratio_returns_sorted_identity() {
        let flat = vec![1.0f32, -5.0, 2.0, 0.1, -3.0, 4.0];
        let (values, indices) = top_k_compress(&flat, 1.0);
        assert_eq!(values.len(), flat.len());
        assert_eq!(indices, (0..flat.len()).collect::<Vec<_>>());
        assert_eq!(values, flat);
    }
}

/// Split a flat `MlmBatch` into a 2-D grid `[accum_step][gpu] -> Option<MlmBatch>`.
///
/// Sequences are distributed evenly across the GPUs in each accumulation step,
/// keeping each cell contiguous and respecting `micro_batch_size`. This avoids
/// the old row-major behaviour where the first GPU(s) would eat the whole batch
/// and leave the remaining GPUs idle.
fn split_mlm_batch(
    batch: &MlmBatch,
    num_gpus: usize,
    accumulation_steps: usize,
    micro_batch_size: usize,
) -> Vec<Vec<Option<MlmBatch>>> {
    let seq_len = batch.seq_len;
    let total = batch.batch_size;
    let mut grid: Vec<Vec<Option<MlmBatch>>> = vec![vec![None; num_gpus]; accumulation_steps];

    if total == 0 || micro_batch_size == 0 || num_gpus == 0 || accumulation_steps == 0 {
        return grid;
    }

    let max_usable = micro_batch_size.saturating_mul(num_gpus).saturating_mul(accumulation_steps);
    if total > max_usable {
        warn!("Batch size {} exceeds usable grid capacity {}; truncating to {}", total, max_usable, max_usable);
    }

    let mut consumed = 0usize;
    let usable = total.min(max_usable);
    for step in 0..accumulation_steps {
        if consumed >= usable {
            break;
        }
        let remaining = usable - consumed;
        let step_capacity = num_gpus.saturating_mul(micro_batch_size);
        let step_total = remaining.min(step_capacity);

        // Spread `step_total` sequences over the GPUs in this step as evenly as
        // possible while never exceeding `micro_batch_size` per GPU.
        let base = step_total / num_gpus;
        let extra = step_total % num_gpus;
        for gpu in 0..num_gpus {
            let gpu_total = base + if gpu < extra { 1 } else { 0 };
            let gpu_total = gpu_total.min(micro_batch_size);
            if gpu_total == 0 {
                continue;
            }
            let end = (consumed + gpu_total).min(usable);
            grid[step][gpu] = Some(extract_mlm_batch(batch, consumed, end, seq_len));
            consumed = end;
        }
    }

    grid
}

/// Extract a contiguous slice of sequences from a flat `MlmBatch`.
fn extract_mlm_batch(batch: &MlmBatch, start: usize, end: usize, seq_len: usize) -> MlmBatch {
    let start_flat = start * seq_len;
    let end_flat = end * seq_len;
    MlmBatch {
        input_ids: batch.input_ids[start_flat..end_flat].to_vec(),
        token_type_ids: batch.token_type_ids[start_flat..end_flat].to_vec(),
        attention_mask: batch.attention_mask[start_flat..end_flat].to_vec(),
        labels: batch.labels[start_flat..end_flat].to_vec(),
        mask: batch.mask[start_flat..end_flat].to_vec(),
        seq_len,
        batch_size: end - start,
    }
}

/// Average a list of per-GPU gradient maps. All tensors are expected to be on
/// the CPU and in F32.
///
/// The maps may have different key sets (for example because a micro-batch had
/// no masked positions or because tied weights produced gradients on a single
/// tensor). Each key is averaged over the gradient maps that actually contain it.
fn average_grad_maps(grads: &[HashMap<String, Tensor>]) -> Result<HashMap<String, Tensor>> {
    if grads.is_empty() {
        bail!("Cannot average empty gradient list");
    }

    // Accumulate the sum and per-key count so missing keys do not poison the average.
    let mut acc: HashMap<String, (Tensor, usize)> = HashMap::new();
    for g in grads {
        for (name, t) in g.iter() {
            let t = to_grad_dtype(t)?;
            match acc.get_mut(name) {
                Some((sum, count)) => {
                    *sum = (&*sum + &t)?;
                    *count += 1;
                }
                None => {
                    acc.insert(name.clone(), (t, 1));
                }
            }
        }
    }

    let mut out = HashMap::with_capacity(acc.len());
    for (name, (sum, count)) in acc {
        let avg = (&sum / (count as f64))?;
        out.insert(name, avg);
    }

    Ok(out)
}

/// Element-wise addition of two gradient maps. Keys present in only one map are
/// kept unchanged, so accumulated gradients survive accumulation steps where a
/// micro-batch did not contribute to every variable.
fn add_grad_maps(a: HashMap<String, Tensor>, b: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(a.len().max(b.len()));
    for (name, ta) in a {
        if let Some(tb) = b.get(&name) {
            let ta = to_grad_dtype(&ta)?;
            let tb = to_grad_dtype(tb)?;
            out.insert(name, (&ta + &tb)?);
        } else {
            out.insert(name, ta);
        }
    }
    for (name, tb) in b {
        if !out.contains_key(&name) {
            out.insert(name, tb);
        }
    }
    Ok(out)
}
