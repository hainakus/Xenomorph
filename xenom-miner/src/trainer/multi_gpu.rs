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
use candle_core::{Device, Tensor};
use tracing::{info, warn};

use crate::data::MlmBatch;
use crate::dnabert2::DnaBert2Model;
use crate::model::DnaBert2Config;
use crate::models::checkpointed_forward::CheckpointedForward;
use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::gpu_trainer::GpuBackend;
use crate::trainer::mixed_precision::{to_grad_dtype, MixedPrecisionScaler};
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};
use crate::trainer::DnaBert2Trainer;

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
            let trainer = DnaBert2Trainer::new(
                config.clone(),
                replica_weights,
                tokenizer.clone(),
                device.clone(),
                threads,
                dtype,
            )
            .with_context(|| format!("Failed to load DNABERT-2 replica on device {:?}", device))?;
            info!("Loaded DNABERT-2 replica {}/{} on device {:?}", idx + 1, devices.len(), device);
            trainers.push(Arc::new(trainer));
        }

        let device_info = Self::build_device_info(&devices, threads);

        Ok(Self {
            trainers,
            config: gpu_config,
            scaler,
            device_info,
        })
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
    ) -> Result<TrainingResult> {
        let start = Instant::now();

        let micro_batches = split_mlm_batch(
            mlm_batch,
            self.trainers.len(),
            self.config.gradient_accumulation_steps,
            self.config.micro_batch_size,
        );

        let mut accumulated_grads: Option<HashMap<String, Tensor>> = None;
        let mut loss_sum = 0.0f64;
        let mut loss_count = 0usize;
        let mut any_overflow = false;

        for step in &micro_batches {
            let mut step_grads = Vec::new();
            let mut step_losses = Vec::new();

            for (gpu_idx, maybe_micro) in step.iter().enumerate() {
                let Some(micro) = maybe_micro else { continue };
                let trainer = &self.trainers[gpu_idx];
                let loss_scale = self
                    .scaler
                    .lock()
                    .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
                    .scale();

                let (loss, grads) = match trainer.compute_gradients(micro, loss_scale) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("GPU {} gradient computation overflowed: {}; skipping micro-batch", gpu_idx, e);
                        any_overflow = true;
                        continue;
                    }
                };

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

                step_losses.push(loss);
                step_grads.push(grads);
            }

            if step_grads.is_empty() {
                continue;
            }

            let avg = average_grad_maps(&step_grads)
                .context("Failed to average gradients across GPUs for an accumulation step")?;
            accumulated_grads = Some(match accumulated_grads {
                None => avg,
                Some(acc) => add_grad_maps(acc, avg)?,
            });

            loss_sum += step_losses.iter().sum::<f64>();
            loss_count += step_losses.len();
        }

        // If every micro-batch overflowed, reduce the scale and bail so the next batch can retry.
        if accumulated_grads.is_none() {
            self.scaler
                .lock()
                .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
                .update_scale(true);
            bail!("All micro-batches overflowed; loss scale reduced. Will retry on next batch.");
        }

        let final_grads = accumulated_grads.unwrap();
        let avg_loss_before = loss_sum / loss_count.max(1) as f64;

        let flat_grads: Vec<(String, Tensor)> = final_grads.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let had_overflow = self
            .scaler
            .lock()
            .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
            .has_overflow(&flat_grads)?;

        let effective_lr = self
            .scaler
            .lock()
            .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
            .effective_learning_rate(learning_rate.min(MAX_LEARNING_RATE));

        // Only update weights when the averaged gradients are finite.
        if !had_overflow {
            self.trainers[0]
                .apply_gradients(&final_grads, effective_lr)
                .context("Failed to apply averaged gradients to master replica")?;
            self.broadcast_master_to_others()
                .context("Failed to broadcast updated weights to replica GPUs")?;
        } else {
            warn!("Averaged gradients are non-finite; skipping optimizer step and reducing loss scale");
            any_overflow = true;
        }

        // Update the loss scale based on overflow status.
        self.scaler
            .lock()
            .map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?
            .update_scale(any_overflow);

        // Compute post-update loss on the master replica using the first
        // valid micro-batch.
        let first_micro = micro_batches
            .iter()
            .flat_map(|step| step.iter())
            .flatten()
            .next()
            .unwrap_or(mlm_batch);
        let loss_after = self.trainers[0].compute_loss_scalar(first_micro)?;
        let gradients_commitment = self.trainers[0].gradient_commitment_from_named_tensors(final_grads)?;

        Ok(TrainingResult {
            model_id: model_id.to_string(),
            batch_indices,
            base_checkpoint,
            loss_before: avg_loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
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
                let replica_var = replica_data
                    .get(name)
                    .with_context(|| format!("Replica {} is missing variable {}", idx, name))?;
                let updated = master_var
                    .as_tensor()
                    .to_device(&replica.device)?
                    .contiguous()?;
                replica_var.set(&updated)?;
            }
        }

        Ok(())
    }
}

impl Trainer for MultiGpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mlm_batch = self.trainers[0]
            .generator
            .generate(batch)
            .context("Failed to generate MLM batch")?;
        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        let batch = &msg.batch;
        let mlm_batch = self.trainers[0]
            .generator
            .generate_from_sequences(&msg.sequences, &batch.genome_merkle_root, batch.batch_id)
            .context("Failed to generate MLM batch from genome sequences")?;

        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();
        self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, batch_indices, 0.01)
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

/// Split a flat `MlmBatch` into a 2-D grid `[accum_step][gpu] -> Option<MlmBatch>`.
///
/// The grid is filled row-major. Each non-None cell holds at most
/// `micro_batch_size` sequences. If the input has more sequences than the grid
/// can hold, the tail is dropped with a warning. If it has fewer, some cells
/// are `None`.
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
        warn!(
            "Batch size {} exceeds usable grid capacity {}; truncating to {}",
            total, max_usable, max_usable
        );
    }

    let mut consumed = 0usize;
    for step in 0..accumulation_steps {
        for gpu in 0..num_gpus {
            if consumed >= total || consumed >= max_usable {
                break;
            }
            let end = (consumed + micro_batch_size).min(total).min(max_usable);
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
fn average_grad_maps(grads: &[HashMap<String, Tensor>]) -> Result<HashMap<String, Tensor>> {
    if grads.is_empty() {
        bail!("Cannot average empty gradient list");
    }
    let n = grads.len() as f64;
    let mut out = HashMap::new();

    // Use the first map as the key set.
    let first = &grads[0];
    for (name, base) in first.iter() {
        let mut sum = to_grad_dtype(base)?;
        for other in &grads[1..] {
            let g = other
                .get(name)
                .with_context(|| format!("Gradient map missing variable {}", name))?;
            let g = to_grad_dtype(g)?;
            sum = (&sum + &g)?;
        }
        let avg = (&sum / n)?;
        out.insert(name.clone(), avg);
    }

    Ok(out)
}

/// Element-wise addition of two gradient maps with the same keys.
fn add_grad_maps(a: HashMap<String, Tensor>, b: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(a.len());
    for (name, ta) in a {
        let tb = b.get(&name).with_context(|| format!("Missing gradient for variable {}", name))?;
        let ta = to_grad_dtype(&ta)?;
        let tb = to_grad_dtype(tb)?;
        out.insert(name, (&ta + &tb)?);
    }
    Ok(out)
}
