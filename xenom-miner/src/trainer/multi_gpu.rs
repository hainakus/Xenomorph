//! Multi-GPU data-parallel DNABERT-2 trainer for `xenom-miner`.
//!
//! This implementation replicates the DNABERT-2 model on every requested GPU,
//! splits the effective batch into micro-batches, computes gradients on each
//! device, gathers them on the master GPU for averaging, applies the result to
//! the master replica, and broadcasts the updated weights back to the other
//! replicas.
//!
//! NCCL, ZeRO and full gradient-checkpointing are intentionally left as
//! compile-time feature stubs; the master-gather path avoids the cross-GPU
//! CPU copy bottleneck while keeping activation memory per micro-batch small.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use tracing::{info, warn};

use crate::data::MlmBatch;
use crate::dnabert2::DnaBert2Model;
use crate::lora::LoraConfig;
use crate::model::DnaBert2Config;
use crate::models::checkpointed_forward::CheckpointedForward;
use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use crate::tokenizer::DnaTokenizer;
use crate::trainer::gpu_trainer::GpuBackend;
use crate::trainer::gradient::{add_grad_maps, average_grad_maps, build_gradient_update, gradient_commitment, move_grads_to_device};
use crate::trainer::mixed_precision::MixedPrecisionScaler;
use crate::trainer::DnaBert2Trainer;
use crate::trainer::{DeviceInfo, DeviceType, GradientUpdate, Trainer, TrainingResult};

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
    /// Top-k gradient compression ratio for FedAvg submissions. 1.0 = dense.
    pub gradient_top_k_ratio: f32,
    /// Optional LoRA configuration. If `None`, full fine-tuning is performed.
    pub lora_config: Option<LoraConfig>,
    /// Maximum sequence length per sample; capped at the model's position limit to save VRAM.
    pub max_seq_len: usize,
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
            gradient_top_k_ratio: 1.0,
            lora_config: None,
            max_seq_len: 512,
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
        if self.max_seq_len == 0 {
            bail!("max-seq-len must be > 0");
        }
        if self.zero_optimization > 0 {
            bail!("ZeRO optimization level > 0 is not yet implemented");
        }
        if self.gradient_top_k_ratio.is_nan() || self.gradient_top_k_ratio < 0.0 || self.gradient_top_k_ratio > 1.0 {
            bail!("gradient-top-k-ratio must be between 0.0 and 1.0");
        }
        if let Some(lora) = &self.lora_config {
            if lora.rank == 0 {
                bail!("LoRA rank must be > 0");
            }
            if lora.alpha <= 0.0 {
                bail!("LoRA alpha must be > 0");
            }
            if lora.dropout.is_nan() || lora.dropout < 0.0 || lora.dropout > 1.0 {
                bail!("LoRA dropout must be between 0.0 and 1.0");
            }
            if lora.target_modules.is_empty() {
                bail!("LoRA target_modules must not be empty");
            }
        }
        Ok(())
    }
}

/// Snapshot of one model replica's trainable variables, used to reset all
/// devices back to the shared base checkpoint after each local step.
struct BaseSnapshot {
    checkpoint: Option<[u8; 32]>,
    per_device: Vec<HashMap<String, Tensor>>,
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
    /// Base checkpoint snapshot; all devices reset here before each batch so
    /// consecutive batches train from the same FedAvg base.
    base: Mutex<BaseSnapshot>,
}

/// Result of computing gradients for a single GPU micro-batch.
struct MicroResult {
    /// Unscaled loss for this micro-batch, averaged over the masked positions.
    loss: f64,
    /// Number of masked positions in this micro-batch.
    weight: f32,
    /// Gradients moved to the master device, if this micro-batch produced any.
    grads: Option<HashMap<String, Tensor>>,
    /// Time spent in forward + backward for this micro-batch.
    compute_ms: u64,
    /// Time spent moving gradients to the master device.
    gather_ms: u64,
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

        // Cap sequence length at the requested maximum to control VRAM usage.
        let mut config = config;
        config.max_position_embeddings = config.max_position_embeddings.min(gpu_config.max_seq_len);
        if config.max_position_embeddings == 0 {
            config.max_position_embeddings = 1;
        }
        info!("Using effective sequence length: {}", config.max_position_embeddings);

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
                gpu_config.lora_config.clone(),
            )
            .with_context(|| format!("Failed to load DNABERT-2 replica on device {:?}", device))?;
            info!("Loaded DNABERT-2 replica {}/{} on device {:?}", idx + 1, devices.len(), device);
            trainers.push(Arc::new(trainer));
        }

        let device_info = Self::build_device_info(&devices, threads);
        let base = Mutex::new(BaseSnapshot { checkpoint: None, per_device: Vec::new() });

        Ok(Self { trainers, config: gpu_config, scaler, device_info, base })
    }

    /// Determine which physical devices to use based on the CLI config and
    /// requested backend.
    pub(crate) fn resolve_devices(gpu_config: &MultiGpuConfig, backend: GpuBackend) -> Result<Vec<Device>> {
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

    pub(crate) fn build_device_info(devices: &[Device], threads: usize) -> DeviceInfo {
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

    /// Snapshot the current weights of every replica as the base checkpoint.
    ///
    /// The snapshot is stored as a deep copy in F32 on the same device so that
    /// `Var::set` later accepts it as an independent tensor. This avoids a host
    /// memory round-trip that can OOM on multi-GPU hosts.
    fn snapshot_base(&self, checkpoint: [u8; 32]) -> Result<()> {
        let mut per_device = Vec::with_capacity(self.trainers.len());
        for (idx, trainer) in self.trainers.iter().enumerate() {
            let data = trainer.varmap.data().lock().map_err(|e| anyhow::anyhow!("Replica {} VarMap poisoned: {}", idx, e))?;
            let mut snap = HashMap::new();
            for (name, var) in data.iter() {
                let t = var.as_tensor();
                let t_f32 = t.to_dtype(DType::F32).with_context(|| format!("Failed to cast {} to F32 for snapshot", name))?;
                let restored = t_f32.copy().with_context(|| format!("Failed to copy snapshot tensor for {}", name))?;
                snap.insert(name.clone(), restored);
            }
            per_device.push(snap);
        }
        let mut base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {}", e))?;
        base.checkpoint = Some(checkpoint);
        base.per_device = per_device;
        Ok(())
    }

    /// Reset every replica to the stored base checkpoint and reset its optimizer state.
    fn restore_base(&self) -> Result<()> {
        let base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {}", e))?;
        if base.checkpoint.is_none() {
            return Ok(());
        }
        for (idx, (trainer, snapshot)) in self.trainers.iter().zip(base.per_device.iter()).enumerate() {
            trainer.reset_optimizer().with_context(|| format!("Failed to reset optimizer on replica {}", idx))?;
            let data = trainer.varmap.data().lock().map_err(|e| anyhow::anyhow!("Replica {} VarMap poisoned: {}", idx, e))?;
            for (name, var) in data.iter() {
                if let Some(snap) = snapshot.get(name) {
                    let target_dtype = var.as_tensor().dtype();
                    let target_device = var.as_tensor().device();
                    let snap = snap.to_dtype(target_dtype)?.to_device(target_device)?;
                    var.set(&snap).with_context(|| format!("Failed to restore {} on replica {}", name, idx))?;
                }
            }
        }
        Ok(())
    }

    /// Ensure the replica weights correspond to `checkpoint`. If this is the
    /// first time we see it, snapshot the current weights as the base. Otherwise
    /// restore the stored base so the batch trains from the same starting point.
    fn ensure_base(&self, checkpoint: [u8; 32]) -> Result<()> {
        let base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {}", e))?;
        if base.checkpoint == Some(checkpoint) {
            drop(base);
            return self.restore_base();
        }
        drop(base);
        self.snapshot_base(checkpoint)
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
        self.ensure_base(base_checkpoint).context("Failed to reset replicas to base checkpoint")?;

        let micro_batches =
            split_mlm_batch(mlm_batch, self.trainers.len(), self.config.gradient_accumulation_steps, self.config.micro_batch_size);

        // Only the first `usable` sequences are actually used for training. Compute
        // post-update loss on that exact slice so the values are comparable.
        let usable =
            mlm_batch.batch_size.min(self.config.micro_batch_size * self.trainers.len() * self.config.gradient_accumulation_steps);
        let used_batch = extract_mlm_batch(mlm_batch, 0, usable, mlm_batch.seq_len);
        let participant_weight = used_batch.mask.iter().filter(|&&m| m == 1).count() as f32;

        let master_device = self.trainers[0].device.clone();
        let mut accumulated_grads: Option<HashMap<String, Tensor>> = None;
        let mut any_overflow = false;

        // loss_before is accumulated from each GPU's micro-batch loss (weighted by the
        // number of masked positions). This avoids an extra full forward pass on the
        // master replica before the backward steps begin.
        let mut loss_before_sum: f64 = 0.0;
        let mut loss_before_weight: f32 = 0.0;

        let mut compute_ms: u64 = 0;
        let mut gather_ms: u64 = 0;
        let mut overflow_check_ms: u64 = 0;
        let mut avg_ms: u64 = 0;
        let mut add_ms: u64 = 0;

        // Snapshot the scaler at the start of the batch; it is updated at the end.
        let mut scaler = *self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))?;

        for step in &micro_batches {
            let step_start = Instant::now();
            let mut step_compute_ms: u64 = 0;
            let mut step_gather_ms: u64 = 0;

            let step_results: Vec<std::thread::Result<Result<MicroResult>>> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(step.len());
                for (gpu_idx, maybe_micro) in step.iter().enumerate() {
                    let Some(micro) = maybe_micro else { continue };
                    let trainer = self.trainers[gpu_idx].clone();
                    let master_device = master_device.clone();
                    let micro = micro.clone();
                    let loss_scale = scaler.scale();
                    let handle = s.spawn(move || -> Result<MicroResult> {
                        let weight = micro.mask.iter().filter(|&&m| m == 1).count() as f32;
                        if weight == 0.0 {
                            return Ok(MicroResult { loss: 0.0, weight: 0.0, grads: None, compute_ms: 0, gather_ms: 0 });
                        }

                        let compute_start = Instant::now();
                        let (loss, grads) = match trainer.compute_gradients(&micro, loss_scale) {
                            Ok(v) => v,
                            Err(e) => {
                                let error_string = e.to_string().to_lowercase();
                                warn!("GPU {} gradient computation failed: {:?}; skipping micro-batch", gpu_idx, e);
                                if error_string.contains("out of memory")
                                    || error_string.contains("oom")
                                    || error_string.contains("cuda")
                                {
                                    return Err(e);
                                }
                                return Ok(MicroResult { loss: 0.0, weight, grads: None, compute_ms: 0, gather_ms: 0 });
                            }
                        };
                        let compute_ms = compute_start.elapsed().as_millis() as u64;

                        if grads.is_empty() {
                            return Ok(MicroResult { loss, weight, grads: None, compute_ms, gather_ms: 0 });
                        }

                        let gather_start = Instant::now();
                        let grads = move_grads_to_device(grads, &master_device)
                            .with_context(|| format!("Failed to move gradients from GPU {} to master device", gpu_idx))?;
                        let gather_ms = gather_start.elapsed().as_millis() as u64;

                        Ok(MicroResult { loss, weight, grads: Some(grads), compute_ms, gather_ms })
                    });
                    handles.push(handle);
                }
                handles.into_iter().map(|h| h.join()).collect()
            });

            let mut step_grads = Vec::with_capacity(step_results.len());
            for result in step_results {
                let micro = result.map_err(|e| anyhow::anyhow!("GPU thread panicked: {:?}", e))??;
                if let Some(grads) = micro.grads {
                    step_compute_ms = step_compute_ms.max(micro.compute_ms);
                    step_gather_ms = step_gather_ms.max(micro.gather_ms);
                    loss_before_sum += micro.loss * micro.weight as f64;
                    loss_before_weight += micro.weight;
                    step_grads.push(grads);
                } else {
                    any_overflow = true;
                }
            }
            compute_ms += step_compute_ms;
            gather_ms += step_gather_ms;
            // The remaining wall time for the step (averaging / accumulation) is captured below.
            let _ = step_start.elapsed();

            if step_grads.is_empty() {
                continue;
            }

            let avg_start = Instant::now();
            let avg = average_grad_maps(&step_grads).context("Failed to average gradients across GPUs for an accumulation step")?;
            avg_ms += avg_start.elapsed().as_millis() as u64;

            let add_start = Instant::now();
            accumulated_grads = Some(match accumulated_grads {
                None => avg,
                Some(acc) => add_grad_maps(acc, avg)?,
            });
            add_ms += add_start.elapsed().as_millis() as u64;
        }

        // If every micro-batch overflowed, reduce the scale and bail so the next batch can retry.
        if accumulated_grads.is_none() {
            scaler.update_scale(true);
            *self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))? = scaler;
            bail!("All micro-batches overflowed; loss scale reduced. Will retry on next batch.");
        }

        let final_grads = accumulated_grads.unwrap();

        let loss_before = if loss_before_weight > 0.0 { loss_before_sum / loss_before_weight as f64 } else { 0.0 };

        let final_overflow_start = Instant::now();
        let had_overflow = if self.config.use_mixed_precision {
            let flat_grads: Vec<(String, Tensor)> = final_grads.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
            scaler.has_overflow(&flat_grads)?
        } else {
            false
        };
        overflow_check_ms += final_overflow_start.elapsed().as_millis() as u64;

        // Update the loss scale based on overflow status before deciding whether to
        // apply the step. If the averaged gradients are non-finite, skip the optimizer
        // step entirely and bail; the caller will retry with the reduced scale.
        any_overflow = any_overflow || had_overflow;
        scaler.update_scale(any_overflow);
        *self.scaler.lock().map_err(|e| anyhow::anyhow!("Mixed-precision scaler poisoned: {}", e))? = scaler;

        if had_overflow {
            bail!("Averaged gradients are non-finite; loss scale reduced to {}. Will retry on next batch.", scaler.scale());
        }

        let effective_lr = scaler.effective_learning_rate(learning_rate, MAX_LEARNING_RATE);

        let apply_start = Instant::now();
        self.trainers[0]
            .apply_gradients(&final_grads, effective_lr)
            .context("Failed to apply averaged gradients to master replica")?;
        let apply_ms = apply_start.elapsed().as_millis() as u64;

        // Compute post-update loss on the master replica using the same
        // sequences that were actually trained.
        let loss_after_start = Instant::now();
        let loss_after = self.trainers[0].compute_loss_scalar(&used_batch)?;
        let loss_after_ms = loss_after_start.elapsed().as_millis() as u64;

        let commitment_start = Instant::now();
        let gradients_commitment = gradient_commitment(&final_grads)?;
        let commitment_ms = commitment_start.elapsed().as_millis() as u64;

        // Restore all replicas to the base checkpoint so the next batch starts from the same point.
        // This keeps every GPU busy and avoids cross-GPU broadcast after every step.
        let restore_start = Instant::now();
        self.restore_base().context("Failed to restore replicas to base checkpoint")?;
        let restore_ms = restore_start.elapsed().as_millis() as u64;

        let total_ms = start.elapsed().as_millis() as u64;
        info!(
            "MultiGpuTrainer timings (ms): total={}, loss_before={:.6}, loss_after={:.6}, compute={}, gather={}, overflow_check={}, avg={}, add={}, apply={}, loss_after_ms={}, commitment={}, restore={}",
            total_ms, loss_before, loss_after, compute_ms, gather_ms, overflow_check_ms, avg_ms, add_ms, apply_ms, loss_after_ms, commitment_ms, restore_ms
        );

        let result = TrainingResult {
            model_id: model_id.to_string(),
            batch_indices,
            base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: total_ms,
        };

        Ok((result, final_grads, participant_weight))
    }
}

impl MultiGpuTrainer {
    /// Return the currently tracked base checkpoint, if any.
    pub fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        self.base.lock().ok()?.checkpoint
    }

    /// Load a new base checkpoint into all replica VarMaps and snapshot it.
    pub fn load_base_checkpoint(&self, checkpoint: [u8; 32], weights: &[u8]) -> Result<()> {
        for (idx, trainer) in self.trainers.iter().enumerate() {
            trainer
                .load_weights_from_bytes(weights)
                .with_context(|| format!("Failed to load base checkpoint into replica {}", idx))?;
        }
        self.snapshot_base(checkpoint)?;
        Ok(())
    }
}

impl Trainer for MultiGpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mlm_batch = self.trainers[0].generator.generate(batch).context("Failed to generate MLM batch")?;
        self.train_mlm_batch(&mlm_batch, &batch.model_id, batch.base_checkpoint, mlm_batch.batch_indices.clone(), batch.learning_rate)
            .map(|(result, _, _)| result)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mlm_batch = self.trainers[0].generator.generate(batch).context("Failed to generate MLM batch")?;
        let (result, named_grads, participant_weight) = self.train_mlm_batch(
            &mlm_batch,
            &batch.model_id,
            batch.base_checkpoint,
            mlm_batch.batch_indices.clone(),
            batch.learning_rate,
        )?;
        let build_start = Instant::now();
        let mut update_result = result;
        let update = build_gradient_update(
            &batch.model_id,
            batch.base_checkpoint,
            named_grads,
            participant_weight,
            self.config.gradient_top_k_ratio,
            &update_result,
            batch.batch_id,
            batch.learning_rate,
            [0u8; 32],
            Vec::new(),
        )?;
        update_result.gradients_commitment = update.gradients_commitment;
        info!("Gradient update build time: {} ms", build_start.elapsed().as_millis());
        Ok((update_result, Some(update)))
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        let batch = &msg.batch;
        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();
        let mlm_batch = self.trainers[0]
            .generator
            .generate_from_sequences_with_indices(&msg.sequences, &batch.genome_merkle_root, batch.batch_id, Some(&batch_indices))
            .context("Failed to generate MLM batch from genome sequences")?;

        self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, mlm_batch.batch_indices.clone(), 0.01)
            .map(|(result, _, _)| result)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let batch = &msg.batch;
        let batch_indices: Vec<u64> = batch.data_indices.iter().map(|slice| slice.chunk_idx).collect();
        let mlm_batch = self.trainers[0]
            .generator
            .generate_from_sequences_with_indices(&msg.sequences, &batch.genome_merkle_root, batch.batch_id, Some(&batch_indices))
            .context("Failed to generate MLM batch from genome sequences")?;

        let (mut result, named_grads, participant_weight) =
            self.train_mlm_batch(&mlm_batch, &batch.model_id, msg.base_checkpoint, mlm_batch.batch_indices.clone(), 0.01)?;
        let build_start = Instant::now();
        let update = build_gradient_update(
            &batch.model_id,
            msg.base_checkpoint,
            named_grads,
            participant_weight,
            self.config.gradient_top_k_ratio,
            &result,
            batch.batch_id,
            0.01,
            msg.batch.genome_merkle_root,
            msg.batch.data_indices.clone(),
        )?;
        result.gradients_commitment = update.gradients_commitment;
        info!("Genome gradient update build time: {} ms", build_start.elapsed().as_millis());
        Ok((result, Some(update)))
    }

    fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        self.current_base_checkpoint()
    }

    fn load_base_checkpoint(&self, base_checkpoint: [u8; 32], weights: &[u8]) -> Result<()> {
        self.load_base_checkpoint(base_checkpoint, weights)
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
    for step_row in grid.iter_mut().take(accumulation_steps) {
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
        for (gpu, cell) in step_row.iter_mut().enumerate().take(num_gpus) {
            let gpu_total = base + if gpu < extra { 1 } else { 0 };
            let gpu_total = gpu_total.min(micro_batch_size);
            if gpu_total == 0 {
                continue;
            }
            let end = (consumed + gpu_total).min(usable);
            *cell = Some(extract_mlm_batch(batch, consumed, end, seq_len));
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
        batch_indices: batch.batch_indices[start..end].to_vec(),
    }
}
