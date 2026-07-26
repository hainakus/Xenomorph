//! Multi-GPU data-parallel trainer for the Mini Genome Model (MGM-1).
//!
//! Replicates the MGM-1 model on every requested CUDA/Metal device, splits each
//! batch into micro-batches, computes gradients on each device, gathers and
//! averages them on the master device, applies the result to the master replica,
//! and restores every replica to the shared base checkpoint before the next batch.
//!
//! CPU-only execution is intentionally not supported; at least one CUDA or Metal
//! device must be available.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle_core::Tensor;
use tracing::{info, warn};

use crate::rpc::messages::{GenomeTrainingBatchMsg, GradientUpdate, TrainingBatch};
use crate::trainer::gpu_trainer::GpuBackend;
use crate::trainer::gradient::{add_grad_maps, average_grad_maps, build_gradient_update, gradient_commitment, move_grads_to_device};
use crate::trainer::mgm1_trainer::Mgm1Trainer;
use crate::trainer::multi_gpu::{MultiGpuConfig, MultiGpuTrainer};
use crate::trainer::{DeviceInfo, Trainer, TrainingResult};

/// Snapshot of each replica's trainable weights at the current base checkpoint.
struct Mgm1BaseSnapshot {
    checkpoint: Option<[u8; 32]>,
    per_device: Vec<HashMap<String, Tensor>>,
}

/// Result of computing gradients for one micro-batch.
struct MicroResult {
    loss: f64,
    weight: f32,
    grads: Option<HashMap<String, Tensor>>,
}

/// Multi-device data-parallel MGM-1 trainer.
pub struct Mgm1MultiGpuTrainer {
    trainers: Vec<Arc<Mgm1Trainer>>,
    config: MultiGpuConfig,
    device_info: DeviceInfo,
    base: Mutex<Mgm1BaseSnapshot>,
    model_id: String,
    lr: f64,
}

impl Mgm1MultiGpuTrainer {
    /// Build one MGM-1 replica on each requested GPU.
    pub fn new(
        model_id: impl Into<String>,
        config_bytes: &[u8],
        tokenizer_bytes: &[u8],
        weights: Vec<u8>,
        base_checkpoint: [u8; 32],
        gpu_config: MultiGpuConfig,
        backend: GpuBackend,
        lr: f64,
        threads: usize,
    ) -> Result<Self> {
        let model_id = model_id.into();
        gpu_config.validate()?;

        let mut devices = MultiGpuTrainer::resolve_devices(&gpu_config, backend)?;
        // NO CPU: multi-GPU MGM-1 requires at least one CUDA/Metal device.
        devices.retain(|d| !d.is_cpu());
        if devices.is_empty() {
            bail!("MGM-1 multi-GPU requires at least one CUDA/Metal device; CPU fallback is disabled");
        }

        if gpu_config.use_mixed_precision {
            warn!("MGM-1 multi-GPU ignores --fp16; mixed precision is not supported");
        }
        if gpu_config.zero_optimization != 0 {
            bail!("ZeRO optimization is not supported for MGM-1");
        }

        let mut trainers = Vec::with_capacity(devices.len());
        for (idx, device) in devices.iter().enumerate() {
            let trainer = Mgm1Trainer::new(
                model_id.clone(),
                config_bytes,
                tokenizer_bytes,
                weights.clone(),
                base_checkpoint,
                device.clone(),
                lr,
                gpu_config.gradient_top_k_ratio,
            )
            .with_context(|| format!("Failed to load MGM-1 replica {idx} on device {:?}", device))?;
            trainers.push(Arc::new(trainer));
        }

        let device_info = MultiGpuTrainer::build_device_info(&devices, threads);
        let base = Mutex::new(Mgm1BaseSnapshot { checkpoint: None, per_device: Vec::new() });

        Ok(Self { trainers, config: gpu_config, device_info, base, model_id, lr })
    }

    /// Snapshot the current weights of every replica as the base checkpoint.
    fn snapshot_base(&self, checkpoint: [u8; 32]) -> Result<()> {
        let mut per_device = Vec::with_capacity(self.trainers.len());
        for (idx, trainer) in self.trainers.iter().enumerate() {
            let snap = trainer.varmap_snapshot().with_context(|| format!("Failed to snapshot replica {}", idx))?;
            per_device.push(snap);
        }
        let mut base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {e}"))?;
        base.checkpoint = Some(checkpoint);
        base.per_device = per_device;
        Ok(())
    }

    /// Reset every replica to the stored base checkpoint and reset its optimizer state.
    fn restore_base(&self) -> Result<()> {
        let base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {e}"))?;
        if base.checkpoint.is_none() {
            return Ok(());
        }
        for (idx, (trainer, snapshot)) in self.trainers.iter().zip(base.per_device.iter()).enumerate() {
            trainer.reset_optimizer().with_context(|| format!("Failed to reset optimizer on replica {}", idx))?;
            trainer.restore_varmap(snapshot).with_context(|| format!("Failed to restore replica {}", idx))?;
        }
        Ok(())
    }

    /// Ensure replica weights correspond to `checkpoint`. If first time, snapshot
    /// current weights as base; otherwise restore to stored base.
    fn ensure_base(&self, checkpoint: [u8; 32]) -> Result<()> {
        let base = self.base.lock().map_err(|e| anyhow::anyhow!("Base snapshot mutex poisoned: {e}"))?;
        if base.checkpoint == Some(checkpoint) {
            drop(base);
            return self.restore_base();
        }
        drop(base);
        self.snapshot_base(checkpoint)
    }

    /// Split a (batch, seq_len) tensor pair into a `[accum_step][gpu]` grid of
    /// contiguous micro-batches. Cells that contain no data are `None`.
    #[allow(clippy::type_complexity, clippy::needless_range_loop)]
    fn split_batch(
        input_ids: &Tensor,
        labels: &Tensor,
        num_gpus: usize,
        micro_batch_size: usize,
        accumulation_steps: usize,
    ) -> Result<Vec<Vec<Option<(Tensor, Tensor)>>>> {
        let batch_size = input_ids.dim(0)?;
        let max_usable = num_gpus.saturating_mul(micro_batch_size).saturating_mul(accumulation_steps);
        let usable = batch_size.min(max_usable);

        let mut grid: Vec<Vec<Option<(Tensor, Tensor)>>> = vec![vec![None; num_gpus]; accumulation_steps];
        for step in 0..accumulation_steps {
            for gpu in 0..num_gpus {
                let start = step * num_gpus * micro_batch_size + gpu * micro_batch_size;
                if start >= usable {
                    continue;
                }
                let len = micro_batch_size.min(usable - start);
                let ids = input_ids.narrow(0, start, len)?;
                let lbls = labels.narrow(0, start, len)?;
                grid[step][gpu] = Some((ids, lbls));
            }
        }
        Ok(grid)
    }

    /// Train on an already-materialised (input_ids, labels) tensor pair.
    fn train_batch(
        &self,
        input_ids_full: &Tensor,
        labels_full: &Tensor,
        base_checkpoint: [u8; 32],
        batch_indices: Vec<u64>,
        learning_rate: f32,
    ) -> Result<(TrainingResult, HashMap<String, Tensor>, f32)> {
        let start = Instant::now();
        self.ensure_base(base_checkpoint).context("Failed to reset replicas to base checkpoint")?;

        let num_gpus = self.trainers.len();
        let micro_batch_size = self.config.micro_batch_size;
        let accumulation_steps = self.config.gradient_accumulation_steps.max(1);

        let grid = Self::split_batch(input_ids_full, labels_full, num_gpus, micro_batch_size, accumulation_steps)?;
        let batch_size = input_ids_full.dim(0)?;
        let max_usable = num_gpus.saturating_mul(micro_batch_size).saturating_mul(accumulation_steps);
        let usable = batch_size.min(max_usable);

        let master_device = self.trainers[0].device().clone();
        let used_input_ids = input_ids_full.narrow(0, 0, usable)?;
        let used_labels = labels_full.narrow(0, 0, usable)?;

        let mut accumulated_grads: Option<HashMap<String, Tensor>> = None;
        let mut loss_before_sum: f64 = 0.0;
        let mut loss_before_weight: f32 = 0.0;

        for step in grid {
            let step_results: Vec<std::thread::Result<Result<MicroResult>>> = std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(step.len());
                for (gpu_idx, maybe_micro) in step.into_iter().enumerate() {
                    let Some((ids, lbls)) = maybe_micro else { continue };
                    let trainer = self.trainers[gpu_idx].clone();
                    let master_device = master_device.clone();
                    let handle = s.spawn(move || -> Result<MicroResult> {
                        let ids = ids.to_device(trainer.device())?;
                        let lbls = lbls.to_device(trainer.device())?;
                        let (loss, grads) = trainer
                            .compute_gradients(&ids, &lbls, 1.0)
                            .with_context(|| format!("Gradient computation failed on GPU {}", gpu_idx))?;
                        let weight = ids.elem_count() as f32;
                        let grads = move_grads_to_device(grads, &master_device)
                            .with_context(|| format!("Failed to move gradients from GPU {} to master", gpu_idx))?;
                        Ok(MicroResult { loss, weight, grads: Some(grads) })
                    });
                    handles.push(handle);
                }
                handles.into_iter().map(|h| h.join()).collect()
            });

            let mut step_grads = Vec::with_capacity(step_results.len());
            for result in step_results {
                let micro = result.map_err(|e| anyhow::anyhow!("GPU thread panicked: {:?}", e))??;
                if let Some(grads) = micro.grads {
                    loss_before_sum += micro.loss * micro.weight as f64;
                    loss_before_weight += micro.weight;
                    step_grads.push(grads);
                }
            }

            if step_grads.is_empty() {
                continue;
            }

            let avg = average_grad_maps(&step_grads).context("Failed to average gradients across GPUs")?;
            accumulated_grads = Some(match accumulated_grads {
                None => avg,
                Some(acc) => add_grad_maps(acc, avg)?,
            });
        }

        let final_grads = accumulated_grads.ok_or_else(|| anyhow::anyhow!("No gradients were produced by any GPU"))?;
        let loss_before = if loss_before_weight > 0.0 { loss_before_sum / loss_before_weight as f64 } else { 0.0 };

        self.trainers[0]
            .apply_gradients(&final_grads, learning_rate)
            .context("Failed to apply averaged gradients to master replica")?;

        let used_input_ids = used_input_ids.to_device(&master_device)?;
        let used_labels = used_labels.to_device(&master_device)?;
        let loss_after = self.trainers[0]
            .compute_loss_scalar(&used_input_ids, &used_labels)
            .context("Failed to compute post-update loss on master replica")?;

        let gradients_commitment = gradient_commitment(&final_grads)?;

        // Restore replicas to base so the next batch starts from the same point.
        self.restore_base().context("Failed to restore replicas to base checkpoint")?;

        let total_ms = start.elapsed().as_millis() as u64;
        info!(
            "Mgm1MultiGpuTrainer loss_before={:.6} loss_after={:.6} devices={} total_ms={}",
            loss_before, loss_after, num_gpus, total_ms
        );

        let result = TrainingResult {
            model_id: self.model_id.clone(),
            batch_indices,
            base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: total_ms,
        };

        Ok((result, final_grads, loss_before_weight))
    }

    /// Encrypt and package the averaged gradients as a `GradientUpdate`.
    fn build_gradient_update_for(
        &self,
        base_checkpoint: [u8; 32],
        named_grads: HashMap<String, Tensor>,
        participant_weight: f32,
    ) -> Result<GradientUpdate> {
        build_gradient_update(&self.model_id, base_checkpoint, named_grads, participant_weight, self.config.gradient_top_k_ratio)
    }
}

impl Trainer for Mgm1MultiGpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.trainers[0].prepare_random(n, seed)?;
        let (result, _, _) =
            self.train_batch(&input_ids, &labels, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)?;
        Ok(result)
    }

    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        seed[..8].copy_from_slice(&batch.batch_id.to_le_bytes());

        let n = batch.data_indices.len().max(1);
        let (input_ids, labels) = self.trainers[0].prepare_random(n, seed)?;
        let (result, final_grads, participant_weight) =
            self.train_batch(&input_ids, &labels, batch.base_checkpoint, batch.data_indices.clone(), batch.learning_rate)?;
        let update = self.build_gradient_update_for(batch.base_checkpoint, final_grads, participant_weight)?;
        Ok((result, Some(update)))
    }

    fn train_genome(&self, msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        if msg.sequences.is_empty() {
            anyhow::bail!("Genome batch contains no sequences");
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&msg.base_checkpoint);
        let batch_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();

        let (input_ids, labels) = self.trainers[0].prepare_sequences(&msg.sequences, seed)?;
        let (result, _, _) = self.train_batch(&input_ids, &labels, msg.base_checkpoint, batch_indices, self.lr as f32)?;
        Ok(result)
    }

    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        if msg.sequences.is_empty() {
            anyhow::bail!("Genome batch contains no sequences");
        }

        let mut seed = [0u8; 32];
        seed.copy_from_slice(&msg.base_checkpoint);
        let batch_indices: Vec<u64> = msg.batch.data_indices.iter().map(|s| s.chunk_idx).collect();

        let (input_ids, labels) = self.trainers[0].prepare_sequences(&msg.sequences, seed)?;
        let (result, final_grads, participant_weight) =
            self.train_batch(&input_ids, &labels, msg.base_checkpoint, batch_indices, self.lr as f32)?;
        let update = self.build_gradient_update_for(msg.base_checkpoint, final_grads, participant_weight)?;
        Ok((result, Some(update)))
    }

    fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        self.base.lock().ok()?.checkpoint
    }

    fn load_base_checkpoint(&self, checkpoint: [u8; 32], weights: &[u8]) -> Result<()> {
        for (idx, trainer) in self.trainers.iter().enumerate() {
            trainer
                .load_base_checkpoint(checkpoint, weights)
                .with_context(|| format!("Failed to load base checkpoint into replica {}", idx))?;
        }
        self.snapshot_base(checkpoint)?;
        Ok(())
    }

    fn device_info(&self) -> DeviceInfo {
        self.device_info.clone()
    }
}
