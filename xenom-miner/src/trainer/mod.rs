pub mod cpu_trainer;
pub mod dnabert2_trainer;
pub mod gpu_trainer;
pub mod mixed_precision;
pub mod mock_trainer;
pub mod multi_gpu;

#[cfg(feature = "cuda")]
pub mod cuda_kernels;

#[cfg(test)]
pub mod gpu_tests;

use crate::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use anyhow::Result;

pub use cpu_trainer::CpuTrainer;
pub use dnabert2_trainer::{DnaBert2Trainer, ManualAdamW};
pub use gpu_trainer::{GpuBackend, GpuTrainer};
pub use mixed_precision::MixedPrecisionScaler;
pub use mock_trainer::MockTrainer;
pub use multi_gpu::{MultiGpuConfig, MultiGpuTrainer};

pub use crate::rpc::messages::{GradientLayer, GradientPayload, GradientUpdate};

/// Information about the training device being used.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: String,
    pub threads: usize,
    /// GPU memory currently in use, in bytes.
    pub memory_used: Option<u64>,
    /// GPU temperature in degrees Celsius.
    pub temperature: Option<u32>,
    /// GPU compute utilization percentage (0-100).
    pub utilization: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum DeviceType {
    #[default]
    Cpu,
    Mock,
    Cuda,
    Metal,
    Rocm,
}

/// Result of training a single batch.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingResult {
    pub model_id: String,
    pub batch_indices: Vec<u64>,
    pub base_checkpoint: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
    pub compute_time_ms: u64,
}

/// Trait implemented by every training backend.
pub trait Trainer: Send + Sync {
    /// Train one batch and return the result.
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult>;

    /// Train one batch and also return an encrypted gradient update for FedAvg.
    /// Trainers that cannot extract gradients (e.g. CPU or mock) return `None`.
    fn train_with_gradients(&self, batch: &TrainingBatch) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        Ok((self.train(batch)?, None))
    }

    /// Train on a genome-backed batch.
    ///
    /// The default implementation returns an error; real trainers (e.g.
    /// `DnaBert2Trainer`) override this to tokenize the provided DNA sequences.
    fn train_genome(&self, _msg: &GenomeTrainingBatchMsg) -> Result<TrainingResult> {
        Err(anyhow::anyhow!("Genome training is not supported by this trainer"))
    }

    /// Train on a genome-backed batch and also return a gradient update for FedAvg.
    fn train_genome_with_gradients(&self, msg: &GenomeTrainingBatchMsg) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        Ok((self.train_genome(msg)?, None))
    }

    /// Return information about the device being used.
    fn device_info(&self) -> DeviceInfo;

    /// Return the base checkpoint this trainer is currently training from, if known.
    fn current_base_checkpoint(&self) -> Option<[u8; 32]> {
        None
    }

    /// Load a new base checkpoint into the trainer. Trainers that do not support
    /// hot-reloading (e.g. mock/CPU) ignore this call.
    fn load_base_checkpoint(&self, _base_checkpoint: [u8; 32], _weights: &[u8]) -> Result<()> {
        Ok(())
    }
}
