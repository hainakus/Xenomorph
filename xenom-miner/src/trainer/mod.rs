pub mod cpu_trainer;
pub mod dnabert2_trainer;
pub mod mock_trainer;

use crate::rpc::messages::TrainingBatch;
use anyhow::Result;

pub use cpu_trainer::CpuTrainer;
pub use dnabert2_trainer::DnaBert2Trainer;
pub use mock_trainer::MockTrainer;

/// Information about the training device being used.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceInfo {
    pub device_type: DeviceType,
    pub name: String,
    pub threads: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    Cpu,
    Mock,
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

    /// Return information about the device being used.
    fn device_info(&self) -> DeviceInfo;
}
