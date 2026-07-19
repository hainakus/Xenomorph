use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use crate::rpc::messages::TrainingBatch;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

const MOCK_LOSS_START: f64 = 2.5;
const MOCK_LOSS_IMPROVEMENT_MIN: f64 = 0.01;
const MOCK_LOSS_IMPROVEMENT_MAX: f64 = 0.05;

/// Fast deterministic trainer used for dry-runs and integration tests.
pub struct MockTrainer;

impl Default for MockTrainer {
    fn default() -> Self {
        Self
    }
}

impl MockTrainer {
    pub fn new() -> Self {
        Self
    }
}

impl Trainer for MockTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let start = Instant::now();

        let mut hasher = blake3::Hasher::new();
        hasher.update(&batch.batch_id.to_le_bytes());
        hasher.update(batch.model_id.as_bytes());
        hasher.update(&batch.base_checkpoint);
        for idx in &batch.data_indices {
            hasher.update(&idx.to_le_bytes());
        }
        let seed_hash = hasher.finalize();

        let mut rng = ChaCha8Rng::from_seed(*seed_hash.as_bytes());
        let improvement = MOCK_LOSS_IMPROVEMENT_MIN + rng.gen::<f64>() * (MOCK_LOSS_IMPROVEMENT_MAX - MOCK_LOSS_IMPROVEMENT_MIN);

        let loss_before = MOCK_LOSS_START + rng.gen::<f64>() * 0.2;
        let loss_after = (loss_before - improvement).max(0.0);

        let mut commitment_hasher = blake3::Hasher::new();
        commitment_hasher.update(&loss_before.to_le_bytes());
        commitment_hasher.update(&loss_after.to_le_bytes());
        commitment_hasher.update(seed_hash.as_bytes());
        let gradients_commitment = *commitment_hasher.finalize().as_bytes();

        Ok(TrainingResult {
            model_id: batch.model_id.clone(),
            batch_indices: batch.data_indices.clone(),
            base_checkpoint: batch.base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Mock,
            name: "Mock CPU trainer".to_string(),
            threads: 1,
            ..Default::default()
        }
    }
}
