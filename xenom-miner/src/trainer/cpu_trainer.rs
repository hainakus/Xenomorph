use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::Instant;

use crate::rpc::messages::TrainingBatch;
use crate::trainer::{DeviceInfo, DeviceType, Trainer, TrainingResult};

const INPUT_DIM: usize = 64;
const HIDDEN_DIM: usize = 32;
const OUTPUT_DIM: usize = 1;
const FEATURE_NOISE_SCALE: f32 = 0.1;

/// Real CPU trainer that performs one SGD step using Candle tensors.
pub struct CpuTrainer {
    threads: usize,
    device: Device,
}

impl CpuTrainer {
    pub fn new(threads: usize) -> Result<Self> {
        let device = Device::Cpu;
        Ok(Self { threads, device })
    }

    fn generate_sample(rng: &mut ChaCha8Rng, index: u64) -> (Vec<f32>, f32) {
        let mut features = Vec::with_capacity(INPUT_DIM);
        for i in 0..INPUT_DIM {
            let base = ((index.wrapping_mul((i + 1) as u64) % 1000) as f32 / 1000.0) * 2.0 - 1.0;
            let noise: f32 = rng.gen_range(-FEATURE_NOISE_SCALE..FEATURE_NOISE_SCALE);
            features.push(base + noise);
        }

        // A simple synthetic target: weighted sum of first few features + bias.
        let mut target = 0.0f32;
        for (i, &f) in features.iter().enumerate().take(8) {
            let weight = ((i + 1) as f32) * 0.05;
            target += f * weight;
        }
        target += 0.5;

        (features, target)
    }

    fn update_gradient_hash(hasher: &mut blake3::Hasher, tensor: &Tensor) -> Result<()> {
        let values = tensor.flatten_all()?.to_vec1::<f32>()?;
        for value in values {
            hasher.update(&value.to_le_bytes());
        }
        Ok(())
    }

    fn build_tensors(&self, batch: &TrainingBatch) -> Result<(Tensor, Tensor, ChaCha8Rng)> {
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&batch.base_checkpoint);
        let mut rng = ChaCha8Rng::from_seed(seed);

        let indices = if batch.data_indices.is_empty() { vec![batch.batch_id] } else { batch.data_indices.clone() };

        let n = indices.len();
        let mut xs_data = Vec::with_capacity(n * INPUT_DIM);
        let mut ys_data = Vec::with_capacity(n * OUTPUT_DIM);

        for &idx in &indices {
            let (features, target) = Self::generate_sample(&mut rng, idx);
            xs_data.extend_from_slice(&features);
            ys_data.push(target);
        }

        let xs = Tensor::from_vec(xs_data, (n, INPUT_DIM), &self.device).with_context(|| "Failed to create input tensor")?;
        let ys = Tensor::from_vec(ys_data, (n, OUTPUT_DIM), &self.device).with_context(|| "Failed to create target tensor")?;

        Ok((xs, ys, rng))
    }

    fn model_forward(&self, xs: &Tensor, w1: &Tensor, b1: &Tensor, w2: &Tensor, b2: &Tensor) -> Result<Tensor> {
        let hidden = xs.matmul(w1)?.broadcast_add(b1)?.relu()?;
        let logits = hidden.matmul(w2)?.broadcast_add(b2)?;
        Ok(logits)
    }

    fn compute_loss(&self, pred: &Tensor, ys: &Tensor) -> Result<Tensor> {
        let loss = pred.broadcast_sub(ys)?.sqr()?.mean_all()?;
        Ok(loss)
    }
}

impl Trainer for CpuTrainer {
    fn train(&self, batch: &TrainingBatch) -> Result<TrainingResult> {
        let start = Instant::now();

        let (xs, ys, _rng) = self.build_tensors(batch)?;
        let n = xs.dim(0)?;
        let n_tensor = Tensor::new(&[n as f32], &self.device)?;

        // Initialize small model weights.
        let mut w1 = Tensor::randn(0.0f32, 0.1f32, (INPUT_DIM, HIDDEN_DIM), &self.device)?;
        let mut b1 = Tensor::zeros((1, HIDDEN_DIM), DType::F32, &self.device)?;
        let mut w2 = Tensor::randn(0.0f32, 0.1f32, (HIDDEN_DIM, OUTPUT_DIM), &self.device)?;
        let mut b2 = Tensor::zeros((1, OUTPUT_DIM), DType::F32, &self.device)?;

        // Forward pass before training to compute initial loss.
        let pred_before = self.model_forward(&xs, &w1, &b1, &w2, &b2)?;
        let loss_before_tensor = self.compute_loss(&pred_before, &ys)?;
        let loss_before = loss_before_tensor.to_vec0::<f32>()? as f64;

        let lr = batch.learning_rate as f64;

        // Backpropagation for the last linear layer.
        let error = pred_before.broadcast_sub(&ys)?;

        // Gradients for w2: hidden^T * error / n.
        let hidden = xs.matmul(&w1)?.broadcast_add(&b1)?.relu()?;
        let d_w2 = hidden.transpose(0, 1)?.matmul(&error)?.broadcast_div(&n_tensor)?;
        let d_b2 = error.sum(0)?.broadcast_div(&n_tensor)?;

        // Update last layer.
        let scaled_w2 = d_w2.affine(lr, 0.0)?;
        w2 = w2.broadcast_sub(&scaled_w2)?;
        let scaled_b2 = d_b2.affine(lr, 0.0)?.reshape((1, OUTPUT_DIM))?;
        b2 = b2.broadcast_sub(&scaled_b2)?;

        // Gradients for w1: xs^T * (error * w2^T * relu') / n.
        // relu' is 1 where hidden > 0, 0 otherwise.
        let error_hidden = error.matmul(&w2.transpose(0, 1)?)?;
        let zeros = Tensor::zeros_like(&hidden)?;
        let relu_mask = hidden.gt(&zeros)?;
        let masked_error = error_hidden.broadcast_mul(&relu_mask.to_dtype(DType::F32)?)?;
        let d_w1 = xs.transpose(0, 1)?.matmul(&masked_error)?.broadcast_div(&n_tensor)?;
        let d_b1 = masked_error.sum(0)?.broadcast_div(&n_tensor)?;

        // Update first layer.
        let scaled_w1 = d_w1.affine(lr, 0.0)?;
        w1 = w1.broadcast_sub(&scaled_w1)?;
        let scaled_b1 = d_b1.affine(lr, 0.0)?.reshape((1, HIDDEN_DIM))?;
        b1 = b1.broadcast_sub(&scaled_b1)?;

        // Forward pass after the SGD step.
        let pred_after = self.model_forward(&xs, &w1, &b1, &w2, &b2)?;
        let loss_after_tensor = self.compute_loss(&pred_after, &ys)?;
        let loss_after = loss_after_tensor.to_vec0::<f32>()? as f64;

        // Commit to the computed gradients.
        let mut commitment_hasher = blake3::Hasher::new();
        Self::update_gradient_hash(&mut commitment_hasher, &d_w1)?;
        Self::update_gradient_hash(&mut commitment_hasher, &d_b1)?;
        Self::update_gradient_hash(&mut commitment_hasher, &d_w2)?;
        Self::update_gradient_hash(&mut commitment_hasher, &d_b2)?;
        let gradients_commitment = *commitment_hasher.finalize().as_bytes();

        Ok(TrainingResult {
            model_id: batch.model_id.clone(),
            batch_indices: batch.data_indices.clone(),
            base_checkpoint: batch.base_checkpoint,
            loss_before,
            loss_after,
            gradients_commitment,
            new_checkpoint: None,
            compute_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            device_type: DeviceType::Cpu,
            name: format!("Candle CPU ({} threads)", self.threads),
            threads: self.threads,
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_cpu_trainer_runs_and_improves() {
        let trainer = CpuTrainer::new(2).unwrap();
        let result = trainer.train(&dummy_batch()).unwrap();

        assert_eq!(result.model_id, "dnabert2");
        assert!(result.loss_after < result.loss_before);
        assert!(!result.gradients_commitment.iter().all(|&b| b == 0));
    }
}
