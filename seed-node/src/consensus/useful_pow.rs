use blake3;
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;
use tracing::info;

#[derive(Error, Debug)]
pub enum UsefulPoWError {
    #[error("Invalid training data")]
    InvalidTrainingData,

    #[error("Model hash mismatch")]
    ModelHashMismatch,

    #[error("Insufficient training rounds")]
    InsufficientRounds,

    #[error("Proof verification failed")]
    ProofVerificationFailed,

    #[error("Gradient computation error: {0}")]
    GradientError(String),

    #[error("Checkpoint validation failed")]
    CheckpointValidationFailed,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct TrainingProof {
    pub block_height: u64,
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub new_checkpoint: [u8; 32],
    pub gradients_hash: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub batch_size: u32,
    pub rounds: u32,
    pub timestamp: u64,
    pub nonce: u64,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct GradientUpdate {
    pub model_id: String,
    pub layer_name: String,
    pub gradient_data: Vec<f32>,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub batch_size: u32,
    pub epochs: u32,
    pub optimizer: String,
    pub momentum: Option<f32>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { learning_rate: 0.001, batch_size: 32, epochs: 10, optimizer: "adam".to_string(), momentum: Some(0.9) }
    }
}

pub struct UsefulPoWEngine {
    min_training_rounds: u32,
    max_training_rounds: u32,
    min_loss_improvement: f64,
}

impl UsefulPoWEngine {
    pub fn new() -> Self {
        Self { min_training_rounds: 100, max_training_rounds: 1000, min_loss_improvement: 0.01 }
    }

    pub fn with_config(min_rounds: u32, max_rounds: u32, min_improvement: f64) -> Self {
        Self { min_training_rounds: min_rounds, max_training_rounds: max_rounds, min_loss_improvement: min_improvement }
    }

    pub async fn validate_training_proof(&self, proof: &TrainingProof) -> Result<bool, UsefulPoWError> {
        // Verify minimum rounds
        if proof.rounds < self.min_training_rounds {
            return Err(UsefulPoWError::InsufficientRounds);
        }

        // Verify loss improvement
        let improvement = proof.loss_before - proof.loss_after;
        if improvement < self.min_loss_improvement {
            return Err(UsefulPoWError::ProofVerificationFailed);
        }

        // Verify gradient hash
        let expected_hash = self.compute_gradient_hash(&proof.model_id, proof.batch_size, proof.rounds);
        if expected_hash != proof.gradients_hash {
            return Err(UsefulPoWError::ProofVerificationFailed);
        }

        Ok(true)
    }

    pub async fn simulate_training(
        &self,
        model_id: &str,
        base_checkpoint: &[u8],
        config: &TrainingConfig,
    ) -> Result<TrainingProof, UsefulPoWError> {
        let start = Instant::now();

        // Simulate gradient computation
        let rounds = self.min_training_rounds + (self.max_training_rounds - self.min_training_rounds) / 2;
        let loss_before = 0.8 + (rand::random::<f64>() * 0.2);
        let loss_after = loss_before - (self.min_loss_improvement + rand::random::<f64>() * 0.05);

        // Compute gradient hash
        let gradients_hash = self.compute_gradient_hash(model_id, config.batch_size, rounds);

        // Simulate new checkpoint
        let new_checkpoint = self.compute_new_checkpoint(base_checkpoint, &gradients_hash);

        let proof = TrainingProof {
            block_height: 0, // Will be set by consensus
            model_id: model_id.to_string(),
            base_checkpoint: {
                let mut hash = [0u8; 32];
                hash.copy_from_slice(&blake3::hash(base_checkpoint).as_bytes()[..32]);
                hash
            },
            new_checkpoint,
            gradients_hash,
            loss_before,
            loss_after,
            batch_size: config.batch_size,
            rounds,
            timestamp: chrono::Utc::now().timestamp() as u64,
            nonce: rand::random(),
        };

        info!("Training simulation completed in {:?}", start.elapsed());
        Ok(proof)
    }

    pub fn compute_gradient_hash(&self, model_id: &str, batch_size: u32, rounds: u32) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(model_id.as_bytes());
        hasher.update(&batch_size.to_le_bytes());
        hasher.update(&rounds.to_le_bytes());
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_bytes());
        result
    }

    pub fn compute_new_checkpoint(&self, base: &[u8], gradient_hash: &[u8; 32]) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(base);
        hasher.update(gradient_hash);
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_bytes());
        result
    }

    pub fn serialize_proof(&self, proof: &TrainingProof) -> Result<Vec<u8>, UsefulPoWError> {
        borsh::to_vec(proof).map_err(|_e| UsefulPoWError::InvalidTrainingData)
    }

    pub fn deserialize_proof(&self, data: &[u8]) -> Result<TrainingProof, UsefulPoWError> {
        <TrainingProof as BorshDeserialize>::try_from_slice(data).map_err(|_e| UsefulPoWError::InvalidTrainingData)
    }
}

impl Default for UsefulPoWEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_proof_serialization() {
        let proof = TrainingProof {
            block_height: 1000,
            model_id: "test_model".to_string(),
            base_checkpoint: [1u8; 32],
            new_checkpoint: [2u8; 32],
            gradients_hash: [3u8; 32],
            loss_before: 0.8,
            loss_after: 0.7,
            batch_size: 32,
            rounds: 100,
            timestamp: 1000,
            nonce: 42,
        };

        let engine = UsefulPoWEngine::new();
        let serialized = engine.serialize_proof(&proof).unwrap();
        let deserialized = engine.deserialize_proof(&serialized).unwrap();

        assert_eq!(proof.model_id, deserialized.model_id);
        assert_eq!(proof.rounds, deserialized.rounds);
    }

    #[test]
    fn test_gradient_hash_computation() {
        let engine = UsefulPoWEngine::new();
        let hash1 = engine.compute_gradient_hash("model1", 32, 100);
        let hash2 = engine.compute_gradient_hash("model1", 32, 100);
        let hash3 = engine.compute_gradient_hash("model2", 32, 100);

        assert_eq!(hash1, hash2);
        assert_ne!(hash1, hash3);
    }

    #[test]
    fn test_checkpoint_computation() {
        let engine = UsefulPoWEngine::new();
        let base = [1u8; 32];
        let gradient_hash = [2u8; 32];

        let checkpoint1 = engine.compute_new_checkpoint(&base, &gradient_hash);
        let checkpoint2 = engine.compute_new_checkpoint(&base, &gradient_hash);

        assert_eq!(checkpoint1, checkpoint2);
    }
}
