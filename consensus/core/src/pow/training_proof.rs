//! Useful Proof of Work - Training Proof data structures
//! 
//! This module implements the training proof system where miners perform
//! AI model training as proof of work instead of traditional hashing.

use kaspa_hashes::Hash;
use serde::{Deserialize, Serialize};
use borsh::{BorshDeserialize, BorshSerialize};

/// Unique identifier for a model being trained
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize, PartialEq, Eq, Hash)]
pub struct ModelId(pub String);

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Difficulty target for training proof
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct DifficultyTarget {
    /// Minimum required loss improvement
    pub min_improvement: f64,
    /// Maximum allowed loss after training
    pub max_loss_after: f64,
}

impl Default for DifficultyTarget {
    fn default() -> Self {
        Self {
            min_improvement: 0.001,  // At least 0.1% improvement
            max_loss_after: 1.0,     // Loss must be below 1.0
        }
    }
}

/// Zero-knowledge proof of valid training computation
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct ZKProof {
    /// Raw proof data (e.g., SNARK proof)
    pub proof_data: Vec<u8>,
    /// Public inputs for verification
    pub public_inputs: PublicInputs,
}

/// Public inputs for ZK proof verification
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct PublicInputs {
    /// Hash of the model architecture
    pub model_hash: Hash,
    /// Hash of the training data batch
    pub input_hash: Hash,
    /// Hash of the computed gradients
    pub output_gradients_hash: Hash,
    /// Loss value before training
    pub loss_before: f64,
    /// Loss value after training
    pub loss_after: f64,
}

impl ZKProof {
    /// Verify the ZK proof (placeholder for actual ZK verification)
    /// In production, this would use EZKL or similar ZK verification system
    pub fn verify(&self) -> bool {
        // TODO: Implement actual ZK verification using EZKL or similar
        // For now, this is a placeholder that checks basic structure
        !self.proof_data.is_empty() && self.public_inputs.loss_after < self.public_inputs.loss_before
    }
}

/// Proof that miner performed valid model training
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct TrainingProof {
    /// Model being trained
    pub model_id: ModelId,
    
    /// Base checkpoint hash (before training)
    pub base_checkpoint: Hash,
    
    /// Training result metrics
    pub loss_before: f64,
    pub loss_after: f64,
    
    /// Commitment to the gradients (hash of gradient tensor)
    pub gradients_commitment: Hash,
    
    /// ZK proof of valid computation
    pub zk_proof: ZKProof,
    
    /// Data indices used for this training batch
    pub batch_indices: Vec<u64>,
}

impl TrainingProof {
    /// Verify proof meets difficulty target
    pub fn verify(&self, target: &DifficultyTarget) -> bool {
        let improvement = self.loss_before - self.loss_after;
        
        // Check loss improvement meets minimum
        if improvement <= target.min_improvement {
            return false;
        }
        
        // Check final loss is below maximum
        if self.loss_after >= target.max_loss_after {
            return false;
        }
        
        // Verify ZK proof
        if !self.zk_proof.verify() {
            return false;
        }
        
        true
    }
    
    /// Calculate the quality score for reward calculation
    pub fn quality_score(&self) -> f64 {
        let improvement = self.loss_before - self.loss_after;
        // Higher improvement = higher quality score
        1.0 + (improvement * 10.0).min(0.5)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_training_proof_verification() {
        let proof = TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([1u8; 32]),
            loss_before: 0.5,
            loss_after: 0.4,  // 0.1 improvement
            gradients_commitment: Hash::from_bytes([2u8; 32]),
            zk_proof: ZKProof {
                proof_data: vec![1, 2, 3],
                public_inputs: PublicInputs {
                    model_hash: Hash::from_bytes([3u8; 32]),
                    input_hash: Hash::from_bytes([4u8; 32]),
                    output_gradients_hash: Hash::from_bytes([5u8; 32]),
                    loss_before: 0.5,
                    loss_after: 0.4,
                },
            },
            batch_indices: vec![0, 1, 2],
        };
        
        let target = DifficultyTarget::default();
        assert!(proof.verify(&target));
    }
    
    #[test]
    fn test_training_proof_insufficient_improvement() {
        let proof = TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([1u8; 32]),
            loss_before: 0.5,
            loss_after: 0.499,  // Only 0.001 improvement (not enough)
            gradients_commitment: Hash::from_bytes([2u8; 32]),
            zk_proof: ZKProof {
                proof_data: vec![1, 2, 3],
                public_inputs: PublicInputs {
                    model_hash: Hash::from_bytes([3u8; 32]),
                    input_hash: Hash::from_bytes([4u8; 32]),
                    output_gradients_hash: Hash::from_bytes([5u8; 32]),
                    loss_before: 0.5,
                    loss_after: 0.499,
                },
            },
            batch_indices: vec![0, 1, 2],
        };
        
        let target = DifficultyTarget::default();
        assert!(!proof.verify(&target));
    }
    
    #[test]
    fn test_quality_score() {
        let proof = TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([1u8; 32]),
            loss_before: 0.5,
            loss_after: 0.3,  // 0.2 improvement
            gradients_commitment: Hash::from_bytes([2u8; 32]),
            zk_proof: ZKProof {
                proof_data: vec![1, 2, 3],
                public_inputs: PublicInputs {
                    model_hash: Hash::from_bytes([3u8; 32]),
                    input_hash: Hash::from_bytes([4u8; 32]),
                    output_gradients_hash: Hash::from_bytes([5u8; 32]),
                    loss_before: 0.5,
                    loss_after: 0.3,
                },
            },
            batch_indices: vec![0, 1, 2],
        };
        
        let score = proof.quality_score();
        assert!(score > 1.0);
        assert!(score <= 1.5);  // Max bonus is 0.5
    }
}
