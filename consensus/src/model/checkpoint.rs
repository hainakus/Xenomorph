//! Model checkpoint management
//! 
//! This module defines the on-chain model checkpoint structure and
//! related functionality for tracking model state across the blockchain.

use kaspa_hashes::Hash;
use serde::{Deserialize, Serialize};
use borsh::{BorshDeserialize, BorshSerialize};

use crate::pow::ModelId;

/// HuggingFace provenance information for model tracking
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct HuggingFaceProvenance {
    pub repo_id: String,
    pub revision: String,
    pub original_hash: Hash,
}

/// On-chain model checkpoint (metadata only)
/// 
/// Checkpoints store the minimal metadata needed to verify training proofs
/// and track model evolution without storing full weights on-chain.
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct ModelCheckpoint {
    pub block_height: u64,
    pub model_id: ModelId,
    pub version: u32,
    pub weights_hash: Hash,
    pub architecture_hash: Hash,
    pub hf_provenance: HuggingFaceProvenance,
    pub metrics: ModelMetrics,
}

/// Model performance metrics at checkpoint time
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct ModelMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub custom_metrics: Vec<(String, f64)>,
}

impl ModelCheckpoint {
    /// Create a new model checkpoint
    pub fn new(
        block_height: u64,
        model_id: ModelId,
        version: u32,
        weights_hash: Hash,
        architecture_hash: Hash,
        hf_provenance: HuggingFaceProvenance,
        metrics: ModelMetrics,
    ) -> Self {
        Self {
            block_height,
            model_id,
            version,
            weights_hash,
            architecture_hash,
            hf_provenance,
            metrics,
        }
    }
    
    /// Compute the checkpoint hash for on-chain references
    pub fn hash(&self) -> Hash {
        // Implementation for checkpoint hashing:
        // 1. Hash all checkpoint fields (model_id, version, weights_hash, metrics, provenance)
        // 2. Use blake3 for cryptographic hash
        // 3. Return hash commitment
        
        let mut hasher = blake3::Hasher::new();
        
        // Hash model ID
        hasher.update(self.model_id.0.as_bytes());
        
        // Hash version
        hasher.update(&self.version.to_le_bytes());
        
        // Hash weights hash
        hasher.update(self.weights_hash.as_bytes());
        
        // Hash metrics
        hasher.update(&self.metrics.loss.to_le_bytes());
        if let Some(accuracy) = self.metrics.accuracy {
            hasher.update(&accuracy.to_le_bytes());
        }
        
        // Hash provenance
        hasher.update(self.hf_provenance.model_id.as_bytes());
        hasher.update(self.hf_provenance.dataset_id.as_bytes());
        hasher.update(self.hf_provenance.commit_hash.as_bytes());
        
        Hash::from_bytes(*hasher.finalize().as_bytes())
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self {
            loss: 0.0,
            accuracy: None,
            custom_metrics: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = ModelCheckpoint::new(
            1000,
            ModelId("test_model".to_string()),
            1,
            Hash::from_bytes([1u8; 32]),
            Hash::from_bytes([2u8; 32]),
            HuggingFaceProvenance {
                repo_id: "test/repo".to_string(),
                revision: "main".to_string(),
                original_hash: Hash::from_bytes([3u8; 32]),
            },
            ModelMetrics {
                loss: 0.5,
                accuracy: Some(0.9),
                custom_metrics: vec![("f1_score".to_string(), 0.85)],
            },
        );
        
        assert_eq!(checkpoint.block_height, 1000);
        assert_eq!(checkpoint.version, 1);
        assert_eq!(checkpoint.metrics.loss, 0.5);
    }
}
