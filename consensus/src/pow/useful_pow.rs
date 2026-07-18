//! Useful Proof of Work - Mining = Model Training
//! 
//! This module implements the validation logic for UsefulPoW where miners
//! perform AI model training as proof of work.

use crate::errors::{BlockProcessResult, RuleError};
use kaspa_consensus_core::header::Header;
use kaspa_consensus_core::pow::{TrainingProof, DifficultyTarget, ModelId};
use crate::model::ModelCheckpoint;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Active model configuration for training
#[derive(Clone, Debug)]
pub struct ActiveModel {
    pub model_id: ModelId,
    pub current_checkpoint: ModelCheckpoint,
    pub reward_per_block: u64,
}

/// UsefulPoW validator
pub struct UsefulPoW {
    /// Active models for training
    pub active_models: Arc<RwLock<HashMap<ModelId, ActiveModel>>>,
    
    /// Current difficulty target
    pub difficulty: DifficultyTarget,
    
    /// DAA score when UsefulPoW activates
    pub activation_daa_score: u64,
}

impl UsefulPoW {
    pub fn new(activation_daa_score: u64, difficulty: DifficultyTarget) -> Self {
        Self {
            active_models: Arc::new(RwLock::new(HashMap::new())),
            difficulty,
            activation_daa_score,
        }
    }
    
    /// Register an active model for training
    pub fn register_model(&self, model: ActiveModel) {
        let mut models = self.active_models.write().unwrap();
        models.insert(model.model_id.clone(), model);
    }
    
    /// Get current checkpoint for a model
    pub fn get_model_checkpoint(&self, model_id: &ModelId) -> Option<ModelCheckpoint> {
        let models = self.active_models.read().unwrap();
        models.get(model_id).map(|m| m.current_checkpoint.clone())
    }
    
    /// Validate block proof based on DAA score
    pub fn validate_block(&self, header: &Header, daa_score: u64) -> BlockProcessResult<()> {
        // Before activation, use legacy validation (no training proof required)
        if daa_score < self.activation_daa_score {
            return self.validate_legacy(header);
        }
        
        // After activation, require training proof
        self.validate_training_proof(header)
    }
    
    /// Validate legacy blocks (before UsefulPoW activation)
    fn validate_legacy(&self, header: &Header) -> BlockProcessResult<()> {
        // Legacy blocks should not have training proof
        if header.training_proof.is_some() {
            return Err(RuleError::UnexpectedTrainingProof);
        }
        Ok(())
    }
    
    /// Validate training proof for UsefulPoW blocks
    fn validate_training_proof(&self, header: &Header) -> BlockProcessResult<()> {
        let proof = header.training_proof
            .as_ref()
            .ok_or(RuleError::MissingTrainingProof)?;
        
        // 1. Verify model exists
        let models = self.active_models.read().unwrap();
        let model = models.get(&proof.model_id)
            .ok_or(RuleError::UnknownModel(proof.model_id.clone()))?;
        
        // 2. Verify base checkpoint matches current model state
        if proof.base_checkpoint != model.current_checkpoint.weights_hash {
            return Err(RuleError::InvalidCheckpoint);
        }
        
        // 3. Verify difficulty (loss improvement)
        if !proof.verify(&self.difficulty) {
            return Err(RuleError::InsufficientTrainingImprovement);
        }
        
        // 4. Verify model checkpoint reference matches
        if let Some(checkpoint_ref) = &header.model_checkpoint {
            if checkpoint_ref != &model.current_checkpoint.weights_hash {
                return Err(RuleError::InvalidCheckpointReference);
            }
        }
        
        Ok(())
    }
    
    /// Calculate block reward based on training quality
    pub fn calculate_reward(&self, header: &Header, base_reward: u64) -> u64 {
        let proof = match &header.training_proof {
            Some(p) => p,
            None => return base_reward, // Legacy blocks get base reward
        };
        
        let models = self.active_models.read().unwrap();
        let model = match models.get(&proof.model_id) {
            Some(m) => m,
            None => return base_reward,
        };
        
        let quality_score = proof.quality_score();
        let model_bonus = (model.reward_per_block as f64 * quality_score) as u64;
        
        // Combine base reward with model-specific bonus
        base_reward.saturating_add(model_bonus)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaspa_consensus_core::pow::{TrainingProof, ZKProof, PublicInputs, ModelId, DifficultyTarget};
    use kaspa_hashes::Hash;
    
    fn create_test_header() -> Header {
        Header {
            hash: Hash::from_bytes([1u8; 32]),
            version: 1,
            parents_by_level: vec![vec![Hash::from_bytes([2u8; 32])]],
            hash_merkle_root: Hash::from_bytes([3u8; 32]),
            accepted_id_merkle_root: Hash::from_bytes([4u8; 32]),
            utxo_commitment: kaspa_muhash::Hash::from_bytes([5u8; 32]),
            timestamp: 1000,
            bits: 0x1d00ffff,
            nonce: 0,
            daa_score: 100,
            blue_work: kaspa_math::Uint192::from_u64(1000),
            blue_score: 100,
            epoch_seed: Hash::from_bytes([6u8; 32]),
            pruning_point: Hash::from_bytes([7u8; 32]),
            training_proof: None,
            model_checkpoint: None,
        }
    }
    
    fn create_test_training_proof() -> TrainingProof {
        TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([10u8; 32]),
            loss_before: 0.5,
            loss_after: 0.4,
            gradients_commitment: Hash::from_bytes([11u8; 32]),
            zk_proof: ZKProof {
                proof_data: vec![1, 2, 3],
                public_inputs: PublicInputs {
                    model_hash: Hash::from_bytes([12u8; 32]),
                    input_hash: Hash::from_bytes([13u8; 32]),
                    output_gradients_hash: Hash::from_bytes([14u8; 32]),
                    loss_before: 0.5,
                    loss_after: 0.4,
                },
            },
            batch_indices: vec![0, 1, 2],
        }
    }
    
    #[test]
    fn test_legacy_validation() {
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());
        let header = create_test_header();
        
        // Should pass - legacy block with no training proof
        assert!(pow.validate_block(&header, 100).is_ok());
    }
    
    #[test]
    fn test_legacy_with_training_proof_fails() {
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());
        let mut header = create_test_header();
        header.training_proof = Some(create_test_training_proof());
        
        // Should fail - legacy block shouldn't have training proof
        assert!(pow.validate_block(&header, 100).is_err());
    }
    
    #[test]
    fn test_usefulpow_validation() {
        let pow = UsefulPoW::new(100, DifficultyTarget::default());
        
        // Register test model
        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: ModelCheckpoint {
                block_height: 0,
                model_id: ModelId("test_model".to_string()),
                version: 1,
                weights_hash: Hash::from_bytes([10u8; 32]),
                architecture_hash: Hash::from_bytes([12u8; 32]),
            },
            reward_per_block: 1000,
        };
        pow.register_model(model);
        
        let mut header = create_test_header();
        header.daa_score = 200; // After activation
        header.training_proof = Some(create_test_training_proof());
        header.model_checkpoint = Some(Hash::from_bytes([10u8; 32]));
        
        // Should pass - valid training proof
        assert!(pow.validate_block(&header, 200).is_ok());
    }
    
    #[test]
    fn test_usefulpow_missing_proof() {
        let pow = UsefulPoW::new(100, DifficultyTarget::default());
        let header = create_test_header();
        header.daa_score = 200; // After activation
        
        // Should fail - missing training proof
        assert!(pow.validate_block(&header, 200).is_err());
    }
    
    #[test]
    fn test_reward_calculation() {
        let pow = UsefulPoW::new(100, DifficultyTarget::default());
        
        // Register test model
        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: ModelCheckpoint {
                block_height: 0,
                model_id: ModelId("test_model".to_string()),
                version: 1,
                weights_hash: Hash::from_bytes([10u8; 32]),
                architecture_hash: Hash::from_bytes([12u8; 32]),
            },
            reward_per_block: 1000,
        };
        pow.register_model(model);
        
        let mut header = create_test_header();
        header.training_proof = Some(create_test_training_proof());
        
        let reward = pow.calculate_reward(&header, 500);
        assert!(reward > 500); // Should include bonus
    }
}
