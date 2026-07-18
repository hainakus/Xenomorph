//! Checkpoint full validation for FedAvg aggregation points
//! 
//! This module implements full validation that occurs every 400 blocks
//! at FedAvg checkpoint aggregation points, where all nodes perform
//! complete ZK verification for maximum security.

use kaspa_hashes::Hash;
use serde::{Deserialize, Serialize};
use borsh::{BorshSerialize, BorshDeserialize};
use std::time::Instant;

use super::zk_verifier::{ZKVerifier, ZKTrainingProof, VerificationResult};
use crate::model::{ModelCheckpoint, FedAvgAggregator};

/// Result of checkpoint validation
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct CheckpointValidationResult {
    pub is_valid: bool,
    pub validation_time_ms: u64,
    pub total_proofs_verified: usize,
    pub valid_proofs: usize,
    pub invalid_proofs: usize,
    pub error_message: Option<String>,
}

/// Checkpoint validator for full validation at aggregation points
pub struct CheckpointValidator {
    verifier: Box<dyn ZKVerifier>,
    checkpoint_interval: u64,
}

impl CheckpointValidator {
    /// Create a new checkpoint validator
    pub fn new(verifier: Box<dyn ZKVerifier>, checkpoint_interval: u64) -> Self {
        Self {
            verifier,
            checkpoint_interval,
        }
    }

    /// Check if a block height is a checkpoint interval
    pub fn is_checkpoint_block(&self, block_height: u64) -> bool {
        block_height % self.checkpoint_interval == 0
    }

    /// Perform full validation of a checkpoint
    /// 
    /// This is called every N blocks at FedAvg aggregation points.
    /// All nodes perform complete ZK verification for maximum security.
    pub fn validate_checkpoint(
        &self,
        checkpoint: &ModelCheckpoint,
        training_proofs: &[ZKTrainingProof],
    ) -> CheckpointValidationResult {
        let start = Instant::now();
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut errors = Vec::new();

        for proof in training_proofs {
            let result = self.verifier.verify(proof);
            if result.is_valid {
                valid_count += 1;
            } else {
                invalid_count += 1;
                if let Some(error) = result.error_message {
                    errors.push(error);
                }
            }
        }

        let elapsed = start.elapsed();
        let is_valid = invalid_count == 0;

        CheckpointValidationResult {
            is_valid,
            validation_time_ms: elapsed.as_millis() as u64,
            total_proofs_verified: training_proofs.len(),
            valid_proofs: valid_count,
            invalid_proofs: invalid_count,
            error_message: if errors.is_empty() {
                None
            } else {
                Some(errors.join("; "))
            },
        }
    }

    /// Validate checkpoint with detailed per-proof results
    pub fn validate_checkpoint_detailed(
        &self,
        checkpoint: &ModelCheckpoint,
        training_proofs: &[ZKTrainingProof],
    ) -> (CheckpointValidationResult, Vec<VerificationResult>) {
        let start = Instant::now();
        let mut results = Vec::new();
        let mut valid_count = 0;
        let mut invalid_count = 0;
        let mut errors = Vec::new();

        for proof in training_proofs {
            let result = self.verifier.verify(proof);
            if result.is_valid {
                valid_count += 1;
            } else {
                invalid_count += 1;
                if let Some(error) = &result.error_message {
                    errors.push(error.clone());
                }
            }
            results.push(result);
        }

        let elapsed = start.elapsed();
        let is_valid = invalid_count == 0;

        let checkpoint_result = CheckpointValidationResult {
            is_valid,
            validation_time_ms: elapsed.as_millis() as u64,
            total_proofs_verified: training_proofs.len(),
            valid_proofs: valid_count,
            invalid_proofs: invalid_count,
            error_message: if errors.is_empty() {
                None
            } else {
                Some(errors.join("; "))
            },
        };

        (checkpoint_result, results)
    }

    /// Get the checkpoint interval
    pub fn checkpoint_interval(&self) -> u64 {
        self.checkpoint_interval
    }
}

/// Checkpoint validation manager for coordinating validation across the network
pub struct CheckpointValidationManager {
    validator: CheckpointValidator,
    pending_validations: std::collections::HashMap<u64, CheckpointValidation>,
}

struct CheckpointValidation {
    block_height: u64,
    checkpoint: ModelCheckpoint,
    training_proofs: Vec<ZKTrainingProof>,
    start_time: Instant,
}

impl CheckpointValidationManager {
    /// Create a new checkpoint validation manager
    pub fn new(validator: CheckpointValidator) -> Self {
        Self {
            validator,
            pending_validations: std::collections::HashMap::new(),
        }
    }

    /// Start a new checkpoint validation
    pub fn start_validation(
        &mut self,
        block_height: u64,
        checkpoint: ModelCheckpoint,
        training_proofs: Vec<ZKTrainingProof>,
    ) {
        let validation = CheckpointValidation {
            block_height,
            checkpoint,
            training_proofs,
            start_time: Instant::now(),
        };
        self.pending_validations.insert(block_height, validation);
    }

    /// Complete validation for a checkpoint
    pub fn complete_validation(&mut self, block_height: u64) -> Option<CheckpointValidationResult> {
        if let Some(validation) = self.pending_validations.remove(&block_height) {
            Some(self.validator.validate_checkpoint(&validation.checkpoint, &validation.training_proofs))
        } else {
            None
        }
    }

    /// Get pending validation count
    pub fn pending_count(&self) -> usize {
        self.pending_validations.len()
    }

    /// Clean up timed-out validations
    pub fn cleanup_timeouts(&mut self, timeout_ms: u64) {
        let timeout = Duration::from_millis(timeout_ms);
        self.pending_validations.retain(|_, validation| {
            validation.start_time.elapsed() < timeout
        });
    }
}

use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::zk_verifier::MockVerifier;
    use crate::pow::{TrainingProof as CoreTrainingProof, ZKProof, PublicInputs as CorePublicInputs, ModelId};
    use crate::model::{HuggingFaceProvenance, ModelMetrics};

    fn create_test_checkpoint() -> ModelCheckpoint {
        ModelCheckpoint {
            block_height: 1000,
            model_id: ModelId("test_model".to_string()),
            version: 1,
            weights_hash: Hash::from_bytes([1u8; 32]),
            architecture_hash: Hash::from_bytes([2u8; 32]),
            hf_provenance: HuggingFaceProvenance {
                repo_id: "test/repo".to_string(),
                revision: "main".to_string(),
                original_hash: Hash::from_bytes([3u8; 32]),
            },
            metrics: ModelMetrics {
                loss: 0.5,
                accuracy: Some(0.9),
                custom_metrics: vec![],
            },
        }
    }

    fn create_test_proof(is_valid: bool) -> ZKTrainingProof {
        ZKTrainingProof {
            core_proof: CoreTrainingProof {
                model_id: ModelId("test_model".to_string()),
                base_checkpoint: Hash::from_bytes([1u8; 32]),
                loss_before: 0.5,
                loss_after: if is_valid { 0.4 } else { 0.5 }, // No improvement if invalid
                gradients_commitment: Hash::from_bytes([2u8; 32]),
                zk_proof: ZKProof {
                    proof_data: vec![1, 2, 3, 4],
                    public_inputs: CorePublicInputs {
                        model_hash: Hash::from_bytes([3u8; 32]),
                        input_hash: Hash::from_bytes([4u8; 32]),
                        output_gradients_hash: Hash::from_bytes([5u8; 32]),
                        loss_before: 0.5,
                        loss_after: if is_valid { 0.4 } else { 0.5 },
                    },
                },
                batch_indices: vec![0, 1, 2],
            },
            vk_hash: Hash::from_bytes([6u8; 32]),
            verification_metadata: crate::validation::zk_verifier::VerificationMetadata {
                batch_size: 32,
                nonce: 12345,
                timestamp: 1000,
            },
        }
    }

    #[test]
    fn test_checkpoint_detection() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        assert!(validator.is_checkpoint_block(400));
        assert!(validator.is_checkpoint_block(800));
        assert!validator.is_checkpoint_block(401));
        assert!validator.is_checkpoint_block(399));
    }

    #[test]
    fn test_full_validation_success() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![
            create_test_proof(true),
            create_test_proof(true),
            create_test_proof(true),
        ];
        
        let result = validator.validate_checkpoint(&checkpoint, &proofs);
        
        assert!(result.is_valid);
        assert_eq!(result.total_proofs_verified, 3);
        assert_eq!(result.valid_proofs, 3);
        assert_eq!(result.invalid_proofs, 0);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_full_validation_mixed() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![
            create_test_proof(true),
            create_test_proof(false), // Invalid
            create_test_proof(true),
        ];
        
        let result = validator.validate_checkpoint(&checkpoint, &proofs);
        
        assert!(!result.is_valid);
        assert_eq!(result.total_proofs_verified, 3);
        assert_eq!(result.valid_proofs, 2);
        assert_eq!(result.invalid_proofs, 1);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_detailed_validation() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![
            create_test_proof(true),
            create_test_proof(false),
        ];
        
        let (checkpoint_result, proof_results) = validator.validate_checkpoint_detailed(&checkpoint, &proofs);
        
        assert_eq!(proof_results.len(), 2);
        assert!(proof_results[0].is_valid);
        assert!(!proof_results[1].is_valid);
        assert_eq!(checkpoint_result.valid_proofs, 1);
        assert_eq!(checkpoint_result.invalid_proofs, 1);
    }

    #[test]
    fn test_validation_manager() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        let mut manager = CheckpointValidationManager::new(validator);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![create_test_proof(true)];
        
        manager.start_validation(400, checkpoint, proofs);
        assert_eq!(manager.pending_count(), 1);
        
        let result = manager.complete_validation(400);
        assert!(result.is_some());
        assert!(result.unwrap().is_valid);
        
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_timeout_cleanup() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        let mut manager = CheckpointValidationManager::new(validator);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![create_test_proof(true)];
        
        manager.start_validation(400, checkpoint, proofs);
        assert_eq!(manager.pending_count(), 1);
        
        // Clean up with very short timeout
        manager.cleanup_timeouts(1);
        std::thread::sleep(Duration::from_millis(10));
        manager.cleanup_timeouts(1);
        
        assert_eq!(manager.pending_count(), 0);
    }

    #[test]
    fn test_validation_time() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = create_test_checkpoint();
        let proofs = vec![create_test_proof(true); 10];
        
        let result = validator.validate_checkpoint(&checkpoint, &proofs);
        
        // Should be fast (< 100ms for 10 mock verifications)
        assert!(result.validation_time_ms < 100);
    }
}
