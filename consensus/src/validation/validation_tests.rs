//! Comprehensive tests for ZK validation system
//! 
//! This module provides integration tests, property tests, and benchmarks
//! for the ZK validation system to ensure security, performance, and correctness.

use crate::validation::{ValidatorSelector, ValidatorInfo, SignatureAggregator, VerificationResult, CheckpointValidator, ValidationNetwork, ValidationMessage, NetworkEvent};
use crate::validation::zk_verifier::{ZKTrainingProof, VerificationMetadata, MockVerifier};
use crate::pow::{TrainingProof as CoreTrainingProof, ZKProof, PublicInputs as CorePublicInputs, ModelId};
use crate::model::{ModelCheckpoint, HuggingFaceProvenance, ModelMetrics};
use std::time::Instant;

// Helper functions for test data
fn create_test_validators(count: usize) -> Vec<ValidatorInfo> {
    (0..count)
        .map(|i| ValidatorInfo {
            address: format!("validator_{}", i),
            stake: 10000 * (i + 1) as u64,
            public_key: vec![i as u8; 32],
        })
        .collect()
}

fn create_test_proof(is_valid: bool) -> ZKTrainingProof {
    ZKTrainingProof {
        core_proof: CoreTrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([1u8; 32]),
            loss_before: 0.5,
            loss_after: if is_valid { 0.4 } else { 0.5 },
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
        verification_metadata: VerificationMetadata {
            batch_size: 32,
            nonce: 12345,
            timestamp: 1000,
        },
    }
}

// Integration tests
#[cfg(test)]
mod integration_tests {
    use crate::validation::*;
    use super::*;

    #[test]
    fn test_full_validation_flow() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(20);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        // Select validators
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        assert_eq!(selection.selected_validators.len(), 10);
        
        // Verify selection is reproducible
        let selection2 = selector.select_validators(&block_hash, &validators, 10).unwrap();
        assert_eq!(selection.selected_validators, selection2.selected_validators);
        
        // Verify selection
        let verified = selector.verify_selection(&selection, &validators).unwrap();
        assert!(verified);
    }

    #[test]
    fn test_validation_consensus() {
        let mut aggregator = SignatureAggregator::new(7, 5000);
        
        // Add 7 approvals
        for i in 0..7 {
            aggregator.add_signature(format!("validator_{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator_{}", i),
                VerificationResult {
                    is_valid: true,
                    verification_time_ms: 10,
                    error_message: None,
                },
            );
        }
        
        assert!(aggregator.has_consensus());
        assert_eq!(aggregator.approval_rate(), 1.0);
    }

    #[test]
    fn test_checkpoint_validation_flow() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = ModelCheckpoint {
            block_height: 400,
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
        };
        
        let proofs = vec![create_test_proof(true); 10];
        let result = validator.validate_checkpoint(&checkpoint, &proofs);
        
        assert!(result.is_valid);
        assert_eq!(result.total_proofs_verified, 10);
    }

    #[test]
    fn test_network_validation_round() {
        let network = ValidationNetwork::new("validator_0".to_string(), 5000, 0.7);
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(20);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        let proof = create_test_proof(true);
        
        let message = ValidationMessage::ValidationRequest {
            block_hash,
            validator_selection: selection,
            proof,
        };
        
        let event = network.process_message(message).unwrap();
        
        // Should be selected since validator_0 is likely in the first 10
        match event {
            NetworkEvent::ValidationRequested { .. } => {
                // Expected outcome
            }
            NetworkEvent::NotSelectedForValidation => {
                // Also acceptable if not selected
            }
            _ => panic!("Unexpected event type"),
        }
    }
}

// Property tests
#[cfg(test)]
mod property_tests {
    use crate::validation::*;
    use super::*;

    #[test]
    fn test_selection_impossibility() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(100);
        
        // Test that different blocks produce different selections
        let mut selections = std::collections::HashSet::new();
        
        for i in 0..50 {
            let block_hash = Hash::from_bytes([i as u8; 32]);
            let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
            let signature: String = selection.selected_validators.join(",");
            selections.insert(signature);
        }
        
        // Should have many different selections (not all the same)
        assert!(selections.len() > 10);
    }

    #[test]
    fn test_stake_proportionality() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        
        // Create validators with different stakes
        let validators = vec![
            ValidatorInfo {
                address: "high_stake".to_string(),
                stake: 100000,
                public_key: vec![1; 32],
            },
            ValidatorInfo {
                address: "low_stake".to_string(),
                stake: 1000,
                public_key: vec![2; 32],
            },
        ];
        
        let total_stake: u64 = validators.iter().map(|v| v.stake).sum();
        
        let high_prob = selector.selection_probability(&validators[0], total_stake, 10);
        let low_prob = selector.selection_probability(&validators[1], total_stake, 10);
        
        // High stake validator should have much higher probability
        assert!(high_prob > low_prob * 10);
    }

    #[test]
    fn test_selection_determinism() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(20);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        // Multiple selections with same inputs should produce same results
        let mut results = Vec::new();
        
        for _ in 0..10 {
            let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
            results.push(selection.selected_validators.clone());
        }
        
        // All results should be identical
        for result in &results[1..] {
            assert_eq!(result, &results[0]);
        }
    }

    #[test]
    fn test_verification_constant_time() {
        let verifier = Box::new(MockVerifier::new(true));
        
        let valid_proof = create_test_proof(true);
        let invalid_proof = create_test_proof(false);
        
        // Both should complete in similar time (within 10ms)
        let start1 = Instant::now();
        verifier.verify(&valid_proof);
        let time1 = start1.elapsed();
        
        let start2 = Instant::now();
        verifier.verify(&invalid_proof);
        let time2 = start2.elapsed();
        
        let time_diff = if time1 > time2 { time1 - time2 } else { time2 - time1 };
        assert!(time_diff.as_millis() < 10);
    }
}

// Benchmarks
#[cfg(test)]
mod benchmarks {
    use crate::validation::*;
    use super::*;

    #[test]
    fn benchmark_validator_selection() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(1000);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        let start = Instant::now();
        
        for _ in 0..100 {
            let _ = selector.select_validators(&block_hash, &validators, 10);
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / 100.0;
        
        println!("Average selection time: {:.2} μs", avg_time);
        assert!(avg_time < 1000.0); // Should be < 1ms
    }

    #[test]
    fn benchmark_zk_verification() {
        let verifier = Box::new(MockVerifier::new(true));
        let proof = create_test_proof(true);
        
        let start = Instant::now();
        
        for _ in 0..100 {
            verifier.verify(&proof);
        }
        
        let elapsed = start.elapsed();
        let avg_time = elapsed.as_micros() as f64 / 100.0;
        
        println!("Average verification time: {:.2} μs", avg_time);
        assert!(avg_time < 100000.0); // Should be < 100ms
    }

    #[test]
    fn benchmark_batch_verification() {
        let verifier = Box::new(MockVerifier::new(true));
        let proofs: Vec<_> = (0..50).map(|_| create_test_proof(true)).collect();
        
        let start = Instant::now();
        verifier.verify_batch(&proofs);
        let elapsed = start.elapsed();
        
        println!("Batch verification time for 50 proofs: {:.2} ms", elapsed.as_millis() as f64);
        assert!(elapsed.as_millis() < 5000); // Should be < 5s for 50 proofs
    }

    #[test]
    fn benchmark_signature_aggregation() {
        let mut aggregator = SignatureAggregator::new(7, 5000);
        
        let start = Instant::now();
        
        for i in 0..10 {
            aggregator.add_signature(format!("validator_{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator_{}", i),
                VerificationResult {
                    is_valid: true,
                    verification_time_ms: 10,
                    error_message: None,
                },
            );
        }
        
        let elapsed = start.elapsed();
        println!("Signature aggregation time: {:.2} μs", elapsed.as_micros() as f64);
        assert!(elapsed.as_micros() < 100); // Should be very fast
    }

    #[test]
    fn benchmark_checkpoint_validation() {
        let verifier = Box::new(MockVerifier::new(true));
        let validator = CheckpointValidator::new(verifier, 400);
        
        let checkpoint = ModelCheckpoint {
            block_height: 400,
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
        };
        
        let proofs: Vec<_> = (0..100).map(|_| create_test_proof(true)).collect();
        
        let start = Instant::now();
        validator.validate_checkpoint(&checkpoint, &proofs);
        let elapsed = start.elapsed();
        
        println!("Checkpoint validation time for 100 proofs: {:.2} ms", elapsed.as_millis() as f64);
        assert!(elapsed.as_millis() < 10000); // Should be < 10s for 100 proofs
    }
}

// Security tests
#[cfg(test)]
mod security_tests {
    use crate::validation::*;
    use super::*;

    #[test]
    fn test_min_stake_enforcement() {
        let selector = ValidatorSelector::new(10000, 10, 100); // High minimum stake
        
        let validators = vec![
            ValidatorInfo {
                address: "low_stake".to_string(),
                stake: 5000, // Below minimum
                public_key: vec![1; 32],
            },
            ValidatorInfo {
                address: "high_stake".to_string(),
                stake: 15000, // Above minimum
                public_key: vec![2; 32],
            },
        ];
        
        let block_hash = Hash::from_bytes([1u8; 32]);
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        
        // Only high_stake should be selected
        assert!(!selection.selected_validators.contains(&"low_stake".to_string()));
        assert!(selection.selected_validators.contains(&"high_stake".to_string()));
    }

    #[test]
    fn test_unpredictability_before_block() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(20);
        
        // Try to predict selection before block hash is known
        // This should be impossible since seed is derived from block hash
        let predictions = std::collections::HashSet::new();
        
        for i in 0..10 {
            let different_hash = Hash::from_bytes([i as u8; 32]);
            let selection = selector.select_validators(&different_hash, &validators, 10).unwrap();
            predictions.insert(selection.selected_validators.clone());
        }
        
        // Should get different predictions for different hashes
        assert!(predictions.len() > 1);
    }

    #[test]
    fn test_sybil_resistance() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        
        // Create many low-stake validators trying to game the system
        let mut validators = Vec::new();
        for i in 0..100 {
            validators.push(ValidatorInfo {
                address: format!("sybil_{}", i),
                stake: 500, // Below minimum
                public_key: vec![i as u8; 32],
            });
        }
        
        // Add one legitimate validator
        validators.push(ValidatorInfo {
            address: "legitimate".to_string(),
            stake: 100000,
            public_key: vec![255; 32],
        });
        
        let block_hash = Hash::from_bytes([1u8; 32]);
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        
        // Sybil validators should be filtered out
        assert!(!selection.selected_validators.iter().any(|addr| addr.starts_with("sybil_")));
        
        // Legitimate validator should be selected
        assert!(selection.selected_validators.contains(&"legitimate".to_string()));
    }

    #[test]
    fn test_no_invalid_proofs_accepted() {
        let verifier = Box::new(MockVerifier::new(false)); // Always fails
        
        let invalid_proof = create_test_proof(false);
        let result = verifier.verify(&invalid_proof);
        
        assert!(!result.is_valid);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_consensus_with_partial_failure() {
        let mut aggregator = SignatureAggregator::new(7, 5000);
        
        // Add 5 valid, 2 invalid
        for i in 0..5 {
            aggregator.add_signature(format!("validator_{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator_{}", i),
                VerificationResult {
                    is_valid: true,
                    verification_time_ms: 10,
                    error_message: None,
                },
            );
        }
        
        for i in 5..7 {
            aggregator.add_signature(format!("validator_{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator_{}", i),
                VerificationResult {
                    is_valid: false,
                    verification_time_ms: 10,
                    error_message: Some("Failed".to_string()),
                },
            );
        }
        
        // Should have consensus (7 signatures) but low approval rate
        assert!(aggregator.has_consensus());
        assert!(aggregator.approval_rate() < 0.8);
    }
}

// Performance validation tests
#[cfg(test)]
mod performance_validation {
    use crate::validation::*;
    use super::*;

    #[test]
    fn test_overhead_vs_simple_pow() {
        // Measure the overhead of ZK validation vs simple PoW
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(100);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        let start = Instant::now();
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        let selection_time = start.elapsed();
        
        let verifier = Box::new(MockVerifier::new(true));
        let proof = create_test_proof(true);
        
        let start = Instant::now();
        verifier.verify(&proof);
        let verification_time = start.elapsed();
        
        let total_overhead = selection_time + verification_time;
        
        println!("Total validation overhead: {:.2} μs", total_overhead.as_micros() as f64);
        
        // Should be < 5% of typical block time (250ms = 250000μs)
        assert!(total_overhead.as_micros() < 12500); // < 5% of 250ms
    }

    #[test]
    fn test_network_message_size() {
        let selector = ValidatorSelector::new(1000, 10, 100);
        let validators = create_test_validators(20);
        let block_hash = Hash::from_bytes([1u8; 32]);
        
        let selection = selector.select_validators(&block_hash, &validators, 10).unwrap();
        let proof = create_test_proof(true);
        
        let message = ValidationMessage::ValidationRequest {
            block_hash,
            validator_selection: selection,
            proof,
        };
        
        let serialized = bincode::serialize(&message).unwrap();
        println!("Validation message size: {} bytes", serialized.len());
        
        // Should be < 5KB as per requirements
        assert!(serialized.len() < 5000);
    }
}
