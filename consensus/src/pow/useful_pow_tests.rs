//! Integration tests for UsefulPoW components
//! 
//! This module provides comprehensive tests for the UsefulPoW system,
//! testing the integration between training proofs, validation, and model management.

use crate::errors::{BlockProcessResult, RuleError};
use crate::pow::useful_pow::{UsefulPoW, ActiveModel};
use crate::model::{ModelCheckpoint, FedAvgAggregator, HuggingFaceProvenance, ModelMetrics};
use crate::pow::{TrainingProof, ZKProof, PublicInputs, ModelId, DifficultyTarget};
use kaspa_consensus_core::header::Header;
use kaspa_hashes::Hash;
use std::sync::Arc;

#[cfg(test)]
mod integration_tests {
    use super::*;

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

    fn create_test_checkpoint() -> ModelCheckpoint {
        ModelCheckpoint {
            block_height: 0,
            model_id: ModelId("test_model".to_string()),
            version: 1,
            weights_hash: Hash::from_bytes([10u8; 32]),
            architecture_hash: Hash::from_bytes([12u8; 32]),
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

    #[test]
    fn test_useful_pow_integration() {
        // Create UsefulPoW validator
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());

        // Register test model
        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: create_test_checkpoint(),
            reward_per_block: 1000,
        };
        pow.register_model(model);

        // Test legacy validation (before activation)
        let header = create_test_header();
        assert!(pow.validate_block(&header, 100).is_ok());

        // Test UsefulPoW validation (after activation)
        let mut header = create_test_header();
        header.daa_score = 2000; // After activation
        header.training_proof = Some(create_test_training_proof());
        header.model_checkpoint = Some(Hash::from_bytes([10u8; 32]));

        assert!(pow.validate_block(&header, 2000).is_ok());
    }

    #[test]
    fn test_fedavg_integration() {
        let initial_checkpoint = create_test_checkpoint();
        let mut aggregator = FedAvgAggregator::new(100, initial_checkpoint);

        // Simulate 10 blocks of training
        for i in 0..10 {
            let proof = create_test_training_proof();
            aggregator.on_block(i, &proof, Hash::from_bytes([i as u8; 32]));
        }

        assert_eq!(aggregator.pending_count(), 10);
        assert!(aggregator.should_aggregate(100));

        let new_checkpoint = aggregator.aggregate(100);
        assert_eq!(new_checkpoint.version, 2);
        assert_eq!(aggregator.pending_count(), 0);
    }

    #[test]
    fn test_reward_calculation() {
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());

        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: create_test_checkpoint(),
            reward_per_block: 1000,
        };
        pow.register_model(model);

        let mut header = create_test_header();
        header.training_proof = Some(create_test_training_proof());

        let base_reward = 500;
        let reward = pow.calculate_reward(&header, base_reward);

        assert!(reward > base_reward); // Should include quality bonus
    }

    #[test]
    fn test_multiple_models() {
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());

        // Register multiple models
        for i in 0..3 {
            let model = ActiveModel {
                model_id: ModelId(format!("model_{}", i)),
                current_checkpoint: ModelCheckpoint {
                    block_height: 0,
                    model_id: ModelId(format!("model_{}", i)),
                    version: 1,
                    weights_hash: Hash::from_bytes([i as u8; 32]),
                    architecture_hash: Hash::from_bytes([2u8; 32]),
                    hf_provenance: HuggingFaceProvenance {
                        repo_id: format!("repo_{}", i),
                        revision: "main".to_string(),
                        original_hash: Hash::from_bytes([3u8; 32]),
                    },
                    metrics: ModelMetrics::default(),
                },
                reward_per_block: 1000,
            };
            pow.register_model(model);
        }

        // Test validation for each model
        for i in 0..3 {
            let mut header = create_test_header();
            header.daa_score = 2000;
            header.training_proof = Some(TrainingProof {
                model_id: ModelId(format!("model_{}", i)),
                base_checkpoint: Hash::from_bytes([i as u8; 32]),
                loss_before: 0.5,
                loss_after: 0.4,
                gradients_commitment: Hash::from_bytes([11u8; 32]),
                zk_proof: ZKProof {
                    proof_data: vec![1, 2, 3],
                    public_inputs: PublicInputs {
                        model_hash: Hash::from_bytes([2u8; 32]),
                        input_hash: Hash::from_bytes([5u8; 32]),
                        output_gradients_hash: Hash::from_bytes([4u8; 32]),
                        loss_before: 0.5,
                        loss_after: 0.4,
                    },
                },
                batch_indices: vec![0, 1, 2],
            });
            header.model_checkpoint = Some(Hash::from_bytes([i as u8; 32]));

            assert!(pow.validate_block(&header, 2000).is_ok());
        }
    }

    #[test]
    fn test_difficulty_adjustment() {
        let pow = UsefulPoW::new(1000, DifficultyTarget {
            min_improvement: 0.01, // Higher difficulty
            max_loss_after: 0.8,
        });

        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: create_test_checkpoint(),
            reward_per_block: 1000,
        };
        pow.register_model(model);

        // Test with insufficient improvement
        let mut header = create_test_header();
        header.daa_score = 2000;
        header.training_proof = Some(TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([10u8; 32]),
            loss_before: 0.5,
            loss_after: 0.495, // Only 0.005 improvement (below 0.01 threshold)
            gradients_commitment: Hash::from_bytes([11u8; 32]),
            zk_proof: ZKProof {
                proof_data: vec![1, 2, 3],
                public_inputs: PublicInputs {
                    model_hash: Hash::from_bytes([12u8; 32]),
                    input_hash: Hash::from_bytes([13u8; 32]),
                    output_gradients_hash: Hash::from_bytes([14u8; 32]),
                    loss_before: 0.5,
                    loss_after: 0.495,
                },
            },
            batch_indices: vec![0, 1, 2],
        });
        header.model_checkpoint = Some(Hash::from_bytes([10u8; 32]));

        assert!(pow.validate_block(&header, 2000).is_err());
    }

    #[test]
    fn test_checkpoint_versioning() {
        let initial_checkpoint = create_test_checkpoint();
        let mut aggregator = FedAvgAggregator::new(10, initial_checkpoint);

        // Perform multiple aggregations
        for version in 1..5 {
            for i in 0..10 {
                let proof = create_test_training_proof();
                aggregator.on_block(i * version, &proof, Hash::from_bytes([i as u8; 32]));
            }

            let new_checkpoint = aggregator.aggregate(version * 10);
            assert_eq!(new_checkpoint.version, version + 1);
        }
    }

    #[test]
    fn test_error_cases() {
        let pow = UsefulPoW::new(1000, DifficultyTarget::default());

        let model = ActiveModel {
            model_id: ModelId("test_model".to_string()),
            current_checkpoint: create_test_checkpoint(),
            reward_per_block: 1000,
        };
        pow.register_model(model);

        // Test unknown model
        let mut header = create_test_header();
        header.daa_score = 2000;
        header.training_proof = Some(TrainingProof {
            model_id: ModelId("unknown_model".to_string()),
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
        });
        header.model_checkpoint = Some(Hash::from_bytes([10u8; 32]));

        assert!(matches!(pow.validate_block(&header, 2000), Err(RuleError::UnknownModel(_))));

        // Test invalid checkpoint
        let mut header = create_test_header();
        header.daa_score = 2000;
        header.training_proof = Some(TrainingProof {
            model_id: ModelId("test_model".to_string()),
            base_checkpoint: Hash::from_bytes([99u8; 32]), // Wrong checkpoint
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
        });
        header.model_checkpoint = Some(Hash::from_bytes([10u8; 32]));

        assert!(matches!(pow.validate_block(&header, 2000), Err(RuleError::InvalidCheckpoint)));
    }
}
