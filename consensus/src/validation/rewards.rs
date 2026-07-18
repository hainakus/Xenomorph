//! Validator reward system based on quality metrics
//! 
//! This module implements quality-based rewards for validators with
//! speed bonuses, accuracy bonuses, and fraud detection incentives.

use kaspa_hashes::Hash;
use serde::{Deserialize, Serialize};
use borsh::{BorshSerialize, BorshDeserialize};
use std::collections::HashMap;
use std::time::Instant;

use super::network::ValidationResult;

/// Reward distribution configuration
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct RewardDistribution {
    /// Base reward for participation
    pub base_participation: u64,
    
    /// Speed bonus for fastest validators
    pub speed_bonus: u64,
    
    /// Accuracy bonus for agreeing with consensus
    pub accuracy_bonus: u64,
    
    /// Fraud detection bonus for successful challenges
    pub fraud_detection_bonus: u64,
}

impl Default for RewardDistribution {
    fn default() -> Self {
        Self {
            base_participation: 500_000, // 0.5 Xenom (assuming 1 Xenom = 1M units)
            speed_bonus: 300_000,       // 0.3 Xenom
            accuracy_bonus: 200_000,     // 0.2 Xenom
            fraud_detection_bonus: 2_300_000, // 2.3 Xenom
        }
    }
}

/// Validation outcome for reward calculation
#[derive(Clone, Debug)]
pub struct ValidationOutcome {
    pub validator: String,
    pub approved: bool,
    pub timestamp_ms: u64,
    pub is_challenge: bool,
    pub challenge_successful: bool,
}

/// Validator rewards manager
pub struct ValidatorRewards {
    /// Total validation pool per block (10% of block reward)
    pub validation_pool: u64,
    
    /// Reward distribution configuration
    pub distribution: RewardDistribution,
    
    /// Validator reputation tracker
    pub reputation: super::reputation::ReputationManager,
}

impl ValidatorRewards {
    /// Create a new validator rewards manager
    pub fn new(validation_pool: u64, distribution: RewardDistribution) -> Self {
        Self {
            validation_pool,
            distribution,
            reputation: super::reputation::ReputationManager::new(),
        }
    }

    /// Calculate reward for a specific validator
    pub fn calculate_rewards(
        &self,
        block_hash: Hash,
        validations: &[ValidationOutcome],
        my_node_id: &str,
    ) -> u64 {
        let my_validation = validations.iter()
            .find(|v| v.validator == my_node_id)
            .expect("Validator did not participate in validation");
        
        // 1. Base reward for participation
        let mut total = self.distribution.base_participation;
        
        // 2. Speed bonus: fastest validators get more
        let rank = self.get_validation_rank(validations, my_node_id);
        if rank == 1 {
            total += self.distribution.speed_bonus; // +50% for first
        } else if rank <= 3 {
            total += self.distribution.speed_bonus / 2; // +25% for top 3
        }
        
        // 3. Accuracy bonus: agreed with consensus?
        let consensus_result = self.determine_consensus(validations);
        if my_validation.approved == consensus_result {
            total += self.distribution.accuracy_bonus;
            
            // Streak bonus: consecutive correct validations
            let streak = self.reputation.get(my_node_id)
                .map(|r| r.current_streak)
                .unwrap_or(0);
            let streak_multiplier = (streak.min(10) as u64 * self.distribution.accuracy_bonus) / 10;
            total += streak_multiplier; // Up to 2x
        } else {
            // Penalty for disagreeing with consensus (unless you're right!)
            total = total.saturating_sub(self.distribution.base_participation / 2);
        }
        
        // 4. Fraud detection: if challenged and was correct
        if my_validation.is_challenge && my_validation.challenge_successful {
            total += self.distribution.fraud_detection_bonus; // Jackpot!
        }
        
        total
    }

    /// Calculate validation rank by speed
    fn get_validation_rank(&self, validations: &[ValidationOutcome], node_id: &str) -> usize {
        let mut sorted: Vec<_> = validations.iter().collect();
        sorted.sort_by_key(|v| v.timestamp_ms);
        
        sorted.iter().position(|v| v.validator == node_id).unwrap_or(99) + 1
    }

    /// Determine consensus result (what majority said)
    fn determine_consensus(&self, validations: &[ValidationOutcome]) -> bool {
        let approvals = validations.iter().filter(|v| v.approved).count();
        let rejections = validations.len() - approvals;
        
        approvals > rejections // Majority approves?
    }

    /// Detect lazy validation (automatic approval without verification)
    pub fn detect_lazy_validation(&self, validation: &ValidationOutcome) -> bool {
        // Check if always validates in exactly X ms (bot behavior)
        let time_variance = self.calculate_time_variance(&validation.validator);
        
        // Check if never challenges suspicious blocks
        let challenge_rate = self.get_challenge_rate(&validation.validator);
        
        // Check if always approves (100% approval rate)
        let approval_rate = self.get_approval_rate(&validation.validator);
        
        if time_variance < 100 && challenge_rate < 0.01 && approval_rate > 0.99 {
            self.apply_lazy_penalty(&validation.validator);
            true
        } else {
            false
        }
    }

    /// Detect validation cartels (groups that always agree)
    pub fn detect_validation_cartel(&self, recent_blocks: &[BlockValidationData]) -> Option<CartelAlert> {
        // Detect if same group is always selected together
        // and always agrees (100% agreement)
        
        let co_occurrence = self.analyze_co_occurrence(recent_blocks);
        
        for (group, rate) in co_occurrence {
            if rate > 0.9 && self.always_agree(&group) {
                return Some(CartelAlert {
                    nodes: group,
                    evidence: "Co-occurrence + 100% agreement".to_string(),
                });
            }
        }
        
        None
    }

    /// Calculate time variance for a validator
    fn calculate_time_variance(&self, validator: &str) -> u64 {
        // Calculate variance in validation times
        // Low variance = suspicious (bot behavior)
        // This would query historical data
        1000 // Placeholder
    }

    /// Get challenge rate for a validator
    fn get_challenge_rate(&self, validator: &str) -> f64 {
        // Calculate how often this validator challenges blocks
        // Low challenge rate = suspicious (never challenges)
        0.05 // Placeholder
    }

    /// Get approval rate for a validator
    fn get_approval_rate(&self, validator: &str) -> f64 {
        // Calculate how often this validator approves blocks
        // 100% approval = suspicious (never rejects)
        0.95 // Placeholder
    }

    /// Apply penalty for lazy validation
    fn apply_lazy_penalty(&self, validator: &str) {
        // Apply reputation penalty
        if let Some(reputation) = self.reputation.get_mut(validator) {
            reputation.apply_penalty(validator, 1000);
        }
    }

    /// Analyze co-occurrence of validators
    fn analyze_co_occurrence(&self, blocks: &[BlockValidationData]) -> HashMap<Vec<String>, f64> {
        let mut co_occurrence: HashMap<Vec<String>, usize> = HashMap::new();
        
        for block in blocks {
            let mut validators: Vec<_> = block.validators.iter().map(|v| v.validator.clone()).collect();
            validators.sort();
            
            *co_occurrence.entry(validators).or_insert(0) += 1;
        }
        
        let total_blocks = blocks.len();
        co_occurrence.into_iter()
            .map(|(group, count)| (group, count as f64 / total_blocks as f64))
            .collect()
    }

    /// Check if a group always agrees
    fn always_agree(&self, group: &[String]) -> bool {
        // Check if validators in this group always have the same opinion
        // This would query historical data
        false // Placeholder
    }
}

/// Block validation data for cartel detection
#[derive(Clone, Debug)]
pub struct BlockValidationData {
    pub block_hash: Hash,
    pub validators: Vec<ValidationOutcome>,
}

/// Cartel detection alert
#[derive(Clone, Debug)]
pub struct CartelAlert {
    pub nodes: Vec<String>,
    pub evidence: String,
}

/// Reward calculation result
#[derive(Clone, Debug)]
pub struct RewardCalculation {
    pub validator: String,
    pub base_reward: u64,
    pub speed_bonus: u64,
    pub accuracy_bonus: u64,
    pub fraud_bonus: u64,
    pub penalty: u64,
    pub total_reward: u64,
}

impl RewardCalculation {
    /// Create a new reward calculation
    pub fn new(validator: String) -> Self {
        Self {
            validator,
            base_reward: 0,
            speed_bonus: 0,
            accuracy_bonus: 0,
            fraud_bonus: 0,
            penalty: 0,
            total_reward: 0,
        }
    }

    /// Calculate total reward
    pub fn calculate_total(&mut self) {
        self.total_reward = self.base_reward + self.speed_bonus + self.accuracy_bonus + self.fraud_bonus - self.penalty;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reward_distribution_default() {
        let dist = RewardDistribution::default();
        
        assert_eq!(dist.base_participation, 500_000);
        assert_eq!(dist.speed_bonus, 300_000);
        assert_eq!(dist.accuracy_bonus, 200_000);
        assert_eq!(dist.fraud_detection_bonus, 2_300_000);
    }

    #[test]
    fn test_validator_rewards_creation() {
        let rewards = ValidatorRewards::new(1_000_000, RewardDistribution::default());
        
        assert_eq!(rewards.validation_pool, 1_000_000);
    }

    #[test]
    fn test_calculate_rewards() {
        let rewards = ValidatorRewards::new(1_000_000, RewardDistribution::default());
        
        let validations = vec![
            ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 10,
                is_challenge: false,
                challenge_successful: false,
            },
            ValidationOutcome {
                validator: "node2".to_string(),
                approved: true,
                timestamp_ms: 20,
                is_challenge: false,
                challenge_successful: false,
            },
            ValidationOutcome {
                validator: "node3".to_string(),
                approved: false,
                timestamp_ms: 30,
                is_challenge: false,
                challenge_successful: false,
            },
        ];
        
        let reward = rewards.calculate_rewards(Hash::from_bytes([1u8; 32]), &validations, "node1");
        
        // Node1: base (500k) + speed bonus (300k) + accuracy bonus (200k) = 1M
        assert!(reward >= 1_000_000);
    }

    #[test]
    fn test_get_validation_rank() {
        let rewards = ValidatorRewards::new(1_000_000, RewardDistribution::default());
        
        let validations = vec![
            ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 10,
                is_challenge: false,
                challenge_successful: false,
            },
            ValidationOutcome {
                validator: "node2".to_string(),
                approved: true,
                timestamp_ms: 20,
                is_challenge: false,
                challenge_successful: false,
            },
        ];
        
        let rank = rewards.get_validation_rank(&validations, "node1");
        assert_eq!(rank, 1);
        
        let rank = rewards.get_validation_rank(&validations, "node2");
        assert_eq!(rank, 2);
    }

    #[test]
    fn test_determine_consensus() {
        let rewards = ValidatorRewards::new(1_000_000, RewardDistribution::default());
        
        let validations = vec![
            ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 10,
                is_challenge: false,
                challenge_successful: false,
            },
            ValidationOutcome {
                validator: "node2".to_string(),
                approved: true,
                timestamp_ms: 20,
                is_challenge: false,
                challenge_successful: false,
            },
            ValidationOutcome {
                validator: "node3".to_string(),
                approved: false,
                timestamp_ms: 30,
                is_challenge: false,
                challenge_successful: false,
            },
        ];
        
        let consensus = rewards.determine_consensus(&validations);
        assert!(consensus); // 2 approve, 1 rejects
    }

    #[test]
    fn test_reward_calculation() {
        let mut calc = RewardCalculation::new("node1".to_string());
        
        calc.base_reward = 500_000;
        calc.speed_bonus = 300_000;
        calc.accuracy_bonus = 200_000;
        calc.fraud_bonus = 0;
        calc.penalty = 0;
        
        calc.calculate_total();
        
        assert_eq!(calc.total_reward, 1_000_000);
    }

    #[test]
    fn test_reward_calculation_with_penalty() {
        let mut calc = RewardCalculation::new("node1".to_string());
        
        calc.base_reward = 500_000;
        calc.speed_bonus = 300_000;
        calc.accuracy_bonus = 0; // No accuracy bonus (disagreed with consensus)
        calc.fraud_bonus = 0;
        calc.penalty = 250_000; // Penalty for disagreement
        
        calc.calculate_total();
        
        assert_eq!(calc.total_reward, 550_000);
    }
}
