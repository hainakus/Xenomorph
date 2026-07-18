//! On-chain validator reputation system
//! 
//! This module tracks validator reputation persistently on-chain,
//! including accuracy, speed, streaks, and reputation scores.

use serde::{Deserialize, Serialize};
use borsh::{BorshSerialize, BorshDeserialize};
use std::collections::HashMap;

/// Validator reputation data (persisted on-chain)
#[derive(Clone, Debug, Serialize, Deserialize, BorshSerialize, BorshDeserialize)]
pub struct ValidatorReputation {
    pub node_id: String,
    
    /// Total statistics
    pub total_validations: u64,
    pub correct_validations: u64,  // Agreed with consensus
    pub frauds_detected: u64,
    pub avg_validation_time_ms: u64,
    
    /// Current streak
    pub current_streak: u64,
    pub max_streak: u64,
    
    /// Composite score (0-10000)
    pub reputation_score: u32,
}

impl Default for ValidatorReputation {
    fn default() -> Self {
        Self {
            node_id: String::new(),
            total_validations: 0,
            correct_validations: 0,
            frauds_detected: 0,
            avg_validation_time_ms: 0,
            current_streak: 0,
            max_streak: 0,
            reputation_score: 5000, // Neutral score
        }
    }
}

impl ValidatorReputation {
    /// Create a new reputation entry
    pub fn new(node_id: String) -> Self {
        Self {
            node_id,
            ..Default::default()
        }
    }

    /// Update reputation after validation
    pub fn update(&mut self, result: &ValidationOutcome) {
        self.total_validations += 1;
        
        if result.approved == self.determine_consensus_for_validator(result) {
            self.correct_validations += 1;
            self.current_streak += 1;
            self.max_streak = self.max_streak.max(self.current_streak);
        } else {
            self.current_streak = 0; // Reset streak
        }
        
        if result.is_challenge && result.challenge_successful {
            self.frauds_detected += 1;
        }
        
        // Update average validation time
        self.avg_validation_time_ms = (self.avg_validation_time_ms * (self.total_validations - 1) + result.timestamp_ms) / self.total_validations;
        
        // Recalculate score
        self.reputation_score = self.compute_score();
    }

    /// Determine what the consensus was for this validator
    fn determine_consensus_for_validator(&self, _result: &ValidationOutcome) -> bool {
        // In production, this would query the blockchain for the actual consensus
        // For now, we'll assume the validator's vote was correct if they approved
        true // Placeholder
    }

    /// Compute reputation score
    fn compute_score(&self) -> u32 {
        let accuracy = if self.total_validations > 0 {
            (self.correct_validations * 10000) / self.total_validations
        } else {
            5000 // Neutral
        };
        
        let speed_bonus = (10000u64.saturating_sub(self.avg_validation_time_ms) * 1000) / 10000;
        let streak_bonus = (self.max_streak * 100).min(2000); // Max 20% bonus
        let fraud_bonus = (self.frauds_detected * 500).min(1000); // Max 10% bonus
        
        let total = (accuracy + speed_bonus + streak_bonus + fraud_bonus) / 4;
        total.min(10000) as u32
    }

    /// Get accuracy streak for a validator
    pub fn get_accuracy_streak(&self, node_id: &str) -> u64 {
        if self.node_id == node_id {
            self.current_streak
        } else {
            0
        }
    }

    /// Apply penalty to reputation
    pub fn apply_penalty(&mut self, _validator: &str, penalty_amount: u32) {
        self.reputation_score = self.reputation_score.saturating_sub(penalty_amount);
    }

    /// Get selection weight (stake × reputation)
    pub fn selection_weight(&self, base_stake: u64) -> u64 {
        // Stake × reputation_score
        // Higher reputation = higher chance of selection
        base_stake * self.reputation_score as u64 / 10000
    }

    /// Get accuracy percentage
    pub fn accuracy_percentage(&self) -> f64 {
        if self.total_validations == 0 {
            return 0.0;
        }
        (self.correct_validations as f64 / self.total_validations as f64) * 100.0
    }

    /// Get validator rating (1-5 stars)
    pub fn rating(&self) -> u8 {
        match self.reputation_score {
            0..=2000 => 1,
            2001..=4000 => 2,
            4001..=6000 => 3,
            6001..=8000 => 4,
            _ => 5,
        }
    }
}

/// Validation outcome for reputation tracking
#[derive(Clone, Debug)]
pub struct ValidationOutcome {
    pub validator: String,
    pub approved: bool,
    pub timestamp_ms: u64,
    pub is_challenge: bool,
    pub challenge_successful: bool,
}

/// Reputation manager for tracking multiple validators
pub struct ReputationManager {
    reputations: HashMap<String, ValidatorReputation>,
}

impl ReputationManager {
    /// Create a new reputation manager
    pub fn new() -> Self {
        Self {
            reputations: HashMap::new(),
        }
    }

    /// Get or create reputation for a validator
    pub fn get_or_create(&mut self, node_id: String) -> &mut ValidatorReputation {
        if !self.reputations.contains_key(&node_id) {
            self.reputations.insert(node_id.clone(), ValidatorReputation::new(node_id));
        }
        self.reputations.get_mut(&node_id).unwrap()
    }

    /// Update reputation for a validator
    pub fn update(&mut self, node_id: &str, result: &ValidationOutcome) {
        if let Some(reputation) = self.reputations.get_mut(node_id) {
            reputation.update(result);
        }
    }

    /// Get reputation for a validator
    pub fn get(&self, node_id: &str) -> Option<&ValidatorReputation> {
        self.reputations.get(node_id)
    }

    /// Get all reputations
    pub fn get_all(&self) -> Vec<&ValidatorReputation> {
        self.reputations.values().collect()
    }

    /// Get top validators by reputation score
    pub fn get_top_validators(&self, limit: usize) -> Vec<&ValidatorReputation> {
        let mut all: Vec<_> = self.reputations.values().collect();
        all.sort_by(|a, b| b.reputation_score.cmp(&a.reputation_score));
        all.into_iter().take(limit).collect()
    }

    /// Get validator statistics summary
    pub fn get_summary(&self) -> ReputationSummary {
        let all = self.get_all();
        
        if all.is_empty() {
            return ReputationSummary::default();
        }

        let total_validations: u64 = all.iter().map(|r| r.total_validations).sum();
        let total_correct: u64 = all.iter().map(|r| r.correct_validations).sum();
        let avg_score: u32 = all.iter().map(|r| r.reputation_score).sum::<u32>() / all.len() as u32;
        let avg_time: u64 = all.iter().map(|r| r.avg_validation_time_ms).sum::<u64>() / all.len() as u64;

        ReputationSummary {
            total_validators: all.len(),
            total_validations,
            avg_accuracy: (total_correct as f64 / total_validations as f64) * 100.0,
            avg_reputation_score: avg_score,
            avg_validation_time_ms: avg_time,
        }
    }
}

/// Reputation summary statistics
#[derive(Clone, Debug, Default)]
pub struct ReputationSummary {
    pub total_validators: usize,
    pub total_validations: u64,
    pub avg_accuracy: f64,
    pub avg_reputation_score: u32,
    pub avg_validation_time_ms: u64,
}

impl Default for ReputationManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validator_reputation_creation() {
        let reputation = ValidatorReputation::new("node1".to_string());
        
        assert_eq!(reputation.node_id, "node1");
        assert_eq!(reputation.total_validations, 0);
        assert_eq!(reputation.reputation_score, 5000);
    }

    #[test]
    fn test_reputation_update() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        let result = ValidationOutcome {
            validator: "node1".to_string(),
            approved: true,
            timestamp_ms: 50,
            is_challenge: false,
            challenge_successful: false,
        };
        
        reputation.update(&result);
        
        assert_eq!(reputation.total_validations, 1);
        assert_eq!(reputation.current_streak, 1);
        assert_eq!(reputation.max_streak, 1);
    }

    #[test]
    fn test_reputation_compute_score() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        // Simulate 10 correct validations
        for _ in 0..10 {
            let result = ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 50,
                is_challenge: false,
                challenge_successful: false,
            };
            reputation.update(&result);
        }
        
        assert!(reputation.reputation_score > 5000); // Should be above neutral
    }

    #[test]
    fn test_reputation_streak() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        for _ in 0..5 {
            let result = ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 50,
                is_challenge: false,
                challenge_successful: false,
            };
            reputation.update(&result);
        }
        
        assert_eq!(reputation.current_streak, 5);
        assert_eq!(reputation.max_streak, 5);
    }

    #[test]
    fn test_reputation_streak_reset() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        // Build streak
        for _ in 0..5 {
            let result = ValidationOutcome {
                validator: "node1".to_string(),
                approved: true,
                timestamp_ms: 50,
                is_challenge: false,
                challenge_successful: false,
            };
            reputation.update(&result);
        }
        
        assert_eq!(reputation.current_streak, 5);
        
        // Reset with incorrect validation
        let result = ValidationOutcome {
            validator: "node1".to_string(),
            approved: false,
            timestamp_ms: 50,
            is_challenge: false,
            challenge_successful: false,
        };
        reputation.update(&result);
        
        assert_eq!(reputation.current_streak, 0);
    }

    #[test]
    fn test_selection_weight() {
        let reputation = ValidatorReputation::new("node1".to_string());
        
        let weight = reputation.selection_weight(10000);
        
        // Neutral score (5000) should give 50% of stake weight
        assert_eq!(weight, 5000);
    }

    #[test]
    fn test_accuracy_percentage() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        // 5 correct out of 10 = 50%
        for i in 0..10 {
            let result = ValidationOutcome {
                validator: "node1".to_string(),
                approved: i < 5,
                timestamp_ms: 50,
                is_challenge: false,
                challenge_successful: false,
            };
            reputation.update(&result);
        }
        
        assert!((reputation.accuracy_percentage() - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_rating() {
        let mut reputation = ValidatorReputation::new("node1".to_string());
        
        assert_eq!(reputation.rating(), 3); // Neutral = 3 stars
        
        reputation.reputation_score = 8500;
        assert_eq!(reputation.rating(), 5); // High = 5 stars
        
        reputation.reputation_score = 1500;
        assert_eq!(reputation.rating(), 1); // Low = 1 star
    }

    #[test]
    fn test_reputation_manager() {
        let mut manager = ReputationManager::new();
        
        let reputation = manager.get_or_create("node1".to_string());
        assert_eq!(reputation.node_id, "node1");
    }

    #[test]
    fn test_reputation_manager_update() {
        let mut manager = ReputationManager::new();
        
        let result = ValidationOutcome {
            validator: "node1".to_string(),
            approved: true,
            timestamp_ms: 50,
            is_challenge: false,
            challenge_successful: false,
        };
        
        manager.update("node1", &result);
        
        let reputation = manager.get("node1").unwrap();
        assert_eq!(reputation.total_validations, 1);
    }

    #[test]
    fn test_get_top_validators() {
        let mut manager = ReputationManager::new();
        
        // Create validators with different scores
        for i in 0..5 {
            let mut reputation = manager.get_or_create(format!("node{}", i));
            reputation.reputation_score = 5000 + (i as u32 * 1000);
        }
        
        let top = manager.get_top_validators(3);
        assert_eq!(top.len(), 3);
        assert_eq!(top[0].reputation_score, 9000); // Highest score
    }

    #[test]
    fn test_reputation_summary() {
        let mut manager = ReputationManager::new();
        
        for i in 0..5 {
            let mut reputation = manager.get_or_create(format!("node{}", i));
            reputation.total_validations = 10;
            reputation.correct_validations = 8;
            reputation.reputation_score = 6000;
        }
        
        let summary = manager.get_summary();
        
        assert_eq!(summary.total_validators, 5);
        assert_eq!(summary.total_validations, 50);
        assert!((summary.avg_accuracy - 80.0).abs() < 0.1);
    }
}
