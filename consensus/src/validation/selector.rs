//! Stake-weighted validator selection for ZK validation
//!
//! This module implements cryptographically secure, stake-weighted validator selection
//! using ChaCha20 RNG seeded from block hashes to ensure deterministic, verifiable selection.

use kaspa_hashes::Hash;
use rand_chacha::ChaCha20Rng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

// ============================================================================
// CONSTANTS
// ============================================================================
const DEFAULT_SAMPLE_SIZE: usize = 10;
const DEFAULT_MIN_STAKE: u64 = 1_000_000_000_000; // 1k Xenom in sompi
const MAX_SAMPLE_SIZE: usize = 100;
const MIN_SAMPLE_SIZE: usize = 1;

// ============================================================================
// ERRORS
// ============================================================================
#[derive(Error, Debug)]
pub enum SelectionError {
    #[error("insufficient eligible validators: {eligible} < {required}")]
    InsufficientValidators { eligible: usize, required: usize },
    
    #[error("total stake is zero, cannot perform weighted selection")]
    ZeroTotalStake,
    
    #[error("sample size {size} exceeds maximum {max}")]
    SampleSizeTooLarge { size: usize, max: usize },
    
    #[error("sample size {size} is below minimum {min}")]
    SampleSizeTooSmall { size: usize, min: usize },
    
    #[error("no validators available for selection")]
    NoValidatorsAvailable,
}

// ============================================================================
// TYPES
// ============================================================================
pub type NodeId = String;
pub type PublicKey = Vec<u8>;

// ============================================================================
// STRUCTS
// ============================================================================
/// Information about a validator
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorInfo {
    pub address: NodeId,
    pub stake: u64,
    pub public_key: PublicKey,
}

/// Result of validator selection
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorSelection {
    pub selected_validators: Vec<ValidatorInfo>,
    pub block_hash: Hash,
    pub seed: [u8; 32],
    pub total_stake: u64,
}

/// Simplified validator selection for network messages
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValidatorSelectionSimple {
    pub selected_validators: Vec<String>,
    pub block_hash: Hash,
    pub seed: [u8; 32],
    pub total_stake: u64,
}

impl From<ValidatorSelection> for ValidatorSelectionSimple {
    fn from(selection: ValidatorSelection) -> Self {
        Self {
            selected_validators: selection.selected_validators.iter().map(|v| v.address.clone()).collect(),
            block_hash: selection.block_hash,
            seed: selection.seed,
            total_stake: selection.total_stake,
        }
    }
}

/// Validator selector with stake-weighted sampling
pub struct ValidatorSelector {
    validators: Vec<ValidatorInfo>,
    min_stake: u64,
    min_validators: usize,
    max_validators: usize,
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================
impl ValidatorSelector {
    /// Create a new validator selector
    pub fn new(validators: Vec<ValidatorInfo>) -> Self {
        Self {
            validators,
            min_stake: DEFAULT_MIN_STAKE,
            min_validators: MIN_SAMPLE_SIZE,
            max_validators: MAX_SAMPLE_SIZE,
        }
    }

    /// Create a new validator selector with custom parameters
    pub fn with_params(
        validators: Vec<ValidatorInfo>,
        min_stake: u64,
        min_validators: usize,
        max_validators: usize,
    ) -> Result<Self, SelectionError> {
        if min_validators > max_validators {
            return Err(SelectionError::SampleSizeTooSmall {
                size: min_validators,
                min: max_validators,
            });
        }
        
        if max_validators > MAX_SAMPLE_SIZE {
            return Err(SelectionError::SampleSizeTooLarge {
                size: max_validators,
                max: MAX_SAMPLE_SIZE,
            });
        }

        Ok(Self {
            validators,
            min_stake,
            min_validators,
            max_validators,
        })
    }

    /// Select validators for a block
    pub fn select_validators(
        &self,
        block_hash: Hash,
        sample_size: usize,
    ) -> Result<ValidatorSelection, SelectionError> {
        // Validate sample size
        if sample_size < self.min_validators {
            return Err(SelectionError::SampleSizeTooSmall {
                size: sample_size,
                min: self.min_validators,
            });
        }
        
        if sample_size > self.max_validators {
            return Err(SelectionError::SampleSizeTooLarge {
                size: sample_size,
                max: self.max_validators,
            });
        }

        // Filter validators by minimum stake
        let eligible: Vec<&ValidatorInfo> = self.validators
            .iter()
            .filter(|v| v.stake >= self.min_stake)
            .collect();

        if eligible.is_empty() {
            return Err(SelectionError::NoValidatorsAvailable);
        }

        if eligible.len() < sample_size {
            return Err(SelectionError::InsufficientValidators {
                eligible: eligible.len(),
                required: sample_size,
            });
        }

        // Calculate total stake
        let total_stake: u64 = eligible.iter().map(|v| v.stake).sum();
        
        if total_stake == 0 {
            return Err(SelectionError::ZeroTotalStake);
        }

        // Derive seed from block hash
        let seed = self.derive_seed(&block_hash);

        // Create RNG
        let mut rng = ChaCha20Rng::from_seed(seed);

        // Perform weighted sampling without replacement
        let selected = self.weighted_sample_without_replacement(&eligible, sample_size, &mut rng)?;

        Ok(ValidatorSelection {
            selected_validators: selected.into_iter().cloned().collect(),
            block_hash,
            seed,
            total_stake,
        })
    }

    /// Weighted sampling without replacement using reservoir sampling
    fn weighted_sample_without_replacement(
        &self,
        candidates: &[&ValidatorInfo],
        k: usize,
        rng: &mut ChaCha20Rng,
    ) -> Result<Vec<&ValidatorInfo>, SelectionError> {
        if k == 0 {
            return Ok(Vec::new());
        }

        if k > candidates.len() {
            return Err(SelectionError::InsufficientValidators {
                eligible: candidates.len(),
                required: k,
            });
        }

        let total_stake: u64 = candidates.iter().map(|c| c.stake).sum();

        if total_stake == 0 {
            return Err(SelectionError::ZeroTotalStake);
        }

        // Use weighted reservoir sampling
        let mut selected: Vec<&ValidatorInfo> = Vec::with_capacity(k);
        let mut remaining: Vec<&ValidatorInfo> = candidates.to_vec();
        let mut remaining_stake = total_stake;

        for _ in 0..k {
            if remaining.is_empty() || remaining_stake == 0 {
                break;
            }

            // Select random target based on stake
            let target: u64 = rng.gen_range(0..remaining_stake);
            let mut cumulative = 0u64;
            let mut chosen_idx = 0;

            for (idx, candidate) in remaining.iter().enumerate() {
                cumulative = cumulative.saturating_add(candidate.stake);
                if cumulative > target {
                    chosen_idx = idx;
                    selected.push(candidate);
                    remaining_stake = remaining_stake.saturating_sub(candidate.stake);
                    break;
                }
            }

            // Remove selected candidate
            if chosen_idx < remaining.len() {
                remaining.remove(chosen_idx);
            }
        }

        Ok(selected)
    }

    /// Derive RNG seed from block hash
    fn derive_seed(&self, block_hash: &Hash) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(block_hash.as_bytes());
        hasher.update(b"xenom-validator-selection-v1");
        
        let hash = hasher.finalize();
        let mut seed = [0u8; 32];
        seed.copy_from_slice(hash.as_bytes());
        seed
    }

    /// Get the number of registered validators
    pub fn validator_count(&self) -> usize {
        self.validators.len()
    }

    /// Get the number of eligible validators (above min stake)
    pub fn eligible_count(&self) -> usize {
        self.validators
            .iter()
            .filter(|v| v.stake >= self.min_stake)
            .count()
    }

    /// Add a validator
    pub fn add_validator(&mut self, validator: ValidatorInfo) {
        self.validators.push(validator);
    }

    /// Remove a validator by address
    pub fn remove_validator(&mut self, address: &str) -> bool {
        let original_len = self.validators.len();
        self.validators.retain(|v| v.address != address);
        self.validators.len() < original_len
    }

    /// Update validator stake
    pub fn update_stake(&mut self, address: &str, new_stake: u64) -> bool {
        if let Some(validator) = self.validators.iter_mut().find(|v| v.address == address) {
            validator.stake = new_stake;
            true
        } else {
            false
        }
    }
}

impl Default for ValidatorSelector {
    fn default() -> Self {
        Self::new(Vec::new())
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_validator(id: u8, stake: u64) -> ValidatorInfo {
        ValidatorInfo {
            address: format!("validator_{}", id),
            stake,
            public_key: vec![id; 32],
        }
    }

    #[test]
    fn test_selector_creation() {
        let selector = ValidatorSelector::new(vec![]);
        assert_eq!(selector.validator_count(), 0);
    }

    #[test]
    fn test_selector_with_params() {
        let validators = vec![create_test_validator(1, 1000)];
        let selector = ValidatorSelector::with_params(validators, 500, 1, 10);
        assert!(selector.is_ok());
    }

    #[test]
    fn test_selector_invalid_params() {
        let validators = vec![create_test_validator(1, 1000)];
        let result = ValidatorSelector::with_params(validators, 500, 10, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_basic_selection() {
        let validators = vec![
            create_test_validator(1, 1000),
            create_test_validator(2, 2000),
            create_test_validator(3, 3000),
        ];

        let selector = ValidatorSelector::new(validators);
        let block_hash = Hash::from_bytes([0u8; 32]);

        let result = selector.select_validators(block_hash, 2);
        assert!(result.is_ok());

        let selection = result.unwrap();
        assert_eq!(selection.selected_validators.len(), 2);
        assert_eq!(selection.total_stake, 6000);
    }

    #[test]
    fn test_selection_with_min_stake() {
        let validators = vec![
            create_test_validator(1, 100),
            create_test_validator(2, 2000),
            create_test_validator(3, 3000),
        ];

        let mut selector = ValidatorSelector::new(validators);
        selector.min_stake = 500;

        let block_hash = Hash::from_bytes([1u8; 32]);
        let result = selector.select_validators(block_hash, 2);
        
        assert!(result.is_ok());
        let selection = result.unwrap();
        assert_eq!(selection.selected_validators.len(), 2);
        
        // Verify only validators with stake >= 500 are selected
        for validator in &selection.selected_validators {
            assert!(validator.stake >= 500);
        }
    }

    #[test]
    fn test_insufficient_validators() {
        let validators = vec![
            create_test_validator(1, 1000),
        ];

        let selector = ValidatorSelector::new(validators);
        let block_hash = Hash::from_bytes([2u8; 32]);

        let result = selector.select_validators(block_hash, 5);
        assert!(result.is_err());

        match result {
            Err(SelectionError::InsufficientValidators { eligible, required }) => {
                assert_eq!(eligible, 1);
                assert_eq!(required, 5);
            }
            _ => panic!("Expected InsufficientValidators error"),
        }
    }

    #[test]
    fn test_no_validators_available() {
        let validators = vec![
            create_test_validator(1, 100),
            create_test_validator(2, 200),
        ];

        let mut selector = ValidatorSelector::new(validators);
        selector.min_stake = 1000;

        let block_hash = Hash::from_bytes([3u8; 32]);
        let result = selector.select_validators(block_hash, 1);
        
        assert!(result.is_err());
        match result {
            Err(SelectionError::NoValidatorsAvailable) => (),
            _ => panic!("Expected NoValidatorsAvailable error"),
        }
    }

    #[test]
    fn test_determinism() {
        let validators = vec![
            create_test_validator(1, 1000),
            create_test_validator(2, 2000),
            create_test_validator(3, 3000),
        ];

        let selector = ValidatorSelector::new(validators);
        let block_hash = Hash::from_bytes([42u8; 32]);

        let result1 = selector.select_validators(block_hash, 2).unwrap();
        let result2 = selector.select_validators(block_hash, 2).unwrap();

        assert_eq!(result1.selected_validators, result2.selected_validators);
        assert_eq!(result1.seed, result2.seed);
    }

    #[test]
    fn test_different_hashes_different_selections() {
        let validators = vec![
            create_test_validator(1, 1000),
            create_test_validator(2, 2000),
            create_test_validator(3, 3000),
        ];

        let selector = ValidatorSelector::new(validators);
        let hash1 = Hash::from_bytes([1u8; 32]);
        let hash2 = Hash::from_bytes([2u8; 32]);

        let result1 = selector.select_validators(hash1, 2).unwrap();
        let result2 = selector.select_validators(hash2, 2).unwrap();

        // Different hashes should produce different seeds
        assert_ne!(result1.seed, result2.seed);
    }

    #[test]
    fn test_sample_size_validation() {
        let validators = vec![
            create_test_validator(1, 1000),
            create_test_validator(2, 2000),
        ];

        let selector = ValidatorSelector::new(validators);
        let block_hash = Hash::from_bytes([4u8; 32]);

        // Sample size too large
        let result = selector.select_validators(block_hash, 100);
        assert!(result.is_err());
        match result {
            Err(SelectionError::SampleSizeTooLarge { .. }) => (),
            _ => panic!("Expected SampleSizeTooLarge error"),
        }

        // Sample size too small
        let result = selector.select_validators(block_hash, 0);
        assert!(result.is_err());
        match result {
            Err(SelectionError::SampleSizeTooSmall { .. }) => (),
            _ => panic!("Expected SampleSizeTooSmall error"),
        }
    }

    #[test]
    fn test_add_validator() {
        let mut selector = ValidatorSelector::new(vec![]);
        assert_eq!(selector.validator_count(), 0);

        selector.add_validator(create_test_validator(1, 1000));
        assert_eq!(selector.validator_count(), 1);
    }

    #[test]
    fn test_remove_validator() {
        let validators = vec![
            create_test_validator(1, 1000),
            create_test_validator(2, 2000),
        ];

        let mut selector = ValidatorSelector::new(validators);
        assert_eq!(selector.validator_count(), 2);

        let removed = selector.remove_validator("validator_1");
        assert!(removed);
        assert_eq!(selector.validator_count(), 1);
    }

    #[test]
    fn test_update_stake() {
        let validators = vec![create_test_validator(1, 1000)];
        let mut selector = ValidatorSelector::new(validators);

        let updated = selector.update_stake("validator_1", 5000);
        assert!(updated);
        
        let validator = selector.validators.first().unwrap();
        assert_eq!(validator.stake, 5000);
    }

    #[test]
    fn test_eligible_count() {
        let validators = vec![
            create_test_validator(1, 100),
            create_test_validator(2, 2000),
            create_test_validator(3, 3000),
        ];

        let mut selector = ValidatorSelector::new(validators);
        selector.min_stake = 500;

        assert_eq!(selector.eligible_count(), 2);
    }

    #[test]
    fn test_weighted_distribution() {
        // Test that higher stake validators are selected more often
        let validators = vec![
            create_test_validator(1, 100),    // Low stake
            create_test_validator(2, 10000), // High stake
            create_test_validator(3, 100),    // Low stake
        ];

        let selector = ValidatorSelector::new(validators);
        let block_hash = Hash::from_bytes([5u8; 32]);

        let mut counts = std::collections::HashMap::new();
        let iterations = 100;

        for i in 0..iterations {
            let hash = Hash::from_bytes([i as u8; 32]);
            if let Ok(selection) = selector.select_validators(hash, 1) {
                let validator = &selection.selected_validators[0];
                *counts.entry(validator.address.clone()).or_insert(0) += 1;
            }
        }

        // Validator 2 should be selected most often due to higher stake
        let count_2 = counts.get("validator_2").unwrap_or(&0);
        let count_1 = counts.get("validator_1").unwrap_or(&0);
        let count_3 = counts.get("validator_3").unwrap_or(&0);

        assert!(count_2 > count_1);
        assert!(count_2 > count_3);
    }
}
