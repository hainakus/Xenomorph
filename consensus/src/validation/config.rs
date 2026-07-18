//! Configuration parameters for ZK validation system
//!
//! This module defines all configurable parameters for the validation system,
//! including validator selection, timeout settings, and consensus thresholds.

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ============================================================================
// CONSTANTS
// ============================================================================
const DEFAULT_SAMPLE_SIZE: usize = 10;
const DEFAULT_MIN_STAKE: u64 = 1_000_000_000_000;
const DEFAULT_APPROVAL_THRESHOLD: f64 = 0.7;
const DEFAULT_TIMEOUT_MS: u64 = 5000;
const DEFAULT_CHECKPOINT_INTERVAL: u64 = 400;
const MIN_SAMPLE_SIZE: usize = 1;
const MAX_SAMPLE_SIZE: usize = 100;
const MIN_APPROVAL_THRESHOLD: f64 = 0.5;
const MAX_APPROVAL_THRESHOLD: f64 = 1.0;
const MIN_TIMEOUT_MS: u64 = 1000;
const MAX_TIMEOUT_MS: u64 = 60000;
const MIN_CHECKPOINT_INTERVAL: u64 = 100;
const MAX_CHECKPOINT_INTERVAL: u64 = 10000;

// ============================================================================
// ERRORS
// ============================================================================
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("invalid sample size: {size} (must be between {min} and {max})")]
    InvalidSampleSize { size: usize, min: usize, max: usize },

    #[error("invalid approval threshold: {threshold} (must be between {min} and {max})")]
    InvalidApprovalThreshold { threshold: f64, min: f64, max: f64 },

    #[error("invalid timeout: {timeout}ms (must be between {min}ms and {max}ms)")]
    InvalidTimeout { timeout: u64, min: u64, max: u64 },

    #[error("invalid checkpoint interval: {interval} (must be between {min} and {max})")]
    InvalidCheckpointInterval { interval: u64, min: u64, max: u64 },

    #[error("configuration error: {0}")]
    ConfigError(String),
}

// ============================================================================
// STRUCTS
// ============================================================================
/// Validation parameters configuration
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ValidationParams {
    /// Number of validators to select for each block
    pub sample_size: usize,

    /// Minimum stake required to be eligible for selection
    pub min_stake: u64,

    /// Approval threshold for consensus (0.0 to 1.0)
    pub approval_threshold: f64,

    /// Timeout for collecting signatures in milliseconds
    pub signature_timeout_ms: u64,

    /// Interval for full checkpoint validation in blocks
    pub checkpoint_interval: u64,

    /// Maximum number of validators in the selection pool
    pub max_validators: usize,

    /// Minimum number of validators required
    pub min_validators: usize,
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================
impl ValidationParams {
    /// Create new validation parameters with defaults
    pub fn new() -> Self {
        Self {
            sample_size: DEFAULT_SAMPLE_SIZE,
            min_stake: DEFAULT_MIN_STAKE,
            approval_threshold: DEFAULT_APPROVAL_THRESHOLD,
            signature_timeout_ms: DEFAULT_TIMEOUT_MS,
            checkpoint_interval: DEFAULT_CHECKPOINT_INTERVAL,
            max_validators: MAX_SAMPLE_SIZE,
            min_validators: MIN_SAMPLE_SIZE,
        }
    }

    /// Create validation parameters with custom values
    pub fn with_params(
        sample_size: usize,
        min_stake: u64,
        approval_threshold: f64,
        signature_timeout_ms: u64,
        checkpoint_interval: u64,
    ) -> Result<Self, ConfigError> {
        let mut params = Self::new();

        params.sample_size = sample_size;
        params.min_stake = min_stake;
        params.approval_threshold = approval_threshold;
        params.signature_timeout_ms = signature_timeout_ms;
        params.checkpoint_interval = checkpoint_interval;

        params.validate()?;
        Ok(params)
    }

    /// Validate the configuration parameters
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.sample_size < MIN_SAMPLE_SIZE || self.sample_size > MAX_SAMPLE_SIZE {
            return Err(ConfigError::InvalidSampleSize { size: self.sample_size, min: MIN_SAMPLE_SIZE, max: MAX_SAMPLE_SIZE });
        }

        if self.approval_threshold < MIN_APPROVAL_THRESHOLD || self.approval_threshold > MAX_APPROVAL_THRESHOLD {
            return Err(ConfigError::InvalidApprovalThreshold {
                threshold: self.approval_threshold,
                min: MIN_APPROVAL_THRESHOLD,
                max: MAX_APPROVAL_THRESHOLD,
            });
        }

        if self.signature_timeout_ms < MIN_TIMEOUT_MS || self.signature_timeout_ms > MAX_TIMEOUT_MS {
            return Err(ConfigError::InvalidTimeout { timeout: self.signature_timeout_ms, min: MIN_TIMEOUT_MS, max: MAX_TIMEOUT_MS });
        }

        if self.checkpoint_interval < MIN_CHECKPOINT_INTERVAL || self.checkpoint_interval > MAX_CHECKPOINT_INTERVAL {
            return Err(ConfigError::InvalidCheckpointInterval {
                interval: self.checkpoint_interval,
                min: MIN_CHECKPOINT_INTERVAL,
                max: MAX_CHECKPOINT_INTERVAL,
            });
        }

        if self.min_validators > self.max_validators {
            return Err(ConfigError::ConfigError("min_validators cannot exceed max_validators".to_string()));
        }

        if self.sample_size > self.max_validators {
            return Err(ConfigError::ConfigError("sample_size cannot exceed max_validators".to_string()));
        }

        Ok(())
    }

    /// Get the number of validators needed for quorum
    pub fn quorum_size(&self) -> usize {
        ((self.sample_size as f64) * self.approval_threshold).ceil() as usize
    }

    /// Check if a block height requires full checkpoint validation
    pub fn requires_checkpoint_validation(&self, block_height: u64) -> bool {
        block_height % self.checkpoint_interval == 0
    }

    /// Set sample size with validation
    pub fn set_sample_size(&mut self, size: usize) -> Result<(), ConfigError> {
        self.sample_size = size;
        self.validate()?;
        Ok(())
    }

    /// Set approval threshold with validation
    pub fn set_approval_threshold(&mut self, threshold: f64) -> Result<(), ConfigError> {
        self.approval_threshold = threshold;
        self.validate()?;
        Ok(())
    }

    /// Set timeout with validation
    pub fn set_timeout(&mut self, timeout_ms: u64) -> Result<(), ConfigError> {
        self.signature_timeout_ms = timeout_ms;
        self.validate()?;
        Ok(())
    }

    /// Set checkpoint interval with validation
    pub fn set_checkpoint_interval(&mut self, interval: u64) -> Result<(), ConfigError> {
        self.checkpoint_interval = interval;
        self.validate()?;
        Ok(())
    }
}

impl Default for ValidationParams {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_params() {
        let params = ValidationParams::new();
        assert_eq!(params.sample_size, DEFAULT_SAMPLE_SIZE);
        assert_eq!(params.min_stake, DEFAULT_MIN_STAKE);
        assert_eq!(params.approval_threshold, DEFAULT_APPROVAL_THRESHOLD);
        assert_eq!(params.signature_timeout_ms, DEFAULT_TIMEOUT_MS);
        assert_eq!(params.checkpoint_interval, DEFAULT_CHECKPOINT_INTERVAL);
    }

    #[test]
    fn test_params_validation_success() {
        let params = ValidationParams::with_params(10, 1_000_000_000_000, 0.7, 5000, 400);
        assert!(params.is_ok());
    }

    #[test]
    fn test_invalid_sample_size_too_small() {
        let params = ValidationParams::with_params(0, 1_000_000_000_000, 0.7, 5000, 400);
        assert!(params.is_err());
        match params {
            Err(ConfigError::InvalidSampleSize { size, min, .. }) => {
                assert_eq!(size, 0);
                assert_eq!(min, MIN_SAMPLE_SIZE);
            }
            _ => panic!("Expected InvalidSampleSize error"),
        }
    }

    #[test]
    fn test_invalid_sample_size_too_large() {
        let params = ValidationParams::with_params(200, 1_000_000_000_000, 0.7, 5000, 400);
        assert!(params.is_err());
        match params {
            Err(ConfigError::InvalidSampleSize { size, max, .. }) => {
                assert_eq!(size, 200);
                assert_eq!(max, MAX_SAMPLE_SIZE);
            }
            _ => panic!("Expected InvalidSampleSize error"),
        }
    }

    #[test]
    fn test_invalid_approval_threshold() {
        let params = ValidationParams::with_params(10, 1_000_000_000_000, 1.5, 5000, 400);
        assert!(params.is_err());
        match params {
            Err(ConfigError::InvalidApprovalThreshold { threshold, max, .. }) => {
                assert_eq!(threshold, 1.5);
                assert_eq!(max, MAX_APPROVAL_THRESHOLD);
            }
            _ => panic!("Expected InvalidApprovalThreshold error"),
        }
    }

    #[test]
    fn test_invalid_timeout() {
        let params = ValidationParams::with_params(10, 1_000_000_000_000, 0.7, 100, 400);
        assert!(params.is_err());
        match params {
            Err(ConfigError::InvalidTimeout { timeout, min, .. }) => {
                assert_eq!(timeout, 100);
                assert_eq!(min, MIN_TIMEOUT_MS);
            }
            _ => panic!("Expected InvalidTimeout error"),
        }
    }

    #[test]
    fn test_invalid_checkpoint_interval() {
        let params = ValidationParams::with_params(10, 1_000_000_000_000, 0.7, 5000, 50);
        assert!(params.is_err());
        match params {
            Err(ConfigError::InvalidCheckpointInterval { interval, min, .. }) => {
                assert_eq!(interval, 50);
                assert_eq!(min, MIN_CHECKPOINT_INTERVAL);
            }
            _ => panic!("Expected InvalidCheckpointInterval error"),
        }
    }

    #[test]
    fn test_quorum_size() {
        let params = ValidationParams::new();
        let quorum = params.quorum_size();
        assert_eq!(quorum, 7); // 10 * 0.7 = 7
    }

    #[test]
    fn test_requires_checkpoint_validation() {
        let params = ValidationParams::new();
        assert!(params.requires_checkpoint_validation(400));
        assert!(params.requires_checkpoint_validation(800));
        assert!(!params.requires_checkpoint_validation(401));
        assert!(!params.requires_checkpoint_validation(399));
    }

    #[test]
    fn test_set_sample_size() {
        let mut params = ValidationParams::new();
        let result = params.set_sample_size(20);
        assert!(result.is_ok());
        assert_eq!(params.sample_size, 20);
    }

    #[test]
    fn test_set_sample_size_invalid() {
        let mut params = ValidationParams::new();
        let result = params.set_sample_size(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_approval_threshold() {
        let mut params = ValidationParams::new();
        let result = params.set_approval_threshold(0.8);
        assert!(result.is_ok());
        assert_eq!(params.approval_threshold, 0.8);
    }

    #[test]
    fn test_set_approval_threshold_invalid() {
        let mut params = ValidationParams::new();
        let result = params.set_approval_threshold(1.5);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_timeout() {
        let mut params = ValidationParams::new();
        let result = params.set_timeout(10000);
        assert!(result.is_ok());
        assert_eq!(params.signature_timeout_ms, 10000);
    }

    #[test]
    fn test_set_timeout_invalid() {
        let mut params = ValidationParams::new();
        let result = params.set_timeout(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_set_checkpoint_interval() {
        let mut params = ValidationParams::new();
        let result = params.set_checkpoint_interval(800);
        assert!(result.is_ok());
        assert_eq!(params.checkpoint_interval, 800);
    }

    #[test]
    fn test_set_checkpoint_interval_invalid() {
        let mut params = ValidationParams::new();
        let result = params.set_checkpoint_interval(50);
        assert!(result.is_err());
    }

    #[test]
    fn test_serialization() {
        let params = ValidationParams::new();
        let serialized = serde_json::to_string(&params);
        assert!(serialized.is_ok());

        let deserialized: Result<ValidationParams, _> = serde_json::from_str(&serialized.unwrap());
        assert!(deserialized.is_ok());
        assert_eq!(deserialized.unwrap(), params);
    }
}
