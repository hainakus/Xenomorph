//! Validation network protocol for validator coordination
//!
//! This module implements the P2P message protocol for validators to coordinate
//! on validation, including signature aggregation and consensus mechanisms.

use thiserror::Error;
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::zk_verifier::VerificationResult;
use super::selector::ValidatorSelectionSimple;

// ============================================================================
// CONSTANTS
// ============================================================================
const DEFAULT_TIMEOUT_MS: u64 = 5000;
const DEFAULT_FANOUT: usize = 8;
const MAX_PEERS: usize = 100;
const GOSSIP_INTERVAL_MS: u64 = 1000;
const MAX_MESSAGE_AGE_MS: u64 = 60000;

// ============================================================================
// ERRORS
// ============================================================================
#[derive(Error, Debug)]
pub enum NetworkError {
    #[error("network error: {0}")]
    NetworkError(String),
    
    #[error("invalid message format")]
    InvalidMessage,
    
    #[error("signature verification failed")]
    SignatureVerificationFailed,
    
    #[error("timeout waiting for signatures")]
    Timeout,
    
    #[error("insufficient signatures: {have}/{required}")]
    InsufficientSignatures { have: usize, required: usize },
    
    #[error("peer not found: {0}")]
    PeerNotFound(String),
}

// ============================================================================
// STRUCTS
// ============================================================================
/// Validation message types
#[derive(Clone, Debug)]
pub enum ValidationMessage {
    /// Request for validation
    ValidationRequest {
        block_hash: [u8; 32],
    },
    
    /// Response with validation result
    ValidationResponse {
        block_hash: [u8; 32],
        validator: String,
        result: VerificationResult,
    },
    
    /// Challenge for suspected invalid block
    SelectionChallenge {
        block_hash: [u8; 32],
        challenge_data: Vec<u8>,
    },
    
    /// Broadcast of aggregated signature
    AggregatedSignatureBroadcast {
        block_hash: [u8; 32],
        aggregated_signature: AggregatedSignature,
    },
}

/// Network events
#[derive(Clone, Debug)]
pub enum NetworkEvent {
    /// Validation requested
    ValidationRequested { block_hash: [u8; 32] },
    
    /// Validation response received
    ValidationResponseReceived { validator: String, result: VerificationResult },
    
    /// Challenge received
    ChallengeReceived { block_hash: [u8; 32] },
    
    /// Aggregated signature received
    AggregatedSignatureReceived { block_hash: [u8; 32] },
    
    /// Not selected for validation
    NotSelectedForValidation,
}

/// Signature aggregator for collecting validator signatures
pub struct SignatureAggregator {
    required_signatures: usize,
    timeout_ms: u64,
    signatures: HashMap<String, ValidationSignature>,
    validation_results: HashMap<String, VerificationResult>,
    public_keys: HashMap<String, Vec<u8>>,
    start_time: Instant,
}

/// Validation signature
#[derive(Clone, Debug)]
pub struct ValidationSignature {
    pub validator: String,
    pub signature: Vec<u8>,
    pub timestamp: u64,
}

/// Aggregated signature from multiple validators
#[derive(Clone, Debug)]
pub struct AggregatedSignature {
    pub signatures: Vec<ValidationSignature>,
    pub block_hash: [u8; 32],
    pub approval_count: usize,
    pub rejection_count: usize,
}

/// Validation network for message processing
pub struct ValidationNetwork {
    local_address: String,
    timeout_ms: u64,
    approval_threshold: f64,
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================
impl SignatureAggregator {
    pub fn new(required_signatures: usize, timeout_ms: u64) -> Self {
        Self {
            required_signatures,
            timeout_ms,
            signatures: HashMap::new(),
            validation_results: HashMap::new(),
            public_keys: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    pub fn register_public_key(&mut self, validator: String, public_key: Vec<u8>) {
        self.public_keys.insert(validator, public_key);
    }

    pub fn add_signature(&mut self, validator: String, signature: Vec<u8>) {
        let sig = ValidationSignature {
            validator: validator.clone(),
            signature,
            timestamp: Instant::now().elapsed().as_millis() as u64,
        };
        self.signatures.insert(validator, sig);
    }

    pub fn add_validation_result(&mut self, validator: String, result: VerificationResult) {
        self.validation_results.insert(validator, result);
    }

    pub fn has_quorum(&self) -> bool {
        self.signatures.len() >= self.required_signatures
    }

    pub fn is_timeout(&self) -> bool {
        self.start_time.elapsed() > Duration::from_millis(self.timeout_ms)
    }

    pub fn aggregate(&self) -> Result<AggregatedSignature, NetworkError> {
        if !self.has_quorum() {
            return Err(NetworkError::InsufficientSignatures {
                have: self.signatures.len(),
                required: self.required_signatures,
            });
        }

        let mut approval_count = 0;
        let mut rejection_count = 0;

        for result in self.validation_results.values() {
            if result.is_valid {
                approval_count += 1;
            } else {
                rejection_count += 1;
            }
        }

        let signatures: Vec<ValidationSignature> = self.signatures.values().cloned().collect();

        Ok(AggregatedSignature {
            signatures,
            block_hash: [0u8; 32], // Would be set from context
            approval_count,
            rejection_count,
        })
    }
}

impl ValidationNetwork {
    pub fn new(local_address: String, timeout_ms: u64, approval_threshold: f64) -> Self {
        Self {
            local_address,
            timeout_ms,
            approval_threshold,
        }
    }

    pub fn process_message(&self, message: ValidationMessage) -> Result<NetworkEvent, NetworkError> {
        match message {
            ValidationMessage::ValidationRequest { block_hash } => {
                Ok(NetworkEvent::ValidationRequested { block_hash })
            }
            ValidationMessage::ValidationResponse { validator, result, .. } => {
                Ok(NetworkEvent::ValidationResponseReceived { validator, result })
            }
            ValidationMessage::SelectionChallenge { block_hash, .. } => {
                Ok(NetworkEvent::ChallengeReceived { block_hash })
            }
            ValidationMessage::AggregatedSignatureBroadcast { block_hash, .. } => {
                Ok(NetworkEvent::AggregatedSignatureReceived { block_hash })
            }
        }
    }

    pub fn should_accept_validation(&self, aggregator: &SignatureAggregator) -> bool {
        let approvals = aggregator.validation_results.values().filter(|r| r.is_valid).count();
        let total = aggregator.validation_results.len();

        if total == 0 {
            return false;
        }

        let approval_rate = approvals as f64 / total as f64;
        approval_rate >= self.approval_threshold
    }

    pub fn send_validation_request(
        &self,
        _validator_selection: &ValidatorSelectionSimple,
        _proof: &super::zk_verifier::ZKTrainingProof,
    ) -> Result<(), NetworkError> {
        // Serialize and send validation request to selected validators
        // In production, this would use the P2P protocol
        Ok(())
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

    fn create_verification_result(is_valid: bool) -> VerificationResult {
        VerificationResult {
            is_valid,
            verification_time_ms: 10,
            error_message: if is_valid { None } else { Some("Failed".to_string()) },
        }
    }

    #[test]
    fn test_signature_aggregator_creation() {
        let aggregator = SignatureAggregator::new(3, 5000);
        assert_eq!(aggregator.required_signatures, 3);
        assert!(!aggregator.has_quorum());
    }

    #[test]
    fn test_add_signature() {
        let mut aggregator = SignatureAggregator::new(3, 5000);
        aggregator.add_signature("validator1".to_string(), vec![1, 2, 3]);
        assert_eq!(aggregator.signatures.len(), 1);
    }

    #[test]
    fn test_quorum_reached() {
        let mut aggregator = SignatureAggregator::new(3, 5000);
        aggregator.add_signature("validator1".to_string(), vec![1, 2, 3]);
        aggregator.add_signature("validator2".to_string(), vec![4, 5, 6]);
        aggregator.add_signature("validator3".to_string(), vec![7, 8, 9]);
        assert!(aggregator.has_quorum());
    }

    #[test]
    fn test_aggregate() {
        let mut aggregator = SignatureAggregator::new(3, 5000);
        
        for i in 0..3 {
            aggregator.add_signature(format!("validator{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator{}", i),
                create_verification_result(true),
            );
        }

        let result = aggregator.aggregate();
        assert!(result.is_ok());
        
        let aggregated = result.unwrap();
        assert_eq!(aggregated.signatures.len(), 3);
        assert_eq!(aggregated.approval_count, 3);
    }

    #[test]
    fn test_aggregate_insufficient_signatures() {
        let mut aggregator = SignatureAggregator::new(5, 5000);
        
        for i in 0..3 {
            aggregator.add_signature(format!("validator{}", i), vec![i as u8; 32]);
        }

        let result = aggregator.aggregate();
        assert!(result.is_err());
    }

    #[test]
    fn test_should_accept_validation() {
        let network = ValidationNetwork::new("validator1".to_string(), 5000, 0.7);
        let mut aggregator = SignatureAggregator::new(7, 5000);
        
        for i in 0..7 {
            aggregator.add_signature(format!("validator{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator{}", i),
                create_verification_result(true),
            );
        }
        
        assert!(network.should_accept_validation(&aggregator));
    }

    #[test]
    fn test_should_reject_validation() {
        let network = ValidationNetwork::new("validator1".to_string(), 5000, 0.7);
        let mut aggregator = SignatureAggregator::new(7, 5000);
        
        for i in 0..3 {
            aggregator.add_signature(format!("validator{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator{}", i),
                create_verification_result(true),
            );
        }
        
        for i in 3..7 {
            aggregator.add_signature(format!("validator{}", i), vec![i as u8; 32]);
            aggregator.add_validation_result(
                format!("validator{}", i),
                create_verification_result(false),
            );
        }
        
        assert!(!network.should_accept_validation(&aggregator));
    }
}
