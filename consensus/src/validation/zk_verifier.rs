//! ZK proof verification for training computations
//!
//! This module provides verification for zero-knowledge proofs of AI model training,
//! ensuring that claimed training improvements are cryptographically verifiable.

use kaspa_hashes::Hash;
use thiserror::Error;
use std::time::Instant;

// ============================================================================
// CONSTANTS
// ============================================================================
const PROOF_DATA_MIN_SIZE: usize = 100;
const MAX_LOSS: f64 = 1000.0;
const MIN_LOSS: f64 = 0.0;

// ============================================================================
// ERRORS
// ============================================================================
#[derive(Error, Debug)]
pub enum VerificationError {
    #[error("empty proof data")]
    EmptyProofData,
    
    #[error("no loss improvement: before={before}, after={after}")]
    NoLossImprovement { before: f64, after: f64 },
    
    #[error("invalid model hash (zero hash)")]
    InvalidModelHash,
    
    #[error("invalid input hash (zero hash)")]
    InvalidInputHash,
    
    #[error("invalid gradients hash (zero hash)")]
    InvalidGradientsHash,
    
    #[error("invalid batch size: {size} (must be > 0 and <= 10000)")]
    InvalidBatchSize { size: u32 },
    
    #[error("verification key mismatch")]
    InvalidVerificationKey,
    
    #[error("proof verification failed: {0}")]
    ProofVerificationFailed(String),
    
    #[error("timeout after {0}ms")]
    Timeout(u64),
}

// ============================================================================
// TYPES
// ============================================================================
pub type ModelId = String;

// ============================================================================
// STRUCTS
// ============================================================================
/// Verification result with metadata
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationResult {
    pub is_valid: bool,
    pub verification_time_ms: u64,
    pub error_message: Option<String>,
}

/// Metadata about the verification
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerificationMetadata {
    pub batch_size: u32,
    pub nonce: u64,
    pub timestamp: u64,
}

/// ZK training proof structure
#[derive(Clone, Debug)]
pub struct ZKTrainingProof {
    pub core_proof: CoreTrainingProof,
    pub vk_hash: Hash,
    pub verification_metadata: VerificationMetadata,
}

/// Core training proof data
#[derive(Clone, Debug)]
pub struct CoreTrainingProof {
    pub model_id: ModelId,
    pub base_checkpoint: Hash,
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: Hash,
    pub zk_proof: ZKProof,
    pub batch_indices: Vec<u64>,
}

/// ZK proof with public inputs
#[derive(Clone, Debug)]
pub struct ZKProof {
    pub proof_data: Vec<u8>,
    pub public_inputs: PublicInputs,
}

/// Public inputs for ZK verification
#[derive(Clone, Debug)]
pub struct PublicInputs {
    pub model_hash: Hash,
    pub input_hash: Hash,
    pub output_gradients_hash: Hash,
    pub loss_before: f64,
    pub loss_after: f64,
}

// ============================================================================
// TRAITS
// ============================================================================
/// ZK verifier trait
pub trait ZKVerifier: Send + Sync {
    /// Verify a training proof
    fn verify(&self, proof: &ZKTrainingProof) -> VerificationResult;
    
    /// Get verification timeout in milliseconds
    fn timeout_ms(&self) -> u64;
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================
impl VerificationResult {
    /// Create a successful verification result
    pub fn success(time_ms: u64) -> Self {
        Self {
            is_valid: true,
            verification_time_ms: time_ms,
            error_message: None,
        }
    }
    
    /// Create a failed verification result
    pub fn failure(time_ms: u64, error: impl Into<String>) -> Self {
        Self {
            is_valid: false,
            verification_time_ms: time_ms,
            error_message: Some(error.into()),
        }
    }
}

impl Default for VerificationResult {
    fn default() -> Self {
        Self {
            is_valid: false,
            verification_time_ms: 0,
            error_message: None,
        }
    }
}

/// Mock verifier for testing
pub struct MockVerifier {
    timeout_ms: u64,
}

impl MockVerifier {
    pub fn new(timeout_ms: u64) -> Self {
        Self { timeout_ms }
    }
}

impl ZKVerifier for MockVerifier {
    fn verify(&self, proof: &ZKTrainingProof) -> VerificationResult {
        let start = Instant::now();
        
        // Perform structural validation
        let result = self.verify_proof_internal(proof);
        let elapsed = start.elapsed();
        
        match result {
            Ok(_) => VerificationResult::success(elapsed.as_millis() as u64),
            Err(e) => VerificationResult::failure(elapsed.as_millis() as u64, e),
        }
    }
    
    fn timeout_ms(&self) -> u64 {
        self.timeout_ms
    }
}

impl MockVerifier {
    fn verify_proof_internal(&self, proof: &ZKTrainingProof) -> Result<(), VerificationError> {
        // Check proof data size
        if proof.core_proof.zk_proof.proof_data.is_empty() {
            return Err(VerificationError::EmptyProofData);
        }

        // Check loss improvement
        if proof.core_proof.loss_after >= proof.core_proof.loss_before {
            return Err(VerificationError::NoLossImprovement {
                before: proof.core_proof.loss_before,
                after: proof.core_proof.loss_after,
            });
        }

        // Check hash validity (non-zero)
        if proof.core_proof.zk_proof.public_inputs.model_hash == Hash::from_bytes([0u8; 32]) {
            return Err(VerificationError::InvalidModelHash);
        }
        if proof.core_proof.zk_proof.public_inputs.input_hash == Hash::from_bytes([0u8; 32]) {
            return Err(VerificationError::InvalidInputHash);
        }
        if proof.core_proof.zk_proof.public_inputs.output_gradients_hash == Hash::from_bytes([0u8; 32]) {
            return Err(VerificationError::InvalidGradientsHash);
        }

        // Check batch size is reasonable
        let batch_size = proof.verification_metadata.batch_size;
        if batch_size == 0 || batch_size > 10000 {
            return Err(VerificationError::InvalidBatchSize { size: batch_size });
        }

        // Verify VK hash matches
        let expected_vk_hash = self.compute_vk_hash();
        if proof.vk_hash != expected_vk_hash {
            return Err(VerificationError::InvalidVerificationKey);
        }

        Ok(())
    }
    
    fn compute_vk_hash(&self) -> Hash {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"mock-verification-key");
        Hash::from_bytes(*hasher.finalize().as_bytes())
    }
}

impl Default for MockVerifier {
    fn default() -> Self {
        Self::new(5000)
    }
}

// ============================================================================
// TESTS
// ============================================================================
#[cfg(test)]
mod tests {
    use super::*;

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
                    public_inputs: PublicInputs {
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

    #[test]
    fn test_mock_verifier_valid_proof() {
        let verifier = MockVerifier::new(5000);
        let proof = create_test_proof(true);
        
        let result = verifier.verify(&proof);
        assert!(result.is_valid);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_mock_verifier_invalid_proof_no_improvement() {
        let verifier = MockVerifier::new(5000);
        let proof = create_test_proof(false);
        
        let result = verifier.verify(&proof);
        assert!(!result.is_valid);
        assert!(result.error_message.is_some());
    }

    #[test]
    fn test_mock_verifier_empty_proof_data() {
        let verifier = MockVerifier::new(5000);
        let mut proof = create_test_proof(true);
        proof.core_proof.zk_proof.proof_data = vec![];
        
        let result = verifier.verify(&proof);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_verification_result_success() {
        let result = VerificationResult::success(100);
        assert!(result.is_valid);
        assert_eq!(result.verification_time_ms, 100);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_verification_result_failure() {
        let result = VerificationResult::failure(100, "test error");
        assert!(!result.is_valid);
        assert_eq!(result.verification_time_ms, 100);
        assert_eq!(result.error_message, Some("test error".to_string()));
    }

    #[test]
    fn test_invalid_batch_size() {
        let verifier = MockVerifier::new(5000);
        let mut proof = create_test_proof(true);
        proof.verification_metadata.batch_size = 0;
        
        let result = verifier.verify(&proof);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_zero_model_hash() {
        let verifier = MockVerifier::new(5000);
        let mut proof = create_test_proof(true);
        proof.core_proof.zk_proof.public_inputs.model_hash = Hash::from_bytes([0u8; 32]);
        
        let result = verifier.verify(&proof);
        assert!(!result.is_valid);
    }
}
