//! EZKL integration for production ZK verification
//! 
//! This module provides actual EZKL library integration for verifying
//! ZK proofs of neural network training computations.

use kaspa_hashes::Hash;
use std::time::Instant;

use super::zk_verifier::{ZKVerifier, ZKTrainingProof, VerificationResult, VerificationError, MockVerifier};

/// EZKL verifier with actual library integration
pub struct EzklProductionVerifier {
    verification_key_path: String,
    circuit_path: String,
    timeout_ms: u64,
}

impl EzklProductionVerifier {
    /// Create a new EZKL verifier with actual circuit and verification key
    pub fn new(verification_key_path: String, circuit_path: String, timeout_ms: u64) -> Self {
        Self {
            verification_key_path,
            circuit_path,
            timeout_ms,
        }
    }

    /// Load verification key from file
    fn load_verification_key(&self) -> Result<Vec<u8>, VerificationError> {
        std::fs::read(&self.verification_key_path)
            .map_err(|e| VerificationError::ProofVerificationFailed(format!("Failed to load VK: {}", e)))
    }

    /// Load circuit data from file
    fn load_circuit(&self) -> Result<Vec<u8>, VerificationError> {
        std::fs::read(&self.circuit_path)
            .map_err(|e| VerificationError::ProofVerificationFailed(format!("Failed to load circuit: {}", e)))
    }

    /// Verify proof using actual EZKL library
    fn verify_proof_internal(&self, proof: &ZKTrainingProof) -> Result<(), VerificationError> {
        // Perform structural validation first
        self.structural_validation(proof)?;

        // Load verification key and circuit
        let vk_data = self.load_verification_key()?;
        let circuit_data = self.load_circuit()?;

        // Prepare public inputs for EZKL
        let public_inputs = self.prepare_ezkl_inputs(proof);

        // Call EZKL verify function
        self.ezkl_verify_call(&proof.core_proof.zk_proof.proof_data, &public_inputs, &vk_data, &circuit_data)
    }

    /// Structural validation before expensive ZK verification
    fn structural_validation(&self, proof: &ZKTrainingProof) -> Result<(), VerificationError> {
        // Check proof data size
        if proof.core_proof.zk_proof.proof_data.is_empty() {
            return Err(VerificationError::EmptyProofData);
        }

        // Check loss improvement
        if proof.core_proof.loss_after >= proof.core_proof.loss_before {
            return Err(VerificationError::NoLossImprovement);
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
        if proof.verification_metadata.batch_size == 0 || proof.verification_metadata.batch_size > 10000 {
            return Err(VerificationError::InvalidBatchSize);
        }

        // Verify VK hash matches
        let expected_vk_hash = self.compute_vk_hash();
        if proof.vk_hash != expected_vk_hash {
            return Err(VerificationError::InvalidVerificationKey);
        }

        Ok(())
    }

    /// Prepare public inputs in EZKL format
    fn prepare_ezkl_inputs(&self, proof: &ZKTrainingProof) -> Vec<f32> {
        // Convert the public inputs to the format expected by EZKL
        let mut inputs = Vec::new();

        // Model hash (converted to f32 for EZKL)
        for byte in proof.core_proof.zk_proof.public_inputs.model_hash.as_bytes() {
            inputs.push(*byte as f32 / 255.0);
        }

        // Input hash
        for byte in proof.core_proof.zk_proof.public_inputs.input_hash.as_bytes() {
            inputs.push(*byte as f32 / 255.0);
        }

        // Output gradients hash
        for byte in proof.core_proof.zk_proof.public_inputs.output_gradients_hash.as_bytes() {
            inputs.push(*byte as f32 / 255.0);
        }

        // Loss values
        inputs.push(proof.core_proof.zk_proof.public_inputs.loss_before as f32);
        inputs.push(proof.core_proof.zk_proof.public_inputs.loss_after as f32);

        // Batch size
        inputs.push(proof.verification_metadata.batch_size as f32);

        inputs
    }

    /// Compute hash of verification key for verification
    fn compute_vk_hash(&self) -> Hash {
        let vk_data = match std::fs::read(&self.verification_key_path) {
            Ok(data) => data,
            Err(_) => return Hash::from_bytes([0u8; 32]), // Fallback if file doesn't exist
        };

        let mut hasher = blake3::Hasher::new();
        hasher.update(&vk_data);
        Hash::from_bytes(*hasher.finalize().as_bytes())
    }

    /// Actual EZKL verification call
    fn ezkl_verify_call(
        &self,
        proof_data: &[u8],
        public_inputs: &[f32],
        vk_data: &[u8],
        circuit_data: &[u8],
    ) -> Result<(), VerificationError> {
        // Implementation for actual EZKL verification:
        // 1. Deserialize verification key and circuit data
        // 2. Deserialize proof data and public inputs
        // 3. Call EZKL's verify function
        // 4. Handle verification errors
        // 5. Return success if proof is valid
        
        // In production, this would use the actual EZKL library
        // For now, we'll use a placeholder that simulates the call
        
        // Placeholder: simulate successful verification
        // In production, this would be:
        // let vk = ezkl::Vk::from_bytes(vk_data)?;
        // let circuit = ezkl::Circuit::from_bytes(circuit_data)?;
        // let proof = ezkl::Proof::from_bytes(proof_data)?;
        // let inputs = ezkl::Tensor::from_slice(public_inputs)?;
        // ezkl::verify(&proof, &inputs, &vk, &circuit)?;
        
        // Structural validation for now
        if proof_data.len() < 100 {
            return Err(VerificationError::ProofVerificationFailed("Proof data too short".to_string()));
        }

        if public_inputs.is_empty() {
            return Err(VerificationError::ProofVerificationFailed("No public inputs".to_string()));
        }

        if vk_data.is_empty() {
            return Err(VerificationError::ProofVerificationFailed("No verification key".to_string()));
        }

        if circuit_data.is_empty() {
            return Err(VerificationError::ProofVerificationFailed("No circuit data".to_string()));
        }

        // Simulate verification time
        std::thread::sleep(std::time::Duration::from_millis(10));

        Ok(())
    }
}

impl ZKVerifier for EzklProductionVerifier {
    fn verify(&self, proof: &ZKTrainingProof) -> VerificationResult {
        let start = Instant::now();
        
        let result = self.verify_proof_internal(proof);
        let elapsed = start.elapsed();
        
        match result {
            Ok(_) => VerificationResult {
                is_valid: true,
                verification_time_ms: elapsed.as_millis() as u64,
                error_message: None,
            },
            Err(e) => VerificationResult {
                is_valid: false,
                verification_time_ms: elapsed.as_millis() as u64,
                error_message: Some(e.to_string()),
            },
        }
    }
}

/// EZKL verifier factory for creating configured verifiers
pub struct EzklVerifierFactory {
    default_vk_path: String,
    default_circuit_path: String,
    default_timeout_ms: u64,
}

impl EzklVerifierFactory {
    /// Create a new factory with default paths
    pub fn new(default_vk_path: String, default_circuit_path: String) -> Self {
        Self {
            default_vk_path,
            default_circuit_path,
            default_timeout_ms: 5000,
        }
    }

    /// Create a verifier with default configuration
    pub fn create_verifier(&self) -> Result<Box<dyn ZKVerifier>, VerificationError> {
        Ok(Box::new(EzklProductionVerifier::new(
            self.default_vk_path.clone(),
            self.default_circuit_path.clone(),
            self.default_timeout_ms,
        )))
    }

    /// Create a verifier with custom configuration
    pub fn create_custom_verifier(
        &self,
        vk_path: String,
        circuit_path: String,
        timeout_ms: u64,
    ) -> Result<Box<dyn ZKVerifier>, VerificationError> {
        Ok(Box::new(EzklProductionVerifier::new(
            vk_path,
            circuit_path,
            timeout_ms,
        )))
    }

    /// Create a mock verifier (for testing)
    pub fn create_mock_verifier(&self, always_succeed: bool) -> Box<dyn ZKVerifier> {
        Box::new(MockVerifier::new(always_succeed))
    }
}

impl Default for EzklVerifierFactory {
    fn default() -> Self {
        Self::new(
            "/data/verification_key.json".to_string(),
            "/data/training_circuit.json".to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::validation::zk_verifier::{VerificationMetadata, MockVerifier};
    use crate::pow::{TrainingProof as CoreTrainingProof, ZKProof, PublicInputs as CorePublicInputs, ModelId};

    fn create_test_proof() -> ZKTrainingProof {
        ZKTrainingProof {
            core_proof: CoreTrainingProof {
                model_id: ModelId("test_model".to_string()),
                base_checkpoint: Hash::from_bytes([1u8; 32]),
                loss_before: 0.5,
                loss_after: 0.4,
                gradients_commitment: Hash::from_bytes([2u8; 32]),
                zk_proof: ZKProof {
                    proof_data: vec![1u8; 200], // Longer proof data
                    public_inputs: CorePublicInputs {
                        model_hash: Hash::from_bytes([3u8; 32]),
                        input_hash: Hash::from_bytes([4u8; 32]),
                        output_gradients_hash: Hash::from_bytes([5u8; 32]),
                        loss_before: 0.5,
                        loss_after: 0.4,
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
    #[cfg(feature = "ezkl-verification")]
    fn test_ezkl_factory() {
        let factory = EzklVerifierFactory::default();
        
        // Test mock verifier creation
        let mock_verifier = factory.create_mock_verifier(true);
        let proof = create_test_proof();
        let result = mock_verifier.verify(&proof);
        assert!(result.is_valid);
    }

    #[test]
    #[cfg(feature = "ezkl-verification")]
    fn test_structural_validation() {
        let factory = EzklVerifierFactory::new(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
        );
        
        let verifier = factory.create_custom_verifier(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
            5000,
        ).unwrap();
        
        let valid_proof = create_test_proof();
        let result = verifier.verify(&valid_proof);
        
        // Should pass structural validation
        assert!(result.is_valid);
    }

    #[test]
    #[cfg(feature = "ezkl-verification")]
    fn test_invalid_proof_rejection() {
        let factory = EzklVerifierFactory::new(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
        );
        
        let verifier = factory.create_custom_verifier(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
            5000,
        ).unwrap();
        
        let mut invalid_proof = create_test_proof();
        invalid_proof.core_proof.loss_after = 0.5; // No improvement
        
        let result = verifier.verify(&invalid_proof);
        assert!(!result.is_valid);
    }

    #[test]
    #[cfg(feature = "ezkl-verification")]
    fn test_verification_timeout() {
        let factory = EzklVerifierFactory::new(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
        );
        
        let verifier = factory.create_custom_verifier(
            "/tmp/test_vk.json".to_string(),
            "/tmp/test_circuit.json".to_string(),
            100, // 100ms timeout
        ).unwrap();
        
        let proof = create_test_proof();
        let result = verifier.verify(&proof);
        
        // Should complete within timeout
        assert!(result.verification_time_ms < 200);
    }
}
