use anyhow::Result;
use borsh::{to_vec, BorshDeserialize, BorshSerialize};

use crate::trainer::TrainingResult;

const PROOF_VERSION: u8 = 1;

/// Public inputs used to verify a ZK proof.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct PublicInputs {
    pub model_id: String,
    pub batch_id: u64,
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
    pub base_checkpoint: [u8; 32],
}

/// Generates and verifies ZK proofs of useful training work.
pub struct ZkProver;

impl ZkProver {
    pub fn new() -> Self {
        Self
    }

    /// Generate a deterministic proof from a training result.
    /// In production this would be replaced by an EZKL-style proof; for the
    /// UsefulPoW prototype the proof is a keyed hash over public inputs.
    pub fn generate_proof(&self, result: &TrainingResult, public_inputs: &PublicInputs) -> Result<Vec<u8>> {
        let mut proof = Vec::with_capacity(1 + 32 + 32);
        proof.push(PROOF_VERSION);
        proof.extend_from_slice(&public_inputs.gradients_commitment);

        let mut hasher = blake3::Hasher::new();
        hasher.update(&[PROOF_VERSION]);
        hasher.update(&result.gradients_commitment);
        let payload = to_vec(public_inputs)?;
        hasher.update(&payload);

        let hash = hasher.finalize();
        proof.extend_from_slice(hash.as_bytes());

        Ok(proof)
    }

    /// Verify that a proof matches the provided public inputs and result.
    pub fn verify_proof(
        &self,
        proof: &[u8],
        result: &TrainingResult,
        public_inputs: &PublicInputs,
    ) -> bool {
        if proof.len() != 1 + 32 + 32 {
            return false;
        }

        if proof[0] != PROOF_VERSION {
            return false;
        }

        let expected = match self.generate_proof(result, public_inputs) {
            Ok(p) => p,
            Err(_) => return false,
        };

        proof == expected
    }
}

impl Default for ZkProver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_result() -> TrainingResult {
        TrainingResult {
            model_id: "dnabert2".to_string(),
            batch_indices: vec![0, 1, 2],
            base_checkpoint: [0u8; 32],
            loss_before: 2.45,
            loss_after: 2.41,
            gradients_commitment: [1u8; 32],
            compute_time_ms: 100,
        }
    }

    fn dummy_inputs() -> PublicInputs {
        PublicInputs {
            model_id: "dnabert2".to_string(),
            batch_id: 1,
            loss_before: 2.45,
            loss_after: 2.41,
            gradients_commitment: [1u8; 32],
            base_checkpoint: [0u8; 32],
        }
    }

    #[test]
    fn test_proof_roundtrip() {
        let prover = ZkProver::new();
        let result = dummy_result();
        let inputs = dummy_inputs();

        let proof = prover.generate_proof(&result, &inputs).unwrap();
        assert!(prover.verify_proof(&proof, &result, &inputs));
    }

    #[test]
    fn test_proof_tampered_inputs_fail() {
        let prover = ZkProver::new();
        let result = dummy_result();
        let mut inputs = dummy_inputs();
        let proof = prover.generate_proof(&result, &inputs).unwrap();

        inputs.loss_after = 2.40;
        assert!(!prover.verify_proof(&proof, &result, &inputs));
    }
}
