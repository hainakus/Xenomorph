use blake3;
use chrono::Utc;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofOfService {
    pub model_id: String,
    pub query_id: String,
    pub seed_node_id: String,
    pub timestamp: u64,
    pub latency_ms: u64,
    pub input_hash: Vec<u8>,
    pub output_hash: Vec<u8>,
    pub signature: Vec<u8>,
}

pub struct ProofGenerator {
    seed_node_id: String,
}

impl ProofGenerator {
    pub fn new() -> Self {
        Self { seed_node_id: uuid::Uuid::new_v4().to_string() }
    }

    pub fn with_node_id(seed_node_id: String) -> Self {
        Self { seed_node_id }
    }

    pub fn generate_proof(&self, model_id: &str, query_id: &str, latency_ms: u64) -> Vec<u8> {
        let proof = ProofOfService {
            model_id: model_id.to_string(),
            query_id: query_id.to_string(),
            seed_node_id: self.seed_node_id.clone(),
            timestamp: Utc::now().timestamp() as u64,
            latency_ms,
            input_hash: vec![0u8; 32],
            output_hash: vec![0u8; 32],
            signature: vec![0u8; 64],
        };

        bincode::serialize(&proof).unwrap_or_default()
    }

    pub fn generate_proof_with_hashes(
        &self,
        model_id: &str,
        query_id: &str,
        latency_ms: u64,
        input_hash: Vec<u8>,
        output_hash: Vec<u8>,
    ) -> Vec<u8> {
        let proof = ProofOfService {
            model_id: model_id.to_string(),
            query_id: query_id.to_string(),
            seed_node_id: self.seed_node_id.clone(),
            timestamp: Utc::now().timestamp() as u64,
            latency_ms,
            input_hash,
            output_hash,
            signature: vec![0u8; 64],
        };

        bincode::serialize(&proof).unwrap_or_default()
    }

    pub fn compute_hash(&self, data: &[u8]) -> Vec<u8> {
        let hash = blake3::hash(data);
        hash.as_bytes().to_vec()
    }

    pub fn verify_proof(&self, proof_data: &[u8]) -> bool {
        if let Ok(proof) = bincode::deserialize::<ProofOfService>(proof_data) {
            // Verify timestamp is recent (within 5 minutes)
            let now = Utc::now().timestamp() as u64;
            let max_age = 300; // 5 minutes
            if now > proof.timestamp + max_age {
                return false;
            }

            // Verify seed node ID matches
            if proof.seed_node_id != self.seed_node_id {
                return false;
            }

            true
        } else {
            false
        }
    }
}

impl Default for ProofGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_generation() {
        let generator = ProofGenerator::new();
        let proof = generator.generate_proof("model1", "query1", 100);

        assert!(!proof.is_empty());
    }

    #[test]
    fn test_proof_verification() {
        let generator = ProofGenerator::new();
        let proof = generator.generate_proof("model1", "query1", 100);

        assert!(generator.verify_proof(&proof));
    }

    #[test]
    fn test_hash_computation() {
        let generator = ProofGenerator::new();
        let data = b"test data";
        let hash1 = generator.compute_hash(data);
        let hash2 = generator.compute_hash(data);

        assert_eq!(hash1, hash2);
    }
}
