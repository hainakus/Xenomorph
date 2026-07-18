use blake3;
use borsh::{BorshDeserialize, BorshSerialize};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct ModelCheckpoint {
    pub block_height: u64,
    pub model_id: String,
    pub version: u32,
    pub weights_hash: [u8; 32],
    pub metrics: ModelMetrics,
    pub encryption: EncryptionData,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct ModelMetrics {
    pub loss: f64,
    pub accuracy: Option<f64>,
    pub f1_score: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
}

#[derive(Debug, Clone, BorshSerialize, BorshDeserialize, Serialize, Deserialize)]
pub struct EncryptionData {
    pub key_hash: [u8; 32],
    pub nonce: [u8; 12],
    pub algorithm: String,
}

impl Default for EncryptionData {
    fn default() -> Self {
        Self { key_hash: [0u8; 32], nonce: [0u8; 12], algorithm: "AES-256-GCM".to_string() }
    }
}

impl Default for ModelMetrics {
    fn default() -> Self {
        Self { loss: 0.0, accuracy: None, f1_score: None, precision: None, recall: None }
    }
}

impl ModelCheckpoint {
    pub fn new(block_height: u64, model_id: String, version: u32, weights: &[u8], metrics: ModelMetrics) -> Self {
        let weights_hash = blake3::hash(weights);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(weights_hash.as_bytes());

        Self { block_height, model_id, version, weights_hash: hash, metrics, encryption: EncryptionData::default() }
    }

    pub fn verify_integrity(&self, data: &[u8]) -> bool {
        let hash = blake3::hash(data);
        hash.as_bytes() == &self.weights_hash
    }

    pub fn with_encryption(mut self, encryption: EncryptionData) -> Self {
        self.encryption = encryption;
        self
    }

    pub fn serialize(&self) -> Result<Vec<u8>, std::io::Error> {
        borsh::to_vec(self).map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, std::io::Error> {
        <Self as BorshDeserialize>::try_from_slice(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_serialization() {
        let checkpoint = ModelCheckpoint {
            block_height: 1000,
            model_id: "dnabert2".to_string(),
            version: 1,
            weights_hash: [0u8; 32],
            metrics: ModelMetrics { loss: 0.5, accuracy: Some(0.9), f1_score: None, precision: None, recall: None },
            encryption: EncryptionData::default(),
        };

        let serialized = checkpoint.serialize().unwrap();
        let deserialized = ModelCheckpoint::deserialize(&serialized).unwrap();

        assert_eq!(checkpoint.block_height, deserialized.block_height);
        assert_eq!(checkpoint.model_id, deserialized.model_id);
    }

    #[test]
    fn test_integrity_verification() {
        let weights = vec![1u8, 2, 3, 4, 5];
        let checkpoint = ModelCheckpoint::new(1000, "test_model".to_string(), 1, &weights, ModelMetrics::default());

        assert!(checkpoint.verify_integrity(&weights));
        assert!(!checkpoint.verify_integrity(&vec![6, 7, 8]));
    }
}
