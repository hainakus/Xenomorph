use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use anyhow::Result;
use rand::Rng;
use sha2::{Digest, Sha256};
use std::path::Path;
use thiserror::Error;
use tokio::fs;

use super::RawModelFiles;

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Encryption error: {0}")]
    EncryptionError(String),

    #[error("Decryption error: {0}")]
    DecryptionError(String),

    #[error("Invalid key")]
    InvalidKey,

    #[error("File not found: {0}")]
    FileNotFound(String),
}

pub struct ModelStorage {
    base_path: String,
    encryption_key: [u8; 32],
}

impl ModelStorage {
    pub fn new(base_path: String, encryption_key: [u8; 32]) -> Self {
        Self { base_path, encryption_key }
    }

    /// Sanitize a model identifier so it is safe to use in filesystem paths.
    /// Replaces path separators and other special characters with underscores.
    fn sanitize_id(model_id: &str) -> String {
        model_id.chars().map(|c| if c == '/' || c == '\\' || c == ':' || c == ' ' || c == '\0' { '_' } else { c }).collect()
    }

    fn model_path(&self, model_id: &str) -> String {
        let safe_id = Self::sanitize_id(model_id);
        format!("{}/{}", self.base_path, safe_id)
    }

    pub async fn store_model(&self, model_id: &str, data: &[u8]) -> Result<String, StorageError> {
        // Encrypt data
        let encrypted = self.encrypt(data)?;

        // Create directory if needed
        let model_path = self.model_path(model_id);
        fs::create_dir_all(&model_path).await?;

        // Store encrypted data
        let file_path = format!("{}/model.enc", model_path);
        fs::write(&file_path, encrypted).await?;

        // Store key hash for verification
        let key_hash = self.compute_key_hash();
        let key_path = format!("{}/model.keyhash", model_path);
        fs::write(&key_path, &key_hash).await?;

        Ok(file_path)
    }

    pub async fn store_model_files(&self, model_id: &str, files: &RawModelFiles) -> Result<(), StorageError> {
        let model_path = self.model_path(model_id);
        fs::create_dir_all(&model_path).await?;

        self.write_encrypted_file(&format!("{}/config.enc", model_path), &files.config).await?;
        self.write_encrypted_file(&format!("{}/tokenizer.enc", model_path), &files.tokenizer).await?;
        self.write_encrypted_file(&format!("{}/weights.enc", model_path), &files.weights).await?;

        Ok(())
    }

    pub async fn load_model_files(&self, model_id: &str) -> Result<RawModelFiles, StorageError> {
        let model_path = self.model_path(model_id);

        let config = self.read_encrypted_file(&format!("{}/config.enc", model_path)).await?;
        let tokenizer = self.read_encrypted_file(&format!("{}/tokenizer.enc", model_path)).await?;
        let weights = self.read_encrypted_file(&format!("{}/weights.enc", model_path)).await?;

        Ok(RawModelFiles { config, tokenizer, weights })
    }

    async fn write_encrypted_file(&self, path: &str, data: &[u8]) -> Result<(), StorageError> {
        let encrypted = self.encrypt(data)?;
        fs::write(path, encrypted).await?;
        Ok(())
    }

    async fn read_encrypted_file(&self, path: &str) -> Result<Vec<u8>, StorageError> {
        if !Path::new(path).exists() {
            return Err(StorageError::FileNotFound(path.to_string()));
        }
        let encrypted = fs::read(path).await?;
        self.decrypt(&encrypted)
    }

    pub async fn load_model(&self, model_id: &str) -> Result<Vec<u8>, StorageError> {
        let model_path = self.model_path(model_id);
        let file_path = format!("{}/model.enc", model_path);

        if !Path::new(&file_path).exists() {
            return Err(StorageError::FileNotFound(file_path));
        }

        let encrypted = fs::read(&file_path).await?;
        self.decrypt(&encrypted)
    }

    pub async fn load_checkpoint(&self, model_id: &str, version: u32) -> Result<Vec<u8>, StorageError> {
        let model_path = self.model_path(model_id);
        let file_path = format!("{}/checkpoint_{}.enc", model_path, version);

        if !Path::new(&file_path).exists() {
            return Err(StorageError::FileNotFound(file_path));
        }

        let encrypted = fs::read(&file_path).await?;
        self.decrypt(&encrypted)
    }

    pub async fn store_checkpoint(&self, model_id: &str, version: u32, data: &[u8]) -> Result<String, StorageError> {
        let encrypted = self.encrypt(data)?;

        let model_path = self.model_path(model_id);
        fs::create_dir_all(&model_path).await?;

        let file_path = format!("{}/checkpoint_{}.enc", model_path, version);
        fs::write(&file_path, encrypted).await?;

        Ok(file_path)
    }

    pub async fn list_checkpoints(&self, model_id: &str) -> Result<Vec<u32>, StorageError> {
        let model_path = self.model_path(model_id);

        if !Path::new(&model_path).exists() {
            return Ok(vec![]);
        }

        let mut entries = fs::read_dir(&model_path).await?;
        let mut versions = Vec::new();

        while let Some(entry) = entries.next_entry().await? {
            let file_name = entry.file_name();
            let name = file_name.to_string_lossy();

            if name.starts_with("checkpoint_") && name.ends_with(".enc") {
                // Extract version number
                let version_str = name.strip_prefix("checkpoint_").and_then(|s| s.strip_suffix(".enc")).unwrap_or("0");

                if let Ok(version) = version_str.parse::<u32>() {
                    versions.push(version);
                }
            }
        }

        versions.sort();
        Ok(versions)
    }

    pub async fn delete_model(&self, model_id: &str) -> Result<(), StorageError> {
        let model_path = self.model_path(model_id);

        if Path::new(&model_path).exists() {
            fs::remove_dir_all(&model_path).await?;
        }

        Ok(())
    }

    pub async fn model_exists(&self, model_id: &str) -> bool {
        let model_path = self.model_path(model_id);
        Path::new(&format!("{}/weights.enc", model_path)).exists() || Path::new(&format!("{}/model.enc", model_path)).exists()
    }

    fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, StorageError> {
        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key).map_err(|e| StorageError::EncryptionError(e.to_string()))?;

        let mut rng = rand::thread_rng();
        let nonce_bytes: [u8; 12] = rng.gen();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher.encrypt(nonce, data).map_err(|e| StorageError::EncryptionError(e.to_string()))?;

        // Prepend nonce to ciphertext
        let mut result = nonce_bytes.to_vec();
        result.extend_from_slice(&ciphertext);

        Ok(result)
    }

    fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, StorageError> {
        if data.len() < 12 {
            return Err(StorageError::DecryptionError("Data too short".to_string()));
        }

        let (nonce_bytes, ciphertext) = data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let cipher = Aes256Gcm::new_from_slice(&self.encryption_key).map_err(|e| StorageError::DecryptionError(e.to_string()))?;

        let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|e| StorageError::DecryptionError(e.to_string()))?;

        Ok(plaintext)
    }

    fn compute_key_hash(&self) -> Vec<u8> {
        let mut hasher = Sha256::new();
        hasher.update(self.encryption_key);
        hasher.finalize().to_vec()
    }

    pub fn generate_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        let mut rng = rand::thread_rng();
        rng.fill(&mut key);
        key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_encryption_decryption() {
        let key = ModelStorage::generate_key();
        let storage = ModelStorage::new("/tmp/test_models".to_string(), key);

        let data = b"test model data".to_vec();
        let encrypted = storage.encrypt(&data).unwrap();
        let decrypted = storage.decrypt(&encrypted).unwrap();

        assert_eq!(data, decrypted);
    }

    #[tokio::test]
    async fn test_store_load_model() {
        let key = ModelStorage::generate_key();
        let storage = ModelStorage::new("/tmp/test_models".to_string(), key);

        let data = b"model weights".to_vec();
        let _path = storage.store_model("test_model", &data).await.unwrap();

        let loaded = storage.load_model("test_model").await.unwrap();
        assert_eq!(data, loaded);

        // Cleanup
        let _ = storage.delete_model("test_model").await;
    }

    #[tokio::test]
    async fn test_store_load_model_files() {
        let key = ModelStorage::generate_key();
        let storage = ModelStorage::new("/tmp/test_models_files".to_string(), key);

        let files = RawModelFiles { config: b"{}".to_vec(), tokenizer: b"[]".to_vec(), weights: b"model weights".to_vec() };
        storage.store_model_files("test_model_files", &files).await.unwrap();

        let loaded = storage.load_model_files("test_model_files").await.unwrap();
        assert_eq!(files.config, loaded.config);
        assert_eq!(files.tokenizer, loaded.tokenizer);
        assert_eq!(files.weights, loaded.weights);

        // Cleanup
        let _ = storage.delete_model("test_model_files").await;
    }
}
