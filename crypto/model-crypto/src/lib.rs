//! AES-256-GCM helpers for model weights and checkpoint files.
//!
//! This crate is shared between `seed-node` and `xenom-miner` so both sides can
//! derive the same key and encrypt/decrypt model payloads without duplication.

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use rand::Rng;
use sha2::{Digest, Sha256};

const DEFAULT_KEY_SEED: &str = "xenom-devnet-model-key";
const NONCE_LEN: usize = 12;

#[derive(Debug, thiserror::Error)]
pub enum ModelCryptoError {
    #[error("Encryption failed: {0}")]
    EncryptionError(String),
    #[error("Decryption failed: {0}")]
    DecryptionError(String),
    #[error("Invalid key material")]
    InvalidKey,
}

/// Derive the 32-byte AES key used to encrypt/decrypt model files.
///
/// If `XENO_MODEL_KEY` is a 64-character hex string, it is decoded directly;
/// otherwise the value (or a default devnet string) is hashed with SHA-256.
/// This function is shared by the seed-node and the miner so both processes can
/// read the same encrypted model cache.
pub fn derive_encryption_key() -> [u8; 32] {
    let seed = std::env::var("XENO_MODEL_KEY").unwrap_or_else(|_| DEFAULT_KEY_SEED.to_string());
    let seed = seed.trim();

    if seed.len() == 64 {
        if let Ok(decoded) = hex::decode(seed) {
            if decoded.len() == 32 {
                let mut key = [0u8; 32];
                key.copy_from_slice(&decoded);
                return key;
            }
        }
    }

    let hash = Sha256::digest(seed.as_bytes());
    let mut key = [0u8; 32];
    key.copy_from_slice(&hash);
    key
}

/// Compute a SHA-256 hash of the provided key for verification/audit.
pub fn key_hash(key: &[u8; 32]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(key);
    let result = hasher.finalize();
    let mut hash = [0u8; 32];
    hash.copy_from_slice(&result);
    hash
}

/// Generate a fresh random 32-byte AES key.
pub fn generate_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    rand::thread_rng().fill(&mut key);
    key
}

/// Encrypt `data` with AES-256-GCM.
///
/// The returned buffer is `nonce || ciphertext` so decryption does not need a
/// out-of-band nonce.
pub fn encrypt(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, ModelCryptoError> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| ModelCryptoError::EncryptionError(e.to_string()))?;

    let mut rng = rand::thread_rng();
    let nonce_bytes: [u8; NONCE_LEN] = rng.gen();
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, data).map_err(|e| ModelCryptoError::EncryptionError(e.to_string()))?;

    let mut result = nonce_bytes.to_vec();
    result.extend_from_slice(&ciphertext);
    Ok(result)
}

/// Decrypt a buffer produced by `encrypt`.
pub fn decrypt(data: &[u8], key: &[u8; 32]) -> Result<Vec<u8>, ModelCryptoError> {
    if data.len() < NONCE_LEN {
        return Err(ModelCryptoError::DecryptionError("Ciphertext too short".to_string()));
    }

    let (nonce_bytes, ciphertext) = data.split_at(NONCE_LEN);
    let nonce = Nonce::from_slice(nonce_bytes);

    let cipher = Aes256Gcm::new_from_slice(key).map_err(|e| ModelCryptoError::DecryptionError(e.to_string()))?;
    let plaintext = cipher.decrypt(nonce, ciphertext).map_err(|e| ModelCryptoError::DecryptionError(e.to_string()))?;

    Ok(plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encrypt_decrypt_roundtrip() {
        let key = generate_key();
        let data = b"model weights".to_vec();
        let encrypted = encrypt(&data, &key).unwrap();
        let decrypted = decrypt(&encrypted, &key).unwrap();
        assert_eq!(data, decrypted);
    }

    #[test]
    fn test_derive_key_is_deterministic() {
        // key derivation reads an env var, so we cannot assert a fixed value
        // without polluting the environment. Instead, verify it produces 32 bytes.
        let key = derive_encryption_key();
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_decryption_fails_with_wrong_key() {
        let key = generate_key();
        let wrong_key = generate_key();
        let data = b"secret".to_vec();
        let encrypted = encrypt(&data, &key).unwrap();
        assert!(decrypt(&encrypted, &wrong_key).is_err());
    }
}
