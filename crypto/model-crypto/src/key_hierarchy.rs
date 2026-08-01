//! Per-model key hierarchy using HKDF-SHA256.
//!
//! Hierarchy (per PRD-001):
//!
//! ```text
//! MK_m     = master key for model m (only the orchestrator holds this)
//! EK_m     = HKDF-SHA256(MK_m, "enc",   model_id || version)
//! SK_{m,s} = HKDF-SHA256(EK_m, "session", miner_public_key || nonce)
//! KAuth_m  = HKDF-SHA256(MK_m, "auth",  model_id)
//! ```
//!
//! All key material is wrapped in [`ModelSecret`], which zeroizes on drop.

use hkdf::Hkdf;
use rand::Rng;
use sha2::{Digest, Sha256};
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::ModelCryptoError;

const KEY_LEN: usize = 32;

/// A 32-byte secret with explicit zeroization on drop.
#[derive(Clone, Debug, Zeroize, ZeroizeOnDrop)]
pub struct ModelSecret(pub [u8; 32]);

impl ModelSecret {
    /// Create a secret from raw bytes.
    pub fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Generate a fresh random 32-byte master key.
    pub fn random() -> Self {
        let mut key = [0u8; 32];
        rand::thread_rng().fill(&mut key);
        Self(key)
    }

    /// Return a SHA-256 hash of the secret for on-chain audit.
    pub fn hash(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(self.0);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Derive a child key using HKDF-SHA256 with the given context and salt.
    pub fn derive(&self, context: &[u8], salt: &[u8]) -> Result<ModelSecret, ModelCryptoError> {
        let hk = Hkdf::<Sha256>::new(Some(salt), &self.0);
        let mut okm = [0u8; KEY_LEN];
        hk.expand(context, &mut okm).map_err(|e| ModelCryptoError::InvalidKey(format!("HKDF expansion failed: {}", e)))?;
        Ok(ModelSecret(okm))
    }

    /// Encrypt a plaintext artifact with this secret using AES-256-GCM.
    pub fn encrypt(&self, data: &[u8]) -> Result<Vec<u8>, crate::ModelCryptoError> {
        crate::encrypt(data, &self.0)
    }

    /// Decrypt a ciphertext produced by [`Self::encrypt`].
    pub fn decrypt(&self, data: &[u8]) -> Result<Vec<u8>, crate::ModelCryptoError> {
        crate::decrypt(data, &self.0)
    }
}

/// Key hierarchy for a single model.
#[derive(Clone, Debug)]
pub struct ModelKeyHierarchy {
    /// Master key `MK_m`. Only the orchestrator should hold this.
    pub master: ModelSecret,
    pub model_id: String,
    pub version: u64,
}

impl ModelKeyHierarchy {
    /// Create a new hierarchy from an existing master key.
    pub fn new(master: ModelSecret, model_id: impl Into<String>, version: u64) -> Self {
        Self { master, model_id: model_id.into(), version }
    }

    /// Generate a fresh hierarchy with a random master key.
    pub fn random(model_id: impl Into<String>, version: u64) -> Self {
        Self::new(ModelSecret::random(), model_id, version)
    }

    /// Derive the per-model encryption key `EK_m`.
    ///
    /// `EK_m = HKDF-SHA256(MK_m, "enc", model_id || version)`
    pub fn encryption_key(&self) -> Result<ModelSecret, ModelCryptoError> {
        let salt = build_salt(&self.model_id, self.version);
        self.master.derive(b"enc", &salt)
    }

    /// Derive the model artifact signing key `KAuth_m`.
    ///
    /// `KAuth_m = HKDF-SHA256(MK_m, "auth", model_id)`
    pub fn auth_key(&self) -> Result<ModelSecret, ModelCryptoError> {
        self.master.derive(b"auth", self.model_id.as_bytes())
    }

    /// Derive an ephemeral per-session key `SK_{m,s}`.
    ///
    /// `SK_{m,s} = HKDF-SHA256(EK_m, "session", miner_public_key || nonce)`
    pub fn session_key(&self, miner_public_key: &[u8], nonce: &[u8]) -> Result<ModelSecret, ModelCryptoError> {
        let encryption_key = self.encryption_key()?;
        let mut salt = miner_public_key.to_vec();
        salt.extend_from_slice(nonce);
        encryption_key.derive(b"session", &salt)
    }
}

fn build_salt(model_id: &str, version: u64) -> Vec<u8> {
    let mut salt = model_id.as_bytes().to_vec();
    salt.extend_from_slice(&version.to_be_bytes());
    salt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_is_deterministic() {
        let master = ModelSecret::random();
        let h1 = ModelKeyHierarchy::new(master.clone(), "xeno/mgm-1", 1);
        let h2 = ModelKeyHierarchy::new(master, "xeno/mgm-1", 1);

        let ek1 = h1.encryption_key().unwrap();
        let ek2 = h2.encryption_key().unwrap();
        assert_eq!(ek1.0, ek2.0);

        let ak1 = h1.auth_key().unwrap();
        let ak2 = h2.auth_key().unwrap();
        assert_eq!(ak1.0, ak2.0);
    }

    #[test]
    fn test_different_models_different_keys() {
        let master = ModelSecret::random();
        let h1 = ModelKeyHierarchy::new(master.clone(), "xeno/mgm-1", 1);
        let h2 = ModelKeyHierarchy::new(master, "xeno/dnabert2", 1);

        let ek1 = h1.encryption_key().unwrap();
        let ek2 = h2.encryption_key().unwrap();
        assert_ne!(ek1.0, ek2.0);
    }

    #[test]
    fn test_session_key_uses_miner_pubkey() {
        let h = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let pk = [1u8; 33];
        let nonce = [2u8; 12];

        let sk1 = h.session_key(&pk, &nonce).unwrap();
        let sk2 = h.session_key(&pk, &nonce).unwrap();
        assert_eq!(sk1.0, sk2.0);

        let mut other_pk = pk;
        other_pk[0] = 0;
        let sk3 = h.session_key(&other_pk, &nonce).unwrap();
        assert_ne!(sk1.0, sk3.0);
    }

    #[test]
    fn test_secret_zeroizes() {
        let mut s = ModelSecret::random();
        s.zeroize();
        assert_eq!(s.0, [0u8; 32]);
    }

    #[test]
    fn test_encrypt_decrypt_with_derived_key() {
        let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let encryption_key = hierarchy.encryption_key().unwrap();

        let plaintext = b"artifact data".to_vec();
        let ciphertext = encryption_key.encrypt(&plaintext).unwrap();
        assert_ne!(plaintext, ciphertext);

        let decrypted = encryption_key.decrypt(&ciphertext).unwrap();
        assert_eq!(plaintext, decrypted);
    }
}
