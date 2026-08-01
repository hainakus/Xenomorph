//! Encrypted session-key exchange for LoRA/artifact distribution.
//!
//! This implements a one-way Diffie-Hellman key exchange:
//! - The orchestrator generates an ephemeral secp256k1 key pair.
//! - It computes `shared_secret = ECDH(ephemeral_secret, miner_public_key)`.
//! - It derives an AES-256-GCM session key from that secret and a nonce.
//! - The miner computes the same `shared_secret = ECDH(miner_secret, ephemeral_public_key)`
//!   and decrypts the artifact.
//!
//! This is intentionally simpler than full ECIES: the encrypted payload is the
//! AES-GCM output of the artifact, and the session key is the per-artifact
//! ephemeral key.

use hkdf::Hkdf;
use secp256k1::ecdh::SharedSecret;
use secp256k1::{PublicKey, SecretKey};
use sha2::Sha256;

use crate::key_hierarchy::ModelSecret;
use crate::ModelCryptoError;

const SESSION_CONTEXT: &[u8] = b"xenom-lora-session-v1";
const NONCE_LEN: usize = 12;

/// Generate a fresh ephemeral secp256k1 key pair.
///
/// The orchestrator uses this to encrypt a single `TrainingArtifact`.
pub fn generate_ephemeral_keypair() -> (SecretKey, PublicKey) {
    let secp = secp256k1::Secp256k1::new();
    let secret = SecretKey::new(&mut rand::thread_rng());
    let public = PublicKey::from_secret_key(&secp, &secret);
    (secret, public)
}

/// Derive a session key from an ECDH shared secret and a nonce.
///
/// `session_key = HKDF-SHA256(shared_secret, "xenom-lora-session-v1", nonce)`
pub fn derive_session_key(shared_secret: &SharedSecret, nonce: &[u8; NONCE_LEN]) -> Result<ModelSecret, ModelCryptoError> {
    let secret = shared_secret.secret_bytes();
    let hk = Hkdf::<Sha256>::new(Some(nonce), &secret);
    let mut okm = [0u8; 32];
    hk.expand(SESSION_CONTEXT, &mut okm).map_err(|e| ModelCryptoError::InvalidKey(format!("HKDF session key failed: {}", e)))?;
    Ok(ModelSecret(okm))
}

/// Compute the shared secret on the orchestrator side.
pub fn orchestrator_shared_secret(ephemeral_secret: &SecretKey, miner_public_key: &PublicKey) -> SharedSecret {
    SharedSecret::new(miner_public_key, ephemeral_secret)
}

/// Compute the shared secret on the miner side.
pub fn miner_shared_secret(miner_secret: &SecretKey, ephemeral_public_key: &PublicKey) -> SharedSecret {
    SharedSecret::new(ephemeral_public_key, miner_secret)
}

/// Encrypt an artifact with a session key and a random nonce.
///
/// Returns `(ciphertext, nonce)` where `ciphertext` is `nonce || aes_gcm(...)`.
/// The returned `nonce` is reused as the HKDF salt so the miner can reconstruct
/// the session key.
pub fn encrypt_artifact(plaintext: &[u8], session_key: &ModelSecret) -> Result<Vec<u8>, ModelCryptoError> {
    session_key.encrypt(plaintext)
}

/// Decrypt an artifact with a session key.
pub fn decrypt_artifact(ciphertext: &[u8], session_key: &ModelSecret) -> Result<Vec<u8>, ModelCryptoError> {
    session_key.decrypt(ciphertext)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_key_roundtrip() {
        let (ephemeral_secret, ephemeral_public) = generate_ephemeral_keypair();
        let miner_secret = SecretKey::new(&mut rand::thread_rng());
        let secp = secp256k1::Secp256k1::new();
        let miner_public = PublicKey::from_secret_key(&secp, &miner_secret);

        let nonce = [1u8; NONCE_LEN];

        let shared_orchestrator = orchestrator_shared_secret(&ephemeral_secret, &miner_public);
        let sk_orchestrator = derive_session_key(&shared_orchestrator, &nonce).unwrap();

        let shared_miner = miner_shared_secret(&miner_secret, &ephemeral_public);
        let sk_miner = derive_session_key(&shared_miner, &nonce).unwrap();

        assert_eq!(sk_orchestrator.0, sk_miner.0);

        let plaintext = b"lm head artifact".to_vec();
        let ciphertext = encrypt_artifact(&plaintext, &sk_orchestrator).unwrap();
        let decrypted = decrypt_artifact(&ciphertext, &sk_miner).unwrap();

        assert_eq!(plaintext, decrypted);
    }
}
