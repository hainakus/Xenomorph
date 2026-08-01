//! Artifact signing and verification using secp256k1.
//!
//! The orchestrator signs `artifact_hash || base_hash` with `KAuth_m`.
//! The on-chain `ModelRegistry` stores the corresponding public key so miners
//! can verify the signature before loading any artifact.

use secp256k1::{Message, PublicKey, Secp256k1, SecretKey, SignOnly, VerifyOnly};
use sha2::{Digest, Sha256};

use crate::key_hierarchy::ModelSecret;
use crate::ModelCryptoError;

const SIGNING_DOMAIN: &[u8] = b"xenom-model-artifact-v1";

/// Artifact signature and the public key used to verify it.
#[derive(Clone, Debug)]
pub struct ArtifactSignature {
    pub signature: [u8; 64],
    pub public_key: [u8; 33],
}

/// A signer derived from a model authorization key (`KAuth_m`).
pub struct ArtifactSigner {
    secret_key: SecretKey,
    public_key: PublicKey,
    secp: Secp256k1<SignOnly>,
}

impl ArtifactSigner {
    /// Create a signer from the model authorization secret.
    pub fn from_auth_key(auth_key: &ModelSecret) -> Result<Self, ModelCryptoError> {
        let secp = Secp256k1::signing_only();
        let secret_key =
            SecretKey::from_slice(&auth_key.0).map_err(|e| ModelCryptoError::InvalidKey(format!("Invalid auth key: {}", e)))?;
        let public_key = PublicKey::from_secret_key(&secp, &secret_key);
        Ok(Self { secret_key, public_key, secp })
    }

    /// Sign the artifact hash (and optional base hash) using a deterministic
    /// message construction.
    pub fn sign(&self, artifact_hash: &[u8; 32], base_hash: Option<&[u8; 32]>) -> Result<ArtifactSignature, ModelCryptoError> {
        let message = build_message(artifact_hash, base_hash);
        let message = Message::from_digest(message);
        let signature = self.secp.sign_ecdsa(&message, &self.secret_key);
        let mut sig_bytes = [0u8; 64];
        sig_bytes.copy_from_slice(&signature.serialize_compact());

        let mut public_key = [0u8; 33];
        public_key.copy_from_slice(&self.public_key.serialize());

        Ok(ArtifactSignature { signature: sig_bytes, public_key })
    }

    /// Return the compressed public key.
    pub fn public_key(&self) -> [u8; 33] {
        self.public_key.serialize()
    }
}

/// Verifier for artifact signatures.
pub struct ArtifactVerifier {
    public_key: PublicKey,
    secp: Secp256k1<VerifyOnly>,
}

impl ArtifactVerifier {
    /// Create a verifier from a compressed (33-byte) public key.
    pub fn from_public_key(public_key: &[u8; 33]) -> Result<Self, ModelCryptoError> {
        let secp = Secp256k1::verification_only();
        let public_key =
            PublicKey::from_slice(public_key).map_err(|e| ModelCryptoError::InvalidKey(format!("Invalid public key: {}", e)))?;
        Ok(Self { public_key, secp })
    }

    /// Verify an artifact signature.
    pub fn verify(
        &self,
        artifact_hash: &[u8; 32],
        base_hash: Option<&[u8; 32]>,
        signature: &[u8; 64],
    ) -> Result<(), ModelCryptoError> {
        let message = build_message(artifact_hash, base_hash);
        let message = Message::from_digest(message);
        let signature = secp256k1::ecdsa::Signature::from_compact(signature)
            .map_err(|e| ModelCryptoError::InvalidKey(format!("Invalid signature: {}", e)))?;
        self.secp
            .verify_ecdsa(&message, &signature, &self.public_key)
            .map_err(|e| ModelCryptoError::InvalidKey(format!("Signature verification failed: {}", e)))
    }

    /// Verify an [`ArtifactSignature`] struct.
    pub fn verify_signature(
        &self,
        artifact_hash: &[u8; 32],
        base_hash: Option<&[u8; 32]>,
        sig: &ArtifactSignature,
    ) -> Result<(), ModelCryptoError> {
        self.verify(artifact_hash, base_hash, &sig.signature)
    }
}

fn build_message(artifact_hash: &[u8; 32], base_hash: Option<&[u8; 32]>) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(SIGNING_DOMAIN);
    hasher.update(artifact_hash);
    if let Some(base_hash) = base_hash {
        hasher.update(base_hash);
    }
    let result = hasher.finalize();
    let mut message = [0u8; 32];
    message.copy_from_slice(&result);
    message
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::key_hierarchy::{ModelKeyHierarchy, ModelSecret};

    #[test]
    fn test_sign_and_verify_roundtrip() {
        let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let auth_key = hierarchy.auth_key().unwrap();
        let signer = ArtifactSigner::from_auth_key(&auth_key).unwrap();

        let artifact_hash = [0xabu8; 32];
        let base_hash = [0xcdu8; 32];

        let sig = signer.sign(&artifact_hash, Some(&base_hash)).unwrap();
        let verifier = ArtifactVerifier::from_public_key(&sig.public_key).unwrap();
        verifier.verify(&artifact_hash, Some(&base_hash), &sig.signature).unwrap();
    }

    #[test]
    fn test_signature_tamper_detection() {
        let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let auth_key = hierarchy.auth_key().unwrap();
        let signer = ArtifactSigner::from_auth_key(&auth_key).unwrap();

        let artifact_hash = [0xabu8; 32];
        let base_hash = [0xcdu8; 32];

        let sig = signer.sign(&artifact_hash, Some(&base_hash)).unwrap();
        let verifier = ArtifactVerifier::from_public_key(&sig.public_key).unwrap();

        let mut tampered = artifact_hash;
        tampered[0] = 0xff;
        assert!(verifier.verify(&tampered, Some(&base_hash), &sig.signature).is_err());
    }

    #[test]
    fn test_signature_wrong_key() {
        let hierarchy1 = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let hierarchy2 = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let auth_key1 = hierarchy1.auth_key().unwrap();
        let auth_key2 = hierarchy2.auth_key().unwrap();

        let signer = ArtifactSigner::from_auth_key(&auth_key1).unwrap();
        let other = ArtifactSigner::from_auth_key(&auth_key2).unwrap();

        let artifact_hash = [0xabu8; 32];
        let sig = signer.sign(&artifact_hash, None).unwrap();

        let verifier = ArtifactVerifier::from_public_key(&other.public_key()).unwrap();
        assert!(verifier.verify(&artifact_hash, None, &sig.signature).is_err());
    }

    #[test]
    fn test_secret_key_from_zero_fails() {
        let zero = ModelSecret::new([0u8; 32]);
        assert!(ArtifactSigner::from_auth_key(&zero).is_err());
    }
}
