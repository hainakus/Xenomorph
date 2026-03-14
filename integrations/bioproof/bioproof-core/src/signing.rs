use secp256k1::{ecdsa::Signature, Message, PublicKey, Secp256k1, SecretKey, rand::rngs::OsRng};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SigningError {
    #[error("invalid private key: {0}")]
    InvalidKey(#[from] secp256k1::Error),
    #[error("invalid hex: {0}")]
    Hex(#[from] hex::FromHexError),
    #[error("invalid signature hex")]
    InvalidSig,
}

/// Ergonomic wrapper around a secp256k1 keypair for BioProof signing.
pub struct BioProofKeypair {
    secret: SecretKey,
    public: PublicKey,
}

impl BioProofKeypair {
    /// Generate a fresh random keypair using OS entropy.
    pub fn generate() -> Self {
        let secp   = Secp256k1::new();
        let secret = SecretKey::new(&mut OsRng);
        let public = PublicKey::from_secret_key(&secp, &secret);
        Self { secret, public }
    }

    /// Hex-encoded 32-byte private key.
    pub fn privkey_hex(&self) -> String {
        hex::encode(self.secret.secret_bytes())
    }

    /// Load from a 32-byte private key hex string.
    pub fn from_hex(hex_str: &str) -> Result<Self, SigningError> {
        let bytes = hex::decode(hex_str)?;
        let secret = SecretKey::from_slice(&bytes)?;
        let secp   = Secp256k1::signing_only();
        let public = PublicKey::from_secret_key(&secp, &secret);
        Ok(Self { secret, public })
    }

    /// Hex-encoded 33-byte compressed public key.
    pub fn pubkey_hex(&self) -> String {
        hex::encode(self.public.serialize())
    }

    /// Sign the 32-byte digest produced by `Manifest::hash_bytes()`.
    pub fn sign(&self, digest: &[u8; 32]) -> String {
        let secp = Secp256k1::signing_only();
        let msg  = Message::from_digest(*digest);
        let sig  = secp.sign_ecdsa(&msg, &self.secret);
        hex::encode(sig.serialize_der())
    }
}

/// Sign a manifest hash digest; returns DER signature as hex.
pub fn sign_manifest(digest: &[u8; 32], private_key_hex: &str) -> Result<String, SigningError> {
    let kp = BioProofKeypair::from_hex(private_key_hex)?;
    Ok(kp.sign(digest))
}

/// Verify a DER signature (hex) against a manifest hash digest and pubkey (hex).
pub fn verify_manifest_sig(
    digest:     &[u8; 32],
    sig_hex:    &str,
    pubkey_hex: &str,
) -> Result<bool, SigningError> {
    let sig_bytes  = hex::decode(sig_hex)?;
    let pub_bytes  = hex::decode(pubkey_hex)?;
    let secp       = Secp256k1::verification_only();
    let msg        = Message::from_digest(*digest);
    let sig        = Signature::from_der(&sig_bytes).map_err(|_| SigningError::InvalidSig)?;
    let pubkey     = PublicKey::from_slice(&pub_bytes)?;
    Ok(secp.verify_ecdsa(&msg, &sig, &pubkey).is_ok())
}
