use anyhow::{Context, Result, bail};
use bip39::{Language, Mnemonic};
use rand::RngCore;
use secp256k1::{PublicKey, SecretKey, Message};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

use crate::rpc::messages::TrainingBlock;

const WALLET_FILE_NAME: &str = "wallet.enc";
const SALT_BYTES: usize = 16;

/// Manages the miner's secp256k1 key pair, derived from a BIP39 mnemonic.
#[derive(Debug, Clone)]
pub struct WalletManager {
    secret_key: SecretKey,
    public_key: PublicKey,
    address: String,
}

impl WalletManager {
    /// Create a new random wallet and persist it encrypted under `data_dir`.
    pub fn create_new(data_dir: &Path, password: &str) -> Result<Self> {
        let mut entropy = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut entropy);

        let mnemonic = Mnemonic::from_entropy_in(Language::English, &entropy)
            .with_context(|| "Failed to generate BIP39 mnemonic")?;
        let phrase = mnemonic.to_string();

        Self::from_mnemonic(data_dir, &phrase, password)
    }

    /// Load an existing wallet or create a new one if none exists.
    pub fn load_or_create(data_dir: &Path, password: &str) -> Result<Self> {
        let wallet_path = wallet_path(data_dir);

        if wallet_path.exists() {
            let encrypted = fs::read(&wallet_path)
                .with_context(|| format!("Failed to read wallet at {:?}", wallet_path))?;
            let phrase = decrypt_with_password(&encrypted, password)
                .with_context(|| "Failed to decrypt wallet (wrong password or corrupt data)")?;
            Self::from_mnemonic(data_dir, &phrase, password)
        } else {
            info!("No wallet found at {:?}; creating a new one", wallet_path);
            Self::create_new(data_dir, password)
        }
    }

    /// Restore a wallet from a BIP39 mnemonic phrase and persist it.
    pub fn from_mnemonic(data_dir: &Path, phrase: &str, password: &str) -> Result<Self> {
        let mnemonic = Mnemonic::parse_in(Language::English, phrase)
            .with_context(|| "Invalid BIP39 mnemonic phrase")?;
        let seed = mnemonic.to_seed("");

        let secret_key = SecretKey::from_slice(&seed[..32])
            .with_context(|| "Failed to derive secret key from seed")?;
        let public_key = PublicKey::from_secret_key_global(&secret_key);
        let address = derive_address(&public_key);

        let manager = Self { secret_key, public_key, address };
        manager.save(data_dir, password, phrase)?;
        Ok(manager)
    }

    /// Sign a `TrainingBlock`, writing the 64-byte ECDSA signature into it.
    pub fn sign_block(&self, block: &mut TrainingBlock) -> Result<()> {
        let message = block_signing_message(block);
        let digest = Sha256::digest(&message);
        let secp_message = Message::from_digest_slice(&digest)
            .with_context(|| "Failed to build secp256k1 message")?;
        let signature = self.secret_key.sign_ecdsa(secp_message);
        block.signature = signature.serialize_compact();
        Ok(())
    }

    /// Verify the signature stored inside a `TrainingBlock`.
    pub fn verify_signature(&self, block: &TrainingBlock) -> Result<bool> {
        let message = block_signing_message(block);
        let digest = Sha256::digest(&message);
        let secp_message = Message::from_digest_slice(&digest)
            .with_context(|| "Failed to build secp256k1 message")?;
        let signature = secp256k1::ecdsa::Signature::from_compact(&block.signature)
            .with_context(|| "Invalid compact signature")?;
        Ok(signature.verify(&secp_message, &self.public_key).is_ok())
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    fn save(&self, data_dir: &Path, password: &str, phrase: &str) -> Result<()> {
        fs::create_dir_all(data_dir)
            .with_context(|| format!("Failed to create wallet directory {:?}", data_dir))?;
        let encrypted = encrypt_with_password(phrase.as_bytes(), password);
        let path = wallet_path(data_dir);
        fs::write(&path, encrypted)
            .with_context(|| format!("Failed to write wallet to {:?}", path))?;
        info!("Saved encrypted wallet to {:?}", path);
        Ok(())
    }
}

fn derive_address(public_key: &PublicKey) -> String {
    let hash = blake3::hash(public_key.serialize().as_ref());
    format!("xnom:{}", hex::encode(hash.as_bytes()))
}

fn wallet_path(data_dir: &Path) -> PathBuf {
    data_dir.join(WALLET_FILE_NAME)
}

fn block_signing_message(block: &TrainingBlock) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&block.header.prev_block_hash);
    bytes.extend_from_slice(&block.header.block_number.to_le_bytes());
    bytes.extend_from_slice(&block.header.timestamp.to_le_bytes());
    bytes.extend_from_slice(&block.header.merkle_root);
    bytes.extend_from_slice(&block.header.difficulty);
    bytes.extend_from_slice(&block.header.nonce.to_le_bytes());
    bytes.extend_from_slice(&block.training_proof.gradients_commitment);
    bytes.extend_from_slice(block.miner_address.as_bytes());
    bytes
}

/// Simple stream-cipher-like encryption using Blake3 as a keystream.
fn encrypt_with_password(plaintext: &[u8], password: &str) -> Vec<u8> {
    let mut salt = [0u8; SALT_BYTES];
    rand::thread_rng().fill_bytes(&mut salt);

    let mut ciphertext = Vec::with_capacity(SALT_BYTES + plaintext.len());
    ciphertext.extend_from_slice(&salt);

    let mut keystream = blake3::Hasher::new();
    keystream.update(password.as_bytes());
    keystream.update(&salt);

    for (i, byte) in plaintext.iter().enumerate() {
        let counter = (i as u64).to_le_bytes();
        let mut h = keystream.clone();
        h.update(&counter);
        let key_byte = h.finalize().as_bytes()[0];
        ciphertext.push(byte ^ key_byte);
    }

    ciphertext
}

fn decrypt_with_password(ciphertext: &[u8], password: &str) -> Result<String> {
    if ciphertext.len() < SALT_BYTES {
        bail!("Wallet file is too short to contain a salt");
    }
    let salt = &ciphertext[..SALT_BYTES];
    let body = &ciphertext[SALT_BYTES..];

    let mut plaintext = Vec::with_capacity(body.len());

    let mut keystream = blake3::Hasher::new();
    keystream.update(password.as_bytes());
    keystream.update(salt);

    for (i, byte) in body.iter().enumerate() {
        let counter = (i as u64).to_le_bytes();
        let mut h = keystream.clone();
        h.update(&counter);
        let key_byte = h.finalize().as_bytes()[0];
        plaintext.push(byte ^ key_byte);
    }

    String::from_utf8(plaintext)
        .with_context(|| "Decrypted wallet is not valid UTF-8")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rpc::messages::{BlockHeader, TrainingProof};

    fn dummy_block() -> TrainingBlock {
        TrainingBlock {
            header: BlockHeader {
                prev_block_hash: [0u8; 32],
                block_number: 1,
                timestamp: 0,
                merkle_root: [1u8; 32],
                difficulty: [2u8; 32],
                nonce: 0,
            },
            training_proof: TrainingProof {
                base_checkpoint: [3u8; 32],
                loss_before: 2.45,
                loss_after: 2.41,
                gradients_commitment: [4u8; 32],
                zk_proof: vec![0u8; 32],
                batch_indices: vec![0, 1, 2],
                compute_time_ms: 100,
            },
            miner_address: "xnom:test".to_string(),
            timestamp: 0,
            signature: [0u8; 64],
        }
    }

    #[test]
    fn test_wallet_creation_and_signing() {
        let tmp = tempfile::tempdir().unwrap();
        let wallet = WalletManager::create_new(tmp.path(), "password").unwrap();

        assert!(wallet.address().starts_with("xnom:"));

        let mut block = dummy_block();
        wallet.sign_block(&mut block).unwrap();
        assert!(wallet.verify_signature(&block).unwrap());

        let loaded = WalletManager::load_or_create(tmp.path(), "password").unwrap();
        assert_eq!(loaded.address(), wallet.address());
    }

    #[test]
    fn test_wrong_password_fails_to_decrypt() {
        let tmp = tempfile::tempdir().unwrap();
        WalletManager::create_new(tmp.path(), "right").unwrap();

        assert!(WalletManager::load_or_create(tmp.path(), "wrong").is_err());
    }

    #[test]
    fn test_mnemonic_restore() {
        let phrase = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";
        let tmp = tempfile::tempdir().unwrap();
        let wallet1 = WalletManager::from_mnemonic(tmp.path(), phrase, "pw").unwrap();

        // Recreating with the same mnemonic should produce the same address.
        let wallet2 = WalletManager::from_mnemonic(tmp.path(), phrase, "pw").unwrap();
        assert_eq!(wallet1.address(), wallet2.address());
    }
}
