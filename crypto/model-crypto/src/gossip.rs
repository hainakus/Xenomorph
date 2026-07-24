//! P2P gossip identity and checkpoint metadata registry.
//!
//! This module is used by `xenom-node`, `seed-node` and `xenom-miner` to sign
//! and verify `(model_id, weights_hash, cid)` announcements propagated over the
//! Kaspa P2P layer.

use bip39::{Language, Mnemonic};
use kaspa_addresses::{Address, Prefix, Version};
use kaspa_consensus_core::network::NetworkType;
use secp256k1::{Message, PublicKey, SecretKey};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    time::{Duration, Instant},
};

use crate::ModelCryptoError;

const GOSSIP_MNEMONIC_ENV: &str = "XENO_GOSSIP_MNEMONIC";
const GOSSIP_KEY_ENV: &str = "XENO_GOSSIP_KEY";
const DEFAULT_GOSSIP_TTL: Duration = Duration::from_secs(300);

/// A signed checkpoint announcement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Announcement {
    pub model_id: String,
    pub weights_hash: [u8; 32],
    pub cid: [u8; 32],
    pub timestamp: u64,
    pub is_genome: bool,
    pub node_address: String,
    pub public_key: [u8; 33],
    pub listen_addr: Option<SocketAddr>,
    pub signature: [u8; 64],
}

impl Announcement {
    /// Build the canonical byte string that is hashed and signed.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        let model_id_bytes = self.model_id.as_bytes();
        buf.extend_from_slice(&(model_id_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(model_id_bytes);

        buf.extend_from_slice(&self.weights_hash);
        buf.extend_from_slice(&self.cid);
        buf.extend_from_slice(&self.timestamp.to_le_bytes());
        buf.extend_from_slice(&[self.is_genome as u8]);

        let node_address_bytes = self.node_address.as_bytes();
        buf.extend_from_slice(&(node_address_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(node_address_bytes);

        buf.extend_from_slice(&self.public_key);

        if let Some(addr) = self.listen_addr {
            let ip_bytes = match addr.ip() {
                IpAddr::V4(ip) => ip.octets().to_vec(),
                IpAddr::V6(ip) => ip.octets().to_vec(),
            };
            buf.push(1);
            buf.extend_from_slice(&(ip_bytes.len() as u8).to_le_bytes());
            buf.extend_from_slice(&ip_bytes);
            buf.extend_from_slice(&addr.port().to_le_bytes());
        } else {
            buf.push(0);
        }

        buf
    }

    /// Compute the SHA-256 digest used for signing/verification.
    pub fn digest(&self) -> [u8; 32] {
        let bytes = self.signing_bytes();
        let result = Sha256::digest(&bytes);
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }
}

/// A gossip identity: secp256k1 key pair plus the derived Xenom/Kaspa address.
#[derive(Debug, Clone)]
pub struct GossipIdentity {
    secret_key: SecretKey,
    public_key: PublicKey,
    address: String,
    network_type: NetworkType,
}

impl GossipIdentity {
    /// Create an identity from a BIP39 mnemonic phrase.
    pub fn from_mnemonic(phrase: &str, network_type: NetworkType) -> Result<Self, ModelCryptoError> {
        let mnemonic = Mnemonic::parse_in(Language::English, phrase)
            .map_err(|e| ModelCryptoError::InvalidKey(e.to_string()))?;
        let seed = mnemonic.to_seed("");
        Self::from_secret_key_bytes(&seed[..32], network_type)
    }

    /// Create an identity from 32 raw secret-key bytes.
    pub fn from_secret_key_bytes(secret: &[u8], network_type: NetworkType) -> Result<Self, ModelCryptoError> {
        let secret_key = SecretKey::from_slice(secret).map_err(|e| ModelCryptoError::InvalidKey(e.to_string()))?;
        Ok(Self::from_secret_key(secret_key, network_type))
    }

    /// Create an identity from a raw secret key.
    pub fn from_secret_key(secret_key: SecretKey, network_type: NetworkType) -> Self {
        let public_key = PublicKey::from_secret_key_global(&secret_key);
        let address = derive_address(&public_key, network_type);
        Self { secret_key, public_key, address, network_type }
    }

    /// Try to load an identity from environment variables.
    ///
    /// Falls back through:
    /// 1. `XENO_GOSSIP_KEY` (64-char hex) -> use as raw secp256k1 secret key.
    /// 2. `XENO_GOSSIP_MNEMONIC` -> BIP39 phrase.
    pub fn from_env(network_type: NetworkType) -> Result<Self, ModelCryptoError> {
        if let Ok(key_hex) = std::env::var(GOSSIP_KEY_ENV) {
            let key_hex = key_hex.trim();
            if key_hex.len() == 64 {
                if let Ok(decoded) = hex::decode(key_hex) {
                    if decoded.len() == 32 {
                        return Self::from_secret_key_bytes(&decoded, network_type);
                    }
                }
            }
        }

        if let Ok(phrase) = std::env::var(GOSSIP_MNEMONIC_ENV) {
            return Self::from_mnemonic(&phrase, network_type);
        }

        Err(ModelCryptoError::InvalidKey("no gossip key configured".to_string()))
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn public_key(&self) -> &PublicKey {
        &self.public_key
    }

    pub fn network_type(&self) -> NetworkType {
        self.network_type
    }

    /// Sign and return an announcement.
    pub fn sign(&self, mut announcement: Announcement) -> Result<Announcement, ModelCryptoError> {
        // Ensure the public address matches this identity.
        announcement.public_key = self.public_key.serialize();
        announcement.node_address = self.address.clone();

        let digest = announcement.digest();
        let message = Message::from_digest_slice(&digest).map_err(|e| ModelCryptoError::InvalidKey(e.to_string()))?;
        let signature = self.secret_key.sign_ecdsa(message);
        announcement.signature = signature.serialize_compact();

        Ok(announcement)
    }

    /// Verify that an announcement was signed by this identity.
    pub fn verify_own(&self, announcement: &Announcement) -> Result<(), ModelCryptoError> {
        if announcement.node_address != self.address {
            return Err(ModelCryptoError::InvalidKey("announcement address mismatch".to_string()));
        }
        verify_announcement(announcement, self.network_type)
    }
}

/// Verify an announcement: check that the address is derived from the included
/// public key and that the signature is valid.
pub fn verify_announcement(announcement: &Announcement, network_type: NetworkType) -> Result<(), ModelCryptoError> {
    let public_key = PublicKey::from_slice(&announcement.public_key)
        .map_err(|e| ModelCryptoError::InvalidKey(format!("invalid public key: {e}")))?;
    let expected_address = derive_address(&public_key, network_type);
    if announcement.node_address != expected_address {
        return Err(ModelCryptoError::InvalidKey("address does not match public key".to_string()));
    }

    let digest = announcement.digest();
    let message = Message::from_digest_slice(&digest).map_err(|e| ModelCryptoError::InvalidKey(e.to_string()))?;
    let signature = secp256k1::ecdsa::Signature::from_compact(&announcement.signature)
        .map_err(|e| ModelCryptoError::InvalidKey(format!("invalid signature: {e}")))?;

    signature
        .verify(&message, &public_key)
        .map_err(|_| ModelCryptoError::InvalidKey("signature verification failed".to_string()))
}

fn derive_address(public_key: &PublicKey, network_type: NetworkType) -> String {
    let (x_only_public_key, _) = public_key.x_only_public_key();
    let prefix = Prefix::from(network_type);
    let address = Address::new(prefix, Version::PubKey, &x_only_public_key.serialize());
    String::from(&address)
}

/// A registry of checkpoint announcements received from peers.
#[derive(Debug, Clone)]
pub struct GossipRegistry {
    entries: HashMap<[u8; 32], Vec<AnnouncementEntry>>,
    ttl: Duration,
}

#[derive(Debug, Clone)]
struct AnnouncementEntry {
    announcement: Announcement,
    received_at: Instant,
}

impl GossipRegistry {
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_GOSSIP_TTL)
    }

    pub fn with_ttl(ttl: Duration) -> Self {
        Self { entries: HashMap::new(), ttl }
    }

    /// Insert a verified announcement. Returns `true` if it was new.
    pub fn insert(&mut self, announcement: Announcement) -> bool {
        self.prune();

        let key = announcement.weights_hash;
        let list = self.entries.entry(key).or_default();

        let is_duplicate = list.iter().any(|entry| {
            entry.announcement.public_key == announcement.public_key
                && entry.announcement.timestamp == announcement.timestamp
        });

        if is_duplicate {
            return false;
        }

        list.push(AnnouncementEntry { announcement, received_at: Instant::now() });
        true
    }

    /// Get all known announcements for a checkpoint.
    pub fn get(&self, weights_hash: &[u8; 32]) -> Vec<Announcement> {
        self.entries.get(weights_hash).map_or_else(Vec::new, |list| {
            list.iter().filter(|entry| entry.received_at.elapsed() <= self.ttl).map(|entry| entry.announcement.clone()).collect()
        })
    }

    /// Get all known announcements for a `model_id`.
    pub fn get_by_model(&self, model_id: &str) -> Vec<Announcement> {
        self.entries
            .values()
            .flatten()
            .filter(|entry| entry.received_at.elapsed() <= self.ttl && entry.announcement.model_id == model_id)
            .map(|entry| entry.announcement.clone())
            .collect()
    }

    /// Prune expired entries from the registry.
    pub fn prune(&mut self) {
        let ttl = self.ttl;
        self.entries.retain(|_, list| {
            list.retain(|entry| entry.received_at.elapsed() <= ttl);
            !list.is_empty()
        });
    }

    pub fn len(&self) -> usize {
        self.entries.values().map(|list| list.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for GossipRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kaspa_consensus_core::network::NetworkType;

    const TEST_PHRASE: &str = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon art";

    fn test_announcement(model_id: &str, weights_hash: [u8; 32]) -> Announcement {
        Announcement {
            model_id: model_id.to_string(),
            weights_hash,
            cid: [1u8; 32],
            timestamp: 1234567890,
            is_genome: false,
            node_address: String::new(),
            public_key: [0u8; 33],
            listen_addr: Some("127.0.0.1:17110".parse().unwrap()),
            signature: [0u8; 64],
        }
    }

    #[test]
    fn test_sign_verify_roundtrip() {
        let identity = GossipIdentity::from_mnemonic(TEST_PHRASE, NetworkType::Devnet).unwrap();
        let ann = test_announcement("multimolecule/dnabert2", [42u8; 32]);
        let signed = identity.sign(ann).unwrap();

        assert_eq!(&signed.node_address, identity.address());
        assert!(verify_announcement(&signed, NetworkType::Devnet).is_ok());
    }

    #[test]
    fn test_registry_insert_and_get() {
        let identity = GossipIdentity::from_mnemonic(TEST_PHRASE, NetworkType::Devnet).unwrap();
        let ann = identity.sign(test_announcement("model", [7u8; 32])).unwrap();

        let mut registry = GossipRegistry::new();
        assert!(registry.insert(ann.clone()));
        assert!(!registry.insert(ann.clone()));

        let got = registry.get(&[7u8; 32]);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn test_tampered_signature_fails() {
        let identity = GossipIdentity::from_mnemonic(TEST_PHRASE, NetworkType::Devnet).unwrap();
        let mut signed = identity.sign(test_announcement("model", [9u8; 32])).unwrap();
        signed.signature[0] ^= 0xff;

        assert!(verify_announcement(&signed, NetworkType::Devnet).is_err());
    }
}
