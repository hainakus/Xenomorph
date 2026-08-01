//! Build LoRA-only training artifacts for the secure distribution protocol.
//!
//! This module extracts only the frozen LM head and tied embedding weights from
//! a full DNABERT-2 checkpoint, packages them in a `TrainingArtifact`, and
//! encrypts + signs the artifact using the model-crypto primitives.

use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use model_crypto::artifact_sign::ArtifactSigner;
use model_crypto::key_hierarchy::{ModelKeyHierarchy, ModelSecret};
use model_crypto::session;
use secp256k1::PublicKey;

use crate::model::manager::ModelManager;
use crate::rpc::messages::{ArtifactType, GetTrainingArtifact, TrainingArtifact};

const LM_HEAD_KEYS: &[&str] = &[
    "lm_head.transform.dense.weight",
    "lm_head.transform.dense.bias",
    "lm_head.transform.layer_norm.weight",
    "lm_head.transform.layer_norm.bias",
    "lm_head.bias",
    "model.embeddings.word_embeddings.weight",
];

const MASTER_KEY_ENV: &str = "XENO_MODEL_MASTER_KEY";
const MASTER_KEY_FILE: &str = "model_master.key";

/// Build a LoRA-only training artifact from the active checkpoint.
///
/// The artifact is encrypted to a session key derived from an ephemeral ECDH key
/// and the miner's public key, then signed with the model auth key.
pub async fn build_training_artifact(model_manager: &ModelManager, request: &GetTrainingArtifact) -> Result<TrainingArtifact> {
    let model_id = request.model_id.clone();
    let (_checkpoint, files) = model_manager
        .get_model_checkpoint(&model_id)
        .await
        .with_context(|| format!("Failed to load model {} for artifact", model_id))?;

    let config = files.config.clone();
    let tokenizer = files.tokenizer.clone();

    // Load the full base safetensors and keep only the LM head + embedding keys.
    let device = Device::Cpu;
    let loaded = candle_core::safetensors::load_buffer(&files.weights, &device)
        .map_err(|e| anyhow!("Failed to load safetensors for {}: {}", model_id, e))?;

    let mut artifact_tensors: Vec<(String, &Tensor)> = Vec::new();
    for key in LM_HEAD_KEYS {
        let t = loaded.get(*key).ok_or_else(|| anyhow!("Missing required LM head key '{}' in {}", key, model_id))?;
        artifact_tensors.push((key.to_string(), t));
    }

    let artifact =
        safetensors::tensor::serialize(artifact_tensors, &None).map_err(|e| anyhow!("Failed to serialize LM head artifact: {}", e))?;
    let artifact_hash = blake3_hash(&artifact);
    let base_checkpoint = request.base_checkpoint;
    let base_hash = request.base_checkpoint;

    // Load (or create) the persistent model key hierarchy.
    let data_dir = Path::new(model_manager.base_path()).join(sanitize_id(&model_id));
    let hierarchy = load_or_create_hierarchy(&model_id, &data_dir)?;

    // Build the auth signer and public key.
    let auth_key = hierarchy.auth_key().context("Failed to derive auth key")?;
    let signer = ArtifactSigner::from_auth_key(&auth_key).context("Failed to create artifact signer")?;
    let auth_public_key = signer.public_key();

    // Sign the artifact before encryption so the signature covers plaintext.
    let signature = signer.sign(&artifact_hash, Some(&base_hash)).context("Failed to sign artifact")?;

    // Encrypt the artifact with a per-miner session key.
    let miner_public_key = parse_public_key(&request.miner_public_key)?;
    let (ephemeral_secret, ephemeral_public_key) = session::generate_ephemeral_keypair();
    let session_nonce = [0u8; 12]; // Fixed nonce for the spike; in production use a random nonce per artifact.
    let shared_secret = session::orchestrator_shared_secret(&ephemeral_secret, &miner_public_key);
    let session_key = session::derive_session_key(&shared_secret, &session_nonce)?;
    let encrypted_artifact = session_key.encrypt(&artifact)?;

    let recipient_key_fingerprint = blake3_hash(&request.miner_public_key);

    Ok(TrainingArtifact {
        model_id,
        base_checkpoint,
        base_hash,
        config,
        tokenizer,
        artifact: encrypted_artifact,
        artifact_type: ArtifactType::LoRA,
        artifact_hash,
        encrypted: true,
        recipient_key_fingerprint,
        signature: signature.signature,
        ephemeral_public_key: ephemeral_public_key.serialize(),
        session_nonce,
        auth_public_key,
    })
}

fn parse_public_key(bytes: &[u8; 33]) -> Result<PublicKey> {
    PublicKey::from_slice(bytes).map_err(|e| anyhow!("Invalid miner public key: {}", e))
}

fn sanitize_id(model_id: &str) -> String {
    model_id.chars().map(|c| if c == '/' || c == '\\' || c == ':' || c == ' ' || c == '\0' { '_' } else { c }).collect()
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}

fn load_or_create_hierarchy(model_id: &str, data_dir: &Path) -> Result<ModelKeyHierarchy> {
    // Try a stable master key from the environment first.
    if let Ok(hex_key) = std::env::var(MASTER_KEY_ENV) {
        if let Ok(bytes) = hex::decode(hex_key.trim()) {
            if bytes.len() == 32 {
                let mut master = [0u8; 32];
                master.copy_from_slice(&bytes);
                return Ok(ModelKeyHierarchy::new(ModelSecret::new(master), model_id, 1));
            }
        }
    }

    // Fall back to a file in the model data directory.
    let key_path = data_dir.join(MASTER_KEY_FILE);
    if key_path.exists() {
        let hex = std::fs::read_to_string(&key_path).with_context(|| format!("Failed to read master key from {:?}", key_path))?;
        let bytes = hex::decode(hex.trim()).with_context(|| format!("Invalid hex in master key file {:?}", key_path))?;
        if bytes.len() != 32 {
            bail!("Master key in {:?} is not 32 bytes", key_path);
        }
        let mut master = [0u8; 32];
        master.copy_from_slice(&bytes);
        return Ok(ModelKeyHierarchy::new(ModelSecret::new(master), model_id, 1));
    }

    // No existing key; generate one and warn.
    let master = ModelSecret::random();
    std::fs::create_dir_all(data_dir)?;
    std::fs::write(&key_path, hex::encode(master.0)).with_context(|| format!("Failed to write master key to {:?}", key_path))?;
    tracing::warn!("No XENO_MODEL_MASTER_KEY set; generated a new master key for {} at {:?}", model_id, key_path);
    Ok(ModelKeyHierarchy::new(master, model_id, 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use model_crypto::key_hierarchy::ModelKeyHierarchy;
    #[test]
    fn test_artifact_signer_from_hierarchy() {
        let hierarchy = ModelKeyHierarchy::random("xeno/mgm-1", 1);
        let auth_key = hierarchy.auth_key().unwrap();
        let signer = ArtifactSigner::from_auth_key(&auth_key).unwrap();
        let _ = signer.public_key();
    }
}
