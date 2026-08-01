//! Merge a LoRA adapter delta into a base checkpoint.
//!
//! The miner submits a signed safetensors buffer containing only the trainable
//! LoRA A/B tensors for `lm_head.transform.dense`.  The orchestrator verifies
//! the signature and plaintext hash, then merges the adapter into the base
//! `lm_head.transform.dense.weight` to produce a new active checkpoint.

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context, Result};
use candle_core::{Device, Tensor};
use model_crypto::session;
use secp256k1::{Message, PublicKey, Secp256k1};

use crate::model::manager::ModelManager;
use crate::model::RawModelFiles;
use crate::rpc::messages::SubmitLoRAUpdate;

const LORA_DENSE_KEY: &str = "lm_head.transform.dense.weight";
const LORA_A_KEY: &str = "lm_head.transform.dense.lora_a";
const LORA_B_KEY: &str = "lm_head.transform.dense.lora_b";

/// Verify and merge a LoRA delta into the active checkpoint.
///
/// Returns the hash of the new merged weights.
pub async fn apply_lora_update(model_manager: Arc<ModelManager>, update: &SubmitLoRAUpdate) -> Result<[u8; 32]> {
    // Verify the miner signature over the plaintext delta hash and base checkpoint.
    let message = build_lora_delta_message_hash(&update.lora_delta_hash, update.base_checkpoint);
    let message = Message::from_digest(message);
    let signature =
        secp256k1::ecdsa::Signature::from_compact(&update.signature).map_err(|e| anyhow!("Invalid LoRA delta signature: {}", e))?;
    let public_key = PublicKey::from_slice(&update.auth_public_key).map_err(|e| anyhow!("Invalid auth public key: {}", e))?;
    let secp = Secp256k1::verification_only();
    secp.verify_ecdsa(&message, &signature, &public_key).map_err(|e| anyhow!("LoRA delta signature verification failed: {}", e))?;

    // Decrypt the LoRA delta if it reuses an AttestedForward session key.
    let delta_bytes = if update.encrypted {
        let session = model_manager
            .take_forward_session(&update.ephemeral_public_key)
            .await
            .ok_or_else(|| anyhow!("No cached AttestedForward session for LoRA delta decryption"))?;
        let miner_public_key =
            PublicKey::from_slice(&update.miner_public_key).map_err(|e| anyhow!("Invalid miner public key: {}", e))?;
        let shared_secret = session::orchestrator_shared_secret(&session.ephemeral_secret, &miner_public_key);
        let session_key = session::derive_session_key(&shared_secret, &session.session_nonce)?;
        session_key.decrypt(&update.lora_delta)?
    } else {
        update.lora_delta.clone()
    };

    // Verify the plaintext hash.
    let plaintext_hash = blake3_hash(&delta_bytes);
    if plaintext_hash != update.lora_delta_hash {
        bail!("LoRA delta hash mismatch");
    }

    // Load the base checkpoint.
    let (checkpoint, files) = model_manager
        .get_model_checkpoint(&update.model_id)
        .await
        .with_context(|| format!("Failed to load base checkpoint for {}", update.model_id))?;

    if checkpoint.weights_hash != update.base_checkpoint {
        bail!(
            "Base checkpoint mismatch for LoRA merge: expected {} != update {}",
            hex::encode(checkpoint.weights_hash),
            hex::encode(update.base_checkpoint)
        );
    }

    let device = Device::Cpu;
    let mut tensors =
        candle_core::safetensors::load_buffer(&files.weights, &device).map_err(|e| anyhow!("Failed to load base weights: {}", e))?;

    // Load the LoRA adapter.
    let lora = candle_core::safetensors::load_buffer(&delta_bytes, &device)
        .map_err(|e| anyhow!("Failed to load LoRA delta safetensors: {}", e))?;

    // Compute the merged dense weight: W_new = W_base + (alpha/rank) * (lora_b @ lora_a).
    let base_weight = tensors.get(LORA_DENSE_KEY).ok_or_else(|| anyhow!("Missing {} in base checkpoint", LORA_DENSE_KEY))?.clone();
    let lora_a = lora.get(LORA_A_KEY).ok_or_else(|| anyhow!("Missing {} in LoRA delta", LORA_A_KEY))?;
    let lora_b = lora.get(LORA_B_KEY).ok_or_else(|| anyhow!("Missing {} in LoRA delta", LORA_B_KEY))?;

    let scale = infer_scale(&base_weight, lora_a, lora_b)?;
    let lora_b_a = lora_b.matmul(lora_a)?;
    let delta_weight = lora_b_a.broadcast_mul(&Tensor::new(scale as f32, &device)?.to_dtype(base_weight.dtype())?)?;
    let new_weight = base_weight.broadcast_add(&delta_weight)?;

    tensors.insert(LORA_DENSE_KEY.to_string(), new_weight);

    // Serialize the merged checkpoint and store it as the new active checkpoint.
    let merged_safetensors = serialize_safetensors(&tensors)?;
    let new_hash = blake3_hash(&merged_safetensors);

    let new_files = RawModelFiles { config: files.config.clone(), tokenizer: files.tokenizer.clone(), weights: merged_safetensors };

    model_manager
        .store_model_files(&update.model_id, &new_files, Default::default())
        .await
        .with_context(|| format!("Failed to store merged checkpoint for {}", update.model_id))?;

    Ok(new_hash)
}

fn build_lora_delta_message_hash(delta_hash: &[u8; 32], base_checkpoint: [u8; 32]) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"xenom-lora-delta-v1");
    hasher.update(delta_hash);
    hasher.update(&base_checkpoint);
    out.copy_from_slice(hasher.finalize().as_bytes());
    out
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}

fn infer_scale(base_weight: &Tensor, lora_a: &Tensor, lora_b: &Tensor) -> Result<f64> {
    let (_, in_features) = base_weight.dims2()?;
    let lora_a_dims = lora_a.dims();
    let lora_b_dims = lora_b.dims();
    if lora_a_dims[1] != in_features {
        bail!("LoRA A in_features mismatch: {} != {}", lora_a_dims[1], in_features);
    }
    if lora_b_dims[0] != in_features || lora_b_dims[1] != lora_a_dims[0] {
        bail!("LoRA B shape mismatch: {:?} with A {:?}", lora_b_dims, lora_a_dims);
    }
    let rank = lora_a_dims[0];
    // Default alpha is twice the rank, matching LoraConfig::default().
    let alpha = 2.0 * rank as f64;
    Ok(alpha / rank as f64)
}

fn serialize_safetensors(tensors: &HashMap<String, Tensor>) -> Result<Vec<u8>> {
    let tensors: Vec<(String, &Tensor)> = tensors.iter().map(|(k, v)| (k.clone(), v)).collect();
    safetensors::tensor::serialize(tensors, &None).map_err(|e| anyhow!("Failed to serialize merged weights: {}", e))
}
