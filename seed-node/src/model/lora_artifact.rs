//! Build LoRA-only training artifacts for the secure distribution protocol.
//!
//! This module extracts only the frozen LM head and tied embedding weights from
//! a full DNABERT-2 checkpoint, producing a small `TrainingArtifact` that a
//! miner can load into `LoraLmHead`.

use anyhow::{anyhow, Context, Result};
use candle_core::{Device, Tensor};

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

/// Build a LoRA-only training artifact from the active checkpoint.
///
/// For this spike the artifact is **not encrypted** and the signature is a
/// placeholder.  The returned `artifact` buffer is a safetensors file that
/// contains only the LM head / embedding weights the miner needs.
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

    // In a production implementation the artifact is encrypted to the miner's
    // session key here and signed with the model auth key.
    let artifact_hash = blake3_hash(&artifact);
    let base_checkpoint = request.base_checkpoint;
    let base_hash = request.base_checkpoint;

    Ok(TrainingArtifact {
        model_id,
        base_checkpoint,
        base_hash,
        config,
        tokenizer,
        artifact,
        artifact_type: ArtifactType::LoRA,
        artifact_hash,
        encrypted: false,
        recipient_key_fingerprint: blake3_hash(&request.miner_public_key),
        signature: [0u8; 64],
    })
}

fn blake3_hash(data: &[u8]) -> [u8; 32] {
    let hash = blake3::hash(data);
    let mut out = [0u8; 32];
    out.copy_from_slice(hash.as_bytes());
    out
}
