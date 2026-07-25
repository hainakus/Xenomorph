use std::collections::HashMap;

use anyhow::{Context, Result};
use candle_core::Tensor;
use model_crypto::{decrypt, derive_encryption_key};
use tracing::{info, warn};

pub use crate::model_cache::{ModelBundle, ModelCache};
use crate::rpc::messages::ModelCheckpointV2;
use crate::rpc::XenomRpcClient;

/// Decrypt the encrypted model files returned by the seed-node.
fn decrypt_v2(cp: &ModelCheckpointV2) -> Result<ModelCheckpointV2> {
    if !cp.encrypted {
        return Ok(cp.clone());
    }

    let key = derive_encryption_key();
    let config = decrypt(&cp.config, &key).context("Failed to decrypt model config")?;
    let tokenizer = decrypt(&cp.tokenizer, &key).context("Failed to decrypt model tokenizer")?;
    let weights = decrypt(&cp.weights, &key).context("Failed to decrypt model weights")?;

    Ok(ModelCheckpointV2 {
        model_id: cp.model_id.clone(),
        base_checkpoint: cp.base_checkpoint,
        base_hash: cp.base_hash,
        config,
        tokenizer,
        weights,
        encrypted: false,
        is_adapter: cp.is_adapter,
    })
}

/// Return a human-readable error if `weights` is clearly a bad payload (HTML page,
/// git-lfs pointer, empty). Both safetensors and PyTorch .bin/.pth are accepted;
/// the trainer/inference engine validates the concrete format.
fn validate_weights_payload(weights: &[u8]) -> Result<()> {
    if weights.is_empty() {
        anyhow::bail!("weights buffer is empty");
    }
    if weights.starts_with(b"version https://git-lfs.github.com/spec/v1") {
        anyhow::bail!("weights look like a git-lfs pointer instead of a checkpoint file");
    }
    if weights.starts_with(b"<!DOCTYPE") || weights.starts_with(b"<html") || weights.starts_with(b"<HTML") {
        anyhow::bail!("weights look like an HTML error page instead of a checkpoint file");
    }
    Ok(())
}

/// Extract the frozen base weights from a merged LoRA checkpoint.
///
/// All tensors except `*.lora_a` and `*.lora_b` are considered base weights.
fn extract_base_weights(merged: &[u8]) -> Result<Vec<u8>> {
    let tensors = candle_core::safetensors::load_buffer(merged, &candle_core::Device::Cpu)
        .context("Failed to load merged weights for base extraction")?;
    let base: HashMap<String, Tensor> =
        tensors.into_iter().filter(|(k, _)| !k.ends_with(".lora_a") && !k.ends_with(".lora_b")).collect();
    let serialized: Vec<(String, &Tensor)> = base.iter().map(|(k, v)| (k.clone(), v)).collect();
    safetensors::tensor::serialize(serialized, &None).map_err(|e| anyhow::anyhow!("Failed to serialize base weights: {}", e))
}

/// Merge a LoRA adapter into a base weights buffer, producing a full checkpoint.
fn merge_adapter_into_base(base: &[u8], adapter: &[u8]) -> Result<Vec<u8>> {
    let mut merged =
        candle_core::safetensors::load_buffer(base, &candle_core::Device::Cpu).context("Failed to load cached base weights")?;
    let adapter_tensors =
        candle_core::safetensors::load_buffer(adapter, &candle_core::Device::Cpu).context("Failed to load adapter weights")?;
    for (k, v) in adapter_tensors {
        merged.insert(k, v);
    }
    let serialized: Vec<(String, &Tensor)> = merged.iter().map(|(k, v)| (k.clone(), v)).collect();
    safetensors::tensor::serialize(serialized, &None).map_err(|e| anyhow::anyhow!("Failed to serialize merged weights: {}", e))
}

/// Fetch a model checkpoint from the node, using the local cache when the
/// active weights hash has not changed.
///
/// Phase 2: the miner first requests V2 checkpoint metadata. If the cached
/// base weights match, only the LoRA adapter is downloaded and merged with the
/// local base. Otherwise the full base+adapter bundle is downloaded.
///
/// The returned `ModelBundle` is always plaintext; if the node sent encrypted
/// files they are decrypted with the same `XENO_MODEL_KEY` used by the node.
pub async fn fetch_model_checkpoint(rpc: &mut XenomRpcClient, model_id: &str, cache: &ModelCache) -> Result<ModelBundle> {
    let info = rpc.get_model_checkpoint_info_v2(model_id).await?;

    // If we already have the combined checkpoint cached, use it directly.
    if cache.is_cached(model_id) {
        match cache.read_base_checkpoint(model_id) {
            Ok(cached_hash) if cached_hash == info.base_checkpoint => {
                info!("Using cached model checkpoint for {} (hash {})", model_id, hex::encode(cached_hash));
                let bundle = cache.read(model_id)?;
                if let Err(e) = validate_weights_payload(&bundle.weights) {
                    warn!("Cached checkpoint for {} looks invalid ({}); clearing local cache and re-downloading", model_id, e);
                    cache.clear(model_id)?;
                } else {
                    return Ok(bundle);
                }
            }
            Ok(cached_hash) => {
                info!(
                    "Model checkpoint for {} changed (cached {} != current {}); refreshing",
                    model_id,
                    hex::encode(cached_hash),
                    hex::encode(info.base_checkpoint)
                );
            }
            Err(e) => {
                info!("Failed to read cached base checkpoint for {}: {}; re-downloading", model_id, e);
            }
        }
    }

    // If the cached base weights match the active base, download only the adapter.
    let cached_base_hash = cache.read_base_hash(model_id);
    let cp = if cached_base_hash == Some(info.base_hash) {
        info!("Base weights for {} already cached; fetching adapter only", model_id);
        let cp = rpc.get_model_checkpoint_v2(model_id, Some(info.base_hash)).await?;
        if !cp.is_adapter {
            warn!("Node returned a full checkpoint despite cached base match; using full response");
        }
        cp
    } else {
        info!("Fetching full model checkpoint for {} from node", model_id);
        rpc.get_model_checkpoint_v2(model_id, None).await?
    };

    let mut cp = decrypt_v2(&cp)?;

    if let Err(e) = validate_weights_payload(&cp.weights) {
        anyhow::bail!(
            "Node returned an invalid payload for {}: {}. \
             Clear the model cache on the node and restart it.",
            model_id,
            e
        );
    }

    // If the node sent only the adapter, merge it with the cached base.
    if cp.is_adapter {
        let base = cache.read_base_weights(model_id).context("Missing cached base weights for adapter merge")?;
        cp.weights = merge_adapter_into_base(&base, &cp.weights)?;
    }

    let bundle = ModelBundle {
        model_id: model_id.to_string(),
        base_checkpoint: cp.base_checkpoint,
        config: cp.config,
        tokenizer: cp.tokenizer,
        weights: cp.weights,
    };

    // Update the base cache whenever we receive a full checkpoint.
    if !cp.is_adapter {
        match extract_base_weights(&bundle.weights) {
            Ok(base_weights) => {
                if let Err(e) = cache.write_base(model_id, info.base_hash, &base_weights) {
                    warn!("Failed to cache base weights for {}: {}; continuing", model_id, e);
                }
            }
            Err(e) => warn!("Failed to extract base weights for {}: {}; continuing", model_id, e),
        }
    }

    if let Err(e) = cache.write(model_id, &bundle) {
        info!("Failed to cache model checkpoint for {}: {}; continuing with in-memory bundle", model_id, e);
    }

    Ok(bundle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    fn make_safetensors(tensors: &[(String, Tensor)]) -> Vec<u8> {
        let refs: Vec<(String, &Tensor)> = tensors.iter().map(|(k, v)| (k.clone(), v)).collect();
        safetensors::tensor::serialize(refs, &None).unwrap()
    }

    #[test]
    fn test_extract_base_weights_removes_lora() {
        let device = Device::Cpu;
        let base_w = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let lora_a = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((2, 2), DType::F32, &device).unwrap();

        let merged = make_safetensors(&[
            ("layer.weight".to_string(), base_w),
            ("layer.lora_a".to_string(), lora_a),
            ("layer.lora_b".to_string(), lora_b),
        ]);

        let base = extract_base_weights(&merged).unwrap();
        let parsed = candle_core::safetensors::load_buffer(&base, &device).unwrap();
        assert!(parsed.contains_key("layer.weight"));
        assert!(!parsed.contains_key("layer.lora_a"));
        assert!(!parsed.contains_key("layer.lora_b"));
    }

    #[test]
    fn test_merge_adapter_into_base() {
        let device = Device::Cpu;
        let base_w = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let lora_a = Tensor::zeros((2, 2), DType::F32, &device).unwrap();
        let lora_b = Tensor::zeros((2, 2), DType::F32, &device).unwrap();

        let base = make_safetensors(&[("layer.weight".to_string(), base_w)]);
        let adapter = make_safetensors(&[("layer.lora_a".to_string(), lora_a), ("layer.lora_b".to_string(), lora_b)]);

        let merged = merge_adapter_into_base(&base, &adapter).unwrap();
        let parsed = candle_core::safetensors::load_buffer(&merged, &device).unwrap();
        assert!(parsed.contains_key("layer.weight"));
        assert!(parsed.contains_key("layer.lora_a"));
        assert!(parsed.contains_key("layer.lora_b"));
    }
}
