use anyhow::{Context, Result};
use model_crypto::{decrypt, derive_encryption_key};
use safetensors::SafeTensors;
use tracing::{info, warn};

pub use crate::model_cache::{ModelBundle, ModelCache};
use crate::rpc::XenomRpcClient;

/// Decrypt the encrypted model files returned by the seed-node.
fn decrypt_model_files(cp: &crate::rpc::messages::ModelCheckpoint) -> Result<ModelBundle> {
    if !cp.encrypted {
        return Ok(ModelBundle {
            model_id: cp.model_id.clone(),
            base_checkpoint: cp.base_checkpoint,
            config: cp.config.clone(),
            tokenizer: cp.tokenizer.clone(),
            weights: cp.weights.clone(),
        });
    }

    let key = derive_encryption_key();
    let config = decrypt(&cp.config, &key).context("Failed to decrypt model config")?;
    let tokenizer = decrypt(&cp.tokenizer, &key).context("Failed to decrypt model tokenizer")?;
    let weights = decrypt(&cp.weights, &key).context("Failed to decrypt model weights")?;

    Ok(ModelBundle { model_id: cp.model_id.clone(), base_checkpoint: cp.base_checkpoint, config, tokenizer, weights })
}

/// Return a human-readable error if `weights` is not a valid safetensors buffer.
fn validate_safetensors(weights: &[u8]) -> Result<()> {
    if weights.starts_with(b"PK\x03\x04") || weights.first() == Some(&0x80) {
        anyhow::bail!("weights look like a PyTorch .bin/.pth file; only safetensors is supported");
    }
    SafeTensors::deserialize(weights).context("weights are not a valid safetensors file")?;
    Ok(())
}

/// Fetch a model checkpoint from the node, using the local cache when the
/// active weights hash has not changed.
///
/// This avoids downloading ~400-500 MB of safetensors weights on every miner
/// restart. The node only needs to send a small `GetModelCheckpointInfo`
/// response; the full checkpoint is requested only when the cache is missing
/// or stale.
///
/// The returned `ModelBundle` is always plaintext; if the node sent encrypted
/// files they are decrypted with the same `XENO_MODEL_KEY` used by the node.
pub async fn fetch_model_checkpoint(rpc: &mut XenomRpcClient, model_id: &str, cache: &ModelCache) -> Result<ModelBundle> {
    let info = rpc.get_model_checkpoint_info(model_id).await?;

    if cache.is_cached(model_id) {
        match cache.read_base_checkpoint(model_id) {
            Ok(cached_hash) if cached_hash == info.base_checkpoint => {
                info!("Using cached model checkpoint for {} (hash {})", model_id, hex::encode(cached_hash));
                let bundle = cache.read(model_id)?;
                if let Err(e) = validate_safetensors(&bundle.weights) {
                    warn!("Cached checkpoint for {} is invalid ({}); clearing local cache and re-downloading", model_id, e);
                    cache.clear(model_id)?;
                } else {
                    return Ok(bundle);
                }
            }
            Ok(cached_hash) => {
                info!(
                    "Model checkpoint for {} changed (cached {} != current {}); re-downloading",
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

    info!("Fetching full model checkpoint for {} from node", model_id);
    let cp = rpc.get_model_checkpoint(model_id).await?;
    let bundle = decrypt_model_files(&cp)?;

    if let Err(e) = validate_safetensors(&bundle.weights) {
        anyhow::bail!(
            "Node returned non-safetensors weights for {}: {}. \
             Clear the model cache on the node ({} -> models/<model>) and restart it so it re-downloads model.safetensors from Hugging Face.",
            model_id,
            e,
            std::env::var("XENO_MODELS_DIR").as_deref().unwrap_or("<XENO_MODELS_DIR>")
        );
    }

    if let Err(e) = cache.write(model_id, &bundle) {
        info!("Failed to cache model checkpoint for {}: {}; continuing with in-memory bundle", model_id, e);
    }

    Ok(bundle)
}
