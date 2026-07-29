use anyhow::{anyhow, Context, Result};
use reqwest::Url;
use std::fs;
use std::time::Duration;
use tracing::{info, warn};

use super::RawModelFiles;

const HF_HUB_URL: &str = "https://huggingface.co";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Download a complete model checkpoint (config.json, tokenizer.json, weights) from Hugging Face.
/// Prefers `model.safetensors`; falls back to `pytorch_model.bin` (zip format) if needed.
/// Built-in models such as `xeno/mgm-1` are generated locally instead of downloaded.
///
/// If `from_scratch` is true and no weights file is available, an empty weights buffer
/// is returned.  This lets the trainer initialize the model with random weights.
pub async fn download_model(model_id: &str, from_scratch: bool) -> Result<RawModelFiles> {
    if model_id.contains("mgm-1") {
        info!("Generating default MGM-1 checkpoint for {}", model_id);
        return build_default_mgm1_files();
    }
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .context("Failed to build HTTP client")?;

    let config = download_file(&client, &model_url(model_id, "config.json")?).await.context("Failed to download config.json")?;
    info!("Downloaded config.json for {}", model_id);

    let tokenizer =
        download_file(&client, &model_url(model_id, "tokenizer.json")?).await.context("Failed to download tokenizer.json")?;
    info!("Downloaded tokenizer.json for {}", model_id);

    let mut weights: Option<Vec<u8>> = None;
    for filename in ["model.safetensors", "pytorch_model.bin"] {
        match download_and_validate_weights(&client, model_id, filename).await {
            Ok(data) => {
                info!("Downloaded {} ({:.2} MB) for {}", filename, data.len() as f64 / 1_048_576.0, model_id);
                weights = Some(data);
                break;
            }
            Err(e) => {
                warn!("Could not use {} for {}: {}", filename, model_id, e);
            }
        }
    }

    let weights = match weights {
        Some(w) => w,
        None if from_scratch => {
            warn!("No weights file found for {} and --from-scratch is set; training from scratch", model_id);
            Vec::new()
        }
        None => {
            return Err(anyhow!(
                "Could not download valid safetensors or pytorch_model.bin weights for {} from Hugging Face",
                model_id
            ));
        }
    };

    Ok(RawModelFiles { config, tokenizer, weights })
}

/// Download a weights file and verify it is a real checkpoint, not an HTML/LFS pointer.
/// If the plain resolve URL returns a pointer, retry once with `?download=true`.
async fn download_and_validate_weights(client: &reqwest::Client, model_id: &str, filename: &str) -> Result<Vec<u8>> {
    let base_url = model_url(model_id, filename)?;

    for url in [&base_url, &format!("{}?download=true", base_url)] {
        info!("Attempting to download model weights from {}", url);
        match download_file(client, url).await {
            Ok(data) if !data.is_empty() && is_valid_weights(&data) => return Ok(data),
            Ok(data) => {
                warn!("Downloaded {} but content does not look like valid weights ({} bytes); will retry if possible", url, data.len())
            }
            Err(e) => warn!("Failed to download {}: {}", url, e),
        }
    }

    Err(anyhow!("{} from {} is not a valid weights file", filename, model_id))
}

/// Validate that `data` is a real checkpoint file.
///
/// Accepts `model.safetensors` (validated with `safetensors`) and zip-format
/// PyTorch `.bin` / `.pth` files (PK magic bytes). Legacy pickle-only `.bin`
/// files (0x80) are not supported by the Rust loader and are rejected so they
/// can be re-downloaded as safetensors.
///
/// An empty buffer is also accepted: it means the model should be trained from
/// scratch, and the trainer will create randomly-initialized tensors from the
/// model config.
pub fn is_valid_weights(data: &[u8]) -> bool {
    if data.is_empty() {
        return true;
    }

    // Fast reject of the most common non-checkpoint payloads.
    if data.starts_with(b"version https://git-lfs.github.com/spec/v1")
        || data.starts_with(b"<!DOCTYPE")
        || data.starts_with(b"<html")
        || data.starts_with(b"<HTML")
    {
        return false;
    }

    // Accept zip-format PyTorch .bin/.pth (torch.save default since PyTorch 1.6).
    if data.starts_with(b"PK\x03\x04") {
        return true;
    }

    // Otherwise require a valid safetensors buffer.
    safetensors::SafeTensors::deserialize(data).is_ok()
}

/// Build a canonical Hugging Face resolve URL for a file in a model repo.
/// Model ids like `multimolecule/dnabert2` are treated as path segments so
/// the `/` separator is preserved while special characters are encoded.
fn model_url(model_id: &str, filename: &str) -> Result<String> {
    let mut url = Url::parse(HF_HUB_URL).context("Invalid HF hub base URL")?;
    {
        let mut path = url.path_segments_mut().map_err(|_| anyhow!("Cannot set path segments"))?;
        for segment in model_id.split('/') {
            path.push(segment);
        }
        path.push("resolve");
        path.push("main");
        path.push(filename);
    }
    Ok(url.to_string())
}

async fn download_file(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let response = client.get(url).send().await.context("HTTP request failed")?;

    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!("HTTP {} for {}", status, url));
    }

    let bytes = response.bytes().await.context("Failed to read response body")?;
    Ok(bytes.to_vec())
}

/// Generate a fresh `xeno/mgm-1` checkpoint with random weights.
fn build_default_mgm1_files() -> Result<RawModelFiles> {
    use candle_core::{DType, Device};
    use candle_nn::{VarBuilder, VarMap};
    use mini_genome_model::{MiniGenomeConfig, MiniGenomeModel};

    let device = Device::Cpu;
    let config = MiniGenomeConfig::default();
    let varmap = VarMap::new();

    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    MiniGenomeModel::new(vb, config.clone()).context("Failed to build default MGM-1 model")?;

    let tmp = std::env::temp_dir().join(format!("mgm1_default_{}.safetensors", rand::random::<u64>()));
    varmap.save(&tmp).context("Failed to save default MGM-1 weights")?;
    let weights = fs::read(&tmp).context("Failed to read default MGM-1 weights")?;
    let _ = fs::remove_file(&tmp);

    let config_bytes = serde_json::to_vec(&config).context("Failed to serialize MGM-1 config")?;
    let tokenizer_bytes = br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","vocab":{"A":0,"C":1,"G":2,"T":3,"[MASK]":4," ":5,"[CLS]":6,"[SEP]":7},"merges":[]}}"#.to_vec();

    Ok(RawModelFiles { config: config_bytes, tokenizer: tokenizer_bytes, weights })
}
