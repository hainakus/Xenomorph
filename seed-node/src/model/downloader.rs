use anyhow::{anyhow, Context, Result};
use reqwest::Url;
use std::time::Duration;
use tracing::{info, warn};

use super::RawModelFiles;

const HF_HUB_URL: &str = "https://huggingface.co";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Download a complete model checkpoint (config.json, tokenizer.json, weights) from Hugging Face.
/// Only `model.safetensors` is supported by the miner; PyTorch `.bin` files are rejected.
pub async fn download_model(model_id: &str) -> Result<RawModelFiles> {
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
    for filename in ["model.safetensors"] {
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

    let weights =
        weights.ok_or_else(|| anyhow!("Could not download valid model.safetensors weights for {} from Hugging Face", model_id))?;

    Ok(RawModelFiles { config, tokenizer, weights })
}

/// Download a weights file and verify it is a real checkpoint, not an HTML/LFS pointer.
/// If the plain resolve URL returns a pointer, retry once with `?download=true`.
async fn download_and_validate_weights(client: &reqwest::Client, model_id: &str, filename: &str) -> Result<Vec<u8>> {
    let base_url = model_url(model_id, filename)?;

    for url in [&base_url, &format!("{}?download=true", base_url)] {
        info!("Attempting to download model weights from {}", url);
        match download_file(client, url).await {
            Ok(data) if is_valid_weights(&data) => return Ok(data),
            Ok(data) => {
                warn!("Downloaded {} but content does not look like valid weights ({} bytes); will retry if possible", url, data.len())
            }
            Err(e) => warn!("Failed to download {}: {}", url, e),
        }
    }

    Err(anyhow!("{} from {} is not a valid weights file", filename, model_id))
}

/// Heuristic validation that `data` is a real `model.safetensors` file.
/// Rejects git-lfs pointers, HTML error pages, PyTorch .bin files, and truncated data.
pub fn is_valid_weights(data: &[u8]) -> bool {
    if data.len() < 9 {
        return false;
    }

    // git-lfs pointer files start with "version https://git-lfs.github.com/spec/v1"
    if data.starts_with(b"version https://git-lfs.github.com/spec/v1") {
        return false;
    }

    // HTML error pages start with "<!DOCTYPE" or "<html".
    if data.starts_with(b"<!DOCTYPE") || data.starts_with(b"<html") || data.starts_with(b"<HTML") {
        return false;
    }

    // PyTorch zip pickle (new torch.save): starts with PK\x03\x04 or PK\x05\x06 or PK\x07\x08.
    if data.starts_with(b"PK\x03\x04") || data.starts_with(b"PK\x05\x06") || data.starts_with(b"PK\x07\x08") {
        return false;
    }

    // Old PyTorch pickle (protocol 2+): first byte is the pickle opcode 0x80.
    if data[0] == 0x80 {
        return false;
    }

    // Safetensors: first 8 bytes are a little-endian u64 header length, followed by JSON.
    let header_len = u64::from_le_bytes(data[0..8].try_into().expect("8 bytes")) as usize;
    header_len + 8 <= data.len() && header_len <= 1_000_000_000 && data[8] == b'{'
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
