use anyhow::{anyhow, Context, Result};
use reqwest::Url;
use std::time::Duration;
use tracing::{info, warn};

use super::RawModelFiles;

const HF_HUB_URL: &str = "https://huggingface.co";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Download a complete model checkpoint (config.json, tokenizer.json, weights) from Hugging Face.
/// Tries `model.safetensors` first, then falls back to `pytorch_model.bin`.
pub async fn download_model(model_id: &str) -> Result<RawModelFiles> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .context("Failed to build HTTP client")?;

    let config = download_file(&client, &model_url(model_id, "config.json")?)
        .await
        .context("Failed to download config.json")?;
    info!("Downloaded config.json for {}", model_id);

    let tokenizer = download_file(&client, &model_url(model_id, "tokenizer.json")?)
        .await
        .context("Failed to download tokenizer.json")?;
    info!("Downloaded tokenizer.json for {}", model_id);

    let mut weights: Option<Vec<u8>> = None;
    for filename in ["model.safetensors", "pytorch_model.bin"] {
        let url = model_url(model_id, filename)?;
        info!("Attempting to download model weights from {}", url);

        match download_file(&client, &url).await {
            Ok(data) => {
                info!("Downloaded {} ({:.2} MB) from {}", filename, data.len() as f64 / 1_048_576.0, url);
                weights = Some(data);
                break;
            }
            Err(e) => {
                warn!("Could not download {}: {}", url, e);
            }
        }
    }

    let weights = weights.ok_or_else(|| anyhow!("Could not download model weights for {} from Hugging Face", model_id))?;

    Ok(RawModelFiles { config, tokenizer, weights })
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
