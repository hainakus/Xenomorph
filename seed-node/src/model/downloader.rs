use anyhow::{anyhow, Context, Result};
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
        .build()
        .context("Failed to build HTTP client")?;

    let encoded_id = model_id.replace('/', "%2F");

    let config = download_file(&client, &format!("{}/{}/resolve/main/config.json", HF_HUB_URL, encoded_id))
        .await
        .context("Failed to download config.json")?;
    info!("Downloaded config.json for {}", model_id);

    let tokenizer = download_file(&client, &format!("{}/{}/resolve/main/tokenizer.json", HF_HUB_URL, encoded_id))
        .await
        .context("Failed to download tokenizer.json")?;
    info!("Downloaded tokenizer.json for {}", model_id);

    let mut weights: Option<Vec<u8>> = None;
    for filename in ["model.safetensors", "pytorch_model.bin"] {
        let url = format!("{}/{}/resolve/main/{}", HF_HUB_URL, encoded_id, filename);
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

async fn download_file(client: &reqwest::Client, url: &str) -> Result<Vec<u8>> {
    let response = client.get(url).send().await.context("HTTP request failed")?;

    let status = response.status();
    if !status.is_success() {
        return Err(anyhow!("HTTP {} for {}", status, url));
    }

    let bytes = response.bytes().await.context("Failed to read response body")?;
    Ok(bytes.to_vec())
}
