use anyhow::{anyhow, Context, Result};
use std::time::Duration;
use tracing::{info, warn};

const HF_HUB_URL: &str = "https://huggingface.co";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(300);

/// Download the raw model weights for a Hugging Face model id.
/// Tries `model.safetensors` first, then falls back to `pytorch_model.bin`.
pub async fn download_model_weights(model_id: &str) -> Result<Vec<u8>> {
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(Duration::from_secs(30))
        .build()
        .context("Failed to build HTTP client")?;

    let encoded_id = model_id.replace('/', "%2F");

    for filename in ["model.safetensors", "pytorch_model.bin"] {
        let url = format!("{}/{}/resolve/main/{}", HF_HUB_URL, encoded_id, filename);
        info!("Attempting to download model weights from {}", url);

        match download_file(&client, &url).await {
            Ok(data) => {
                info!("Downloaded {} ({:.2} MB) from {}", filename, data.len() as f64 / 1_048_576.0, url);
                return Ok(data);
            }
            Err(e) => {
                warn!("Could not download {}: {}", url, e);
            }
        }
    }

    Err(anyhow!("Could not download model weights for {} from Hugging Face", model_id))
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
