use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob, now_secs};

use crate::SourceFetcher;

/// Fetches genomics competitions from the Kaggle API.
///
/// Requires a Kaggle API key in `username:token` format.
/// Set via `--kaggle-key` or the `KAGGLE_KEY` environment variable.
pub struct KaggleFetcher {
    api_key: String,
    http:    reqwest::Client,
}

impl KaggleFetcher {
    pub fn new(api_key: String) -> Self {
        Self { api_key, http: reqwest::Client::new() }
    }
}

#[async_trait::async_trait]
impl SourceFetcher for KaggleFetcher {
    fn name(&self) -> &str { "kaggle" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // Kaggle API: GET https://www.kaggle.com/api/v1/competitions/list
        // Auth: Basic username:token (base64)
        let parts: Vec<&str> = self.api_key.splitn(2, ':').collect();
        if parts.len() != 2 {
            anyhow::bail!("KAGGLE_KEY must be 'username:token'");
        }
        let (username, token) = (parts[0], parts[1]);

        let resp = self.http
            .get("https://www.kaggle.com/api/v1/competitions/list?search=genomics&sortBy=deadline&pageSize=20")
            .basic_auth(username, Some(token))
            .send()
            .await
            .context("Kaggle API request")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body   = resp.text().await.unwrap_or_default();
            anyhow::bail!("Kaggle API {status}: {body}");
        }

        let competitions: serde_json::Value = resp.json().await.context("Kaggle JSON")?;
        let mut jobs = Vec::new();

        if let Some(list) = competitions.as_array() {
            for comp in list {
                let title       = comp["title"].as_str().unwrap_or("").to_owned();
                let slug        = comp["ref"].as_str().unwrap_or("").to_owned();
                let reward_usd  = comp["reward"].as_str()
                    .and_then(|s| s.replace(['$', ',', ' '], "").parse::<u64>().ok())
                    .unwrap_or(0);
                let tags: Vec<String> = comp["tags"]
                    .as_array()
                    .map(|a| a.iter().filter_map(|t| t.as_str().map(str::to_owned)).collect())
                    .unwrap_or_default();

                // Only pick competitions tagged with genomics/biology
                let is_bio = tags.iter().any(|t| {
                    matches!(t.to_lowercase().as_str(),
                        "genomics" | "biology" | "bioinformatics" | "dna" | "protein"
                        | "genetics" | "rna" | "medicine" | "drug-discovery")
                });
                if !is_bio && !title.to_lowercase().contains("genom")
                    && !title.to_lowercase().contains("dna") {
                    continue;
                }

                let algorithm = infer_algorithm(&title, &tags);

                // Kaggle competitions reference a dataset — we use slug as dataset_root stub
                let dataset_root = hex::encode(
                    blake3::hash(format!("kaggle:{slug}").as_bytes()).as_bytes()
                );
                let dataset_url = Some(format!("https://www.kaggle.com/c/{slug}/data"));

                // Convert USD reward to sompi approximation (1 XEN ~ $0.001)
                let reward_sompi = reward_usd.saturating_mul(1_000_000);

                let job = ScientificJob::new(
                    ExternalSource::Kaggle,
                    Some(slug),
                    dataset_root,
                    dataset_url,
                    algorithm,
                    title,
                    reward_sompi,
                    86_400, // 24h default
                );
                jobs.push(job);
            }
        }

        Ok(jobs)
    }
}

fn infer_algorithm(title: &str, tags: &[String]) -> Algorithm {
    let t = title.to_lowercase();
    let combined = tags.join(" ").to_lowercase();
    let all = format!("{t} {combined}");

    if all.contains("protein") || all.contains("folding") {
        Algorithm::ProteinFolding
    } else if all.contains("variant") || all.contains("snp") || all.contains("mutation") {
        Algorithm::VariantCalling
    } else if all.contains("assembly") {
        Algorithm::GenomeAssembly
    } else if all.contains("rna") || all.contains("expression") || all.contains("transcri") {
        Algorithm::RnaExpression
    } else if all.contains("metagenom") || all.contains("microbiome") {
        Algorithm::Metagenomics
    } else if all.contains("alignment") || all.contains("mapping") {
        Algorithm::SequenceAlignment
    } else if all.contains("docking") || all.contains("ligand") {
        Algorithm::MolecularDocking
    } else {
        Algorithm::SequenceAlignment
    }
}
