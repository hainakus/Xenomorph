use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

/// Fetches life-science competitions from the Kaggle API.
///
/// Requires a Kaggle API key in `username:token` format.
/// Set via `--kaggle-key` or the `KAGGLE_KEY` environment variable.
pub struct KaggleFetcher {
    api_key:           String,
    http:              reqwest::Client,
    /// If set, only this competition slug is seeded (e.g. "birdclef-2026").
    pub target_slug:   Option<String>,
}

impl KaggleFetcher {
    pub fn new(api_key: String) -> Self {
        Self { api_key, http: reqwest::Client::new(), target_slug: None }
    }

    pub fn with_competition(mut self, slug: String) -> Self {
        self.target_slug = Some(slug);
        self
    }
}

#[async_trait::async_trait]
impl SourceFetcher for KaggleFetcher {
    fn name(&self) -> &str { "kaggle" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // If targeting a specific competition, seed it directly.
        if let Some(ref slug) = self.target_slug {
            return self.seed_competition(slug).await;
        }

        // Kaggle API: GET https://www.kaggle.com/api/v1/competitions/list
        // Auth: Basic username:token (base64)
        let parts: Vec<&str> = self.api_key.splitn(2, ':').collect();
        if parts.len() != 2 {
            anyhow::bail!("KAGGLE_KEY must be 'username:token'");
        }
        let (username, token) = (parts[0], parts[1]);

        let resp = self.http
            .get("https://www.kaggle.com/api/v1/competitions/list?search=biology&sortBy=deadline&pageSize=40")
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

                // Accept genomics, bioacoustics, ecology, and wildlife competitions.
                let is_life_sci = tags.iter().any(|t| {
                    matches!(t.to_lowercase().as_str(),
                        "genomics" | "biology" | "bioinformatics" | "dna" | "protein"
                        | "genetics" | "rna" | "medicine" | "drug-discovery"
                        | "birds" | "audio" | "bioacoustics" | "ecology" | "species"
                        | "wildlife" | "conservation" | "environment")
                });
                let t_lower = title.to_lowercase();
                if !is_life_sci
                    && !t_lower.contains("genom") && !t_lower.contains("dna")
                    && !t_lower.contains("bird") && !t_lower.contains("species")
                    && !t_lower.contains("acoustic") && !t_lower.contains("bioclef")
                    && !t_lower.contains("birdclef")
                {
                    continue;
                }

                let algorithm = infer_algorithm(&title, &tags);

                // Kaggle competitions reference a dataset — we use slug as dataset_root stub
                let dataset_root = hex::encode(
                    blake3::hash(format!("kaggle:{slug}").as_bytes()).as_bytes()
                );
                let dataset_url = Some(format!("https://www.kaggle.com/c/{slug}/data"));

                // Convert USD reward to sompi (1 XEN ~ $0.001). Min 100k XEN for non-monetary competitions.
                let reward_sompi = if reward_usd > 0 { reward_usd.saturating_mul(1_000_000) } else { 100_000_000_000 };

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

    if all.contains("bird") || all.contains("acoustic") || all.contains("bioacoustic")
        || all.contains("species") || all.contains("clef") || all.contains("wildlife")
    {
        Algorithm::AcousticClassification
    } else if all.contains("protein") || all.contains("folding") {
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

// ── Direct competition seeding ─────────────────────────────────────────────────

impl KaggleFetcher {
    /// Seed a specific Kaggle competition by slug, fetching its metadata from the API.
    /// Falls back to a hardcoded stub if the API call fails (useful for testing without a key).
    async fn seed_competition(&self, slug: &str) -> Result<Vec<ScientificJob>> {
        // Try Kaggle API metadata
        let meta = self.fetch_competition_meta(slug).await;

        let (title, description, reward_usd, dataset_url) = match meta {
            Ok(m) => (
                m["title"].as_str().unwrap_or(slug).to_owned(),
                m["subtitle"].as_str().unwrap_or("Kaggle competition").to_owned(),
                m["reward"].as_str()
                    .and_then(|s| s.replace(['$', ',', ' '], "").parse::<u64>().ok())
                    .unwrap_or(0),
                Some(format!("https://www.kaggle.com/competitions/{slug}/data")),
            ),
            Err(_) => {
                // Hardcoded stub for birdclef-2026
                let is_birdclef = slug.contains("birdclef");
                (
                    if is_birdclef {
                        format!("BirdCLEF+ {}", &slug[slug.len().saturating_sub(4)..])
                    } else {
                        slug.to_owned()
                    },
                    "Acoustic species identification".to_owned(),
                    0u64,
                    Some(format!("https://www.kaggle.com/competitions/{slug}/data")),
                )
            }
        };

        let tags: Vec<String> = vec!["birds".into(), "audio".into(), "ecology".into()];
        let algorithm = infer_algorithm(&title, &tags);

        // For BirdCLEF competitions, create jobs with sample audio clips
        // The dataset_url should point to the Kaggle dataset download
        // Miners will use kagglehub to download the actual audio files
        let is_birdclef = slug.contains("birdclef");
        let dataset_url_final = if is_birdclef {
            // Point to the Kaggle dataset that can be downloaded via kagglehub
            Some(format!("kaggle://competitions/{slug}"))
        } else {
            dataset_url
        };

        let dataset_root = hex::encode(
            blake3::hash(format!("kaggle:{slug}").as_bytes()).as_bytes()
        );
        let reward_sompi = if reward_usd > 0 { reward_usd.saturating_mul(1_000_000) } else { 100_000_000_000 };
        let task_desc = format!("{title} — {description}");

        let job = ScientificJob::new(
            ExternalSource::Kaggle,
            Some(slug.to_owned()),
            dataset_root,
            dataset_url_final,
            algorithm,
            task_desc,
            reward_sompi,
            86_400,
        );
        Ok(vec![job])
    }

    async fn fetch_competition_meta(&self, slug: &str) -> Result<serde_json::Value> {
        let parts: Vec<&str> = self.api_key.splitn(2, ':').collect();
        if parts.len() != 2 { anyhow::bail!("KAGGLE_KEY must be 'username:token'"); }
        let (username, token) = (parts[0], parts[1]);
        let resp = self.http
            .get(format!("https://www.kaggle.com/api/v1/competitions/{slug}"))
            .basic_auth(username, Some(token))
            .send().await.context("Kaggle meta")?;
        if !resp.status().is_success() {
            anyhow::bail!("Kaggle API {} → {}", slug, resp.status());
        }
        resp.json().await.context("Kaggle meta JSON")
    }
}
