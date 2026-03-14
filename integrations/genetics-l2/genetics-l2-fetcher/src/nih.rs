use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// ── NihFetcher — NCBI SRA public datasets ────────────────────────────────────

/// Fetches public genomics datasets from the NIH / NCBI APIs.
///
/// Uses the NCBI E-utilities API (no key required for low-volume access).
/// Endpoint: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/
pub struct NihFetcher {
    http: reqwest::Client,
}

impl NihFetcher {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

#[async_trait::async_trait]
impl SourceFetcher for NihFetcher {
    fn name(&self) -> &str { "nih" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // NCBI E-search: find recent SRA datasets tagged with "genomics"
        let search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi\
            ?db=sra&term=genomics+variant+calling&retmax=10&retmode=json&sort=recently_added";

        let resp = self.http
            .get(search_url)
            .send()
            .await
            .context("NCBI esearch request")?;

        if !resp.status().is_success() {
            anyhow::bail!("NCBI API {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await.context("NCBI JSON")?;
        let ids = json["esearchresult"]["idlist"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let mut jobs = Vec::new();
        for id_val in &ids {
            let Some(srr_id) = id_val.as_str() else { continue };

            let dataset_root = hex::encode(
                blake3::hash(format!("nih:sra:{srr_id}").as_bytes()).as_bytes()
            );
            let dataset_url = Some(format!(
                "https://www.ncbi.nlm.nih.gov/sra/{srr_id}"
            ));

            let job = ScientificJob::new(
                ExternalSource::Nih,
                Some(srr_id.to_owned()),
                dataset_root,
                dataset_url,
                Algorithm::VariantCalling,
                format!("NIH SRA variant calling — sample {srr_id}"),
                5_000_000, // 5 XEN reward
                7_200,     // 2h max
            );
            jobs.push(job);
        }

        Ok(jobs)
    }
}

// ── NihChallengeFetcher — challenges.nih.gov Prize Challenges ─────────────────

/// Maps a challenge topic string to the most specific Algorithm variant.
fn algorithm_from_topic(topic: &str) -> Algorithm {
    let t = topic.to_lowercase();
    if t.contains("drug") || t.contains("compound") || t.contains("screening") {
        Algorithm::DrugDiscovery
    } else if t.contains("cancer") || t.contains("tumor") || t.contains("oncol") {
        Algorithm::CancerGenomics
    } else if t.contains("biomarker") || t.contains("diagnostic") {
        Algorithm::BiomarkerDiscovery
    } else if t.contains("protein") || t.contains("folding") || t.contains("structure") {
        Algorithm::ProteinFolding
    } else if t.contains("network") || t.contains("pathway") || t.contains("interaction") {
        Algorithm::NetworkBiology
    } else if t.contains("gene") || t.contains("expression") || t.contains("rna") {
        Algorithm::GeneExpression
    } else if t.contains("variant") || t.contains("mutation") || t.contains("snp") {
        Algorithm::VariantCalling
    } else if t.contains("metagenom") || t.contains("microbiome") {
        Algorithm::Metagenomics
    } else if t.contains("sequence") || t.contains("alignment") {
        Algorithm::SequenceAlignment
    } else {
        Algorithm::Custom(topic.to_owned())
    }
}

/// Reward bucket: scale XEN reward proportional to the NIH prize amount.
/// $10k  →   10 XEN  (10_000_000 sompi)
/// $100k →  100 XEN (100_000_000 sompi)
/// $1M   → 1000 XEN (highest bracket)
fn reward_from_prize_usd(usd: u64) -> u64 {
    match usd {
        0..=9_999          =>    1_000_000, // < $10k   →   1 XEN stub
        10_000..=49_999    =>   10_000_000, // ~$10k    →  10 XEN
        50_000..=199_999   =>   50_000_000, // ~$100k   →  50 XEN
        200_000..=999_999  =>  200_000_000, // ~$200k   → 200 XEN
        _                  => 1_000_000_000, // $1M+    → 1000 XEN
    }
}

/// Fetches open Prize Challenges from the NIH challenge.gov API.
///
/// Public endpoint (no API key required):
/// https://api.challenge.gov/api/challenges?status=open&limit=25
pub struct NihChallengeFetcher {
    http: reqwest::Client,
}

impl NihChallengeFetcher {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

#[async_trait::async_trait]
impl SourceFetcher for NihChallengeFetcher {
    fn name(&self) -> &str { "nih_challenge" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        let url = "https://www.challenge.gov/api/challenges\
            ?status=open&agency=HHS,NIH&limit=25";

        let resp = self.http
            .get(url)
            .header("Accept", "application/json")
            .send()
            .await
            .context("challenge.gov API request")?;

        if !resp.status().is_success() {
            anyhow::bail!("challenge.gov API {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await.context("challenge.gov JSON")?;

        // API returns { "challenges": [...] } or just an array
        let challenges = json["challenges"]
            .as_array()
            .or_else(|| json.as_array())
            .cloned()
            .unwrap_or_default();

        let mut jobs = Vec::new();
        for ch in &challenges {
            let id    = ch["id"].as_str()
                .or_else(|| ch["challenge_id"].as_str())
                .unwrap_or_default();
            let title = ch["title"].as_str().unwrap_or("NIH Challenge");
            let topic = ch["tagline"].as_str()
                .or_else(|| ch["brief_description"].as_str())
                .unwrap_or(title);

            if id.is_empty() { continue; }

            let prize_usd: u64 = ch["prize_total"]
                .as_str()
                .and_then(|s| s.replace(['$', ',', ' '], "").parse().ok())
                .or_else(|| ch["prize_total"].as_u64())
                .unwrap_or(10_000);

            let dataset_root = hex::encode(
                blake3::hash(format!("nih_challenge:{id}").as_bytes()).as_bytes()
            );
            let dataset_url  = ch["external_url"].as_str()
                .map(str::to_owned)
                .or_else(|| Some(format!("https://www.challenge.gov/challenge/{id}/")));

            let algorithm   = algorithm_from_topic(topic);
            let reward      = reward_from_prize_usd(prize_usd);
            let max_time    = 86_400u64; // 24h — NIH challenges are compute-heavy

            let job = ScientificJob::new(
                ExternalSource::NihChallenge,
                Some(id.to_owned()),
                dataset_root,
                dataset_url,
                algorithm,
                format!("NIH Challenge: {title} (${prize_usd})"),
                reward,
                max_time,
            );
            jobs.push(job);
        }

        log::info!("[nih_challenge] {} open challenge(s) found", jobs.len());
        Ok(jobs)
    }
}
