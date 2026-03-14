use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// ── Algorithm mapping ─────────────────────────────────────────────────────────

fn algorithm_from_horizon_topic(text: &str) -> Algorithm {
    let t = text.to_lowercase();
    if t.contains("digital") || t.contains("e-health") || t.contains("ehealth") || t.contains("data analyt") {
        Algorithm::DigitalHealth
    } else if t.contains("biotech") || t.contains("synthetic biology") || t.contains("ferment") || t.contains("cell engineer") {
        Algorithm::Biotechnology
    } else if t.contains("cancer") || t.contains("tumour") || t.contains("tumor") || t.contains("oncol") {
        Algorithm::CancerGenomics
    } else if t.contains("drug") || t.contains("compound") || t.contains("therapeut") {
        Algorithm::DrugDiscovery
    } else if t.contains("protein") || t.contains("folding") || t.contains("structure") {
        Algorithm::ProteinFolding
    } else if t.contains("biomarker") || t.contains("diagnostic") {
        Algorithm::BiomarkerDiscovery
    } else if t.contains("network") || t.contains("pathway") || t.contains("regulat") {
        Algorithm::NetworkBiology
    } else if t.contains("gene") || t.contains("genomic") || t.contains("sequenc") {
        Algorithm::GeneExpression
    } else if t.contains("variant") || t.contains("mutation") {
        Algorithm::VariantCalling
    } else {
        Algorithm::Custom(t.split_whitespace().take(3).collect::<Vec<_>>().join("_"))
    }
}

/// Map EC contribution (€) to XEN reward in sompi.
/// Horizon prizes: €1M–€5M → 1000–5000 XEN
fn reward_from_ec_contribution_eur(eur: u64) -> u64 {
    match eur {
        0..=499_999         =>     500_000_000, // < €500k  →   500 XEN
        500_000..=1_499_999 =>   1_000_000_000, // ~€1M     →  1000 XEN
        1_500_000..=2_999_999 => 2_000_000_000, // ~€2M     →  2000 XEN
        3_000_000..=4_999_999 => 3_000_000_000, // ~€3-5M   →  3000 XEN
        _                   =>   5_000_000_000, // €5M+     →  5000 XEN (max)
    }
}

// ── HorizonFetcher ────────────────────────────────────────────────────────────

/// Fetches EU Horizon Prize Challenges from the CORDIS REST API.
///
/// CORDIS (Community Research and Development Information Service) is the
/// primary public database for EU-funded R&D projects and prizes.
/// Endpoint (no API key required):
/// https://cordis.europa.eu/api/search/projects?q=...&format=json
///
/// Prize areas covered: medicine, genetics, digital health, biotechnology.
/// Prize range: €1M – €5M per challenge.
pub struct HorizonFetcher {
    http: reqwest::Client,
}

impl HorizonFetcher {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

#[async_trait::async_trait]
impl SourceFetcher for HorizonFetcher {
    fn name(&self) -> &str { "horizon_prize" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // Search CORDIS for Horizon prize projects in biomedical / genetics areas
        let url = "https://cordis.europa.eu/api/search/projects\
            ?q=horizon+prize+health+genetics+biotechnology\
            &p=1&n=25&srt=ecMaxContribution:desc&format=json";

        let resp = self.http
            .get(url)
            .header("Accept", "application/json")
            .send()
            .await
            .context("CORDIS API request")?;

        if !resp.status().is_success() {
            anyhow::bail!("CORDIS API {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await.context("CORDIS JSON")?;

        // CORDIS JSON: { "results": { "project": [...] } }  or  { "results": [...] }
        let projects = json["results"]["project"]
            .as_array()
            .or_else(|| json["results"].as_array())
            .cloned()
            .unwrap_or_default();

        let mut jobs = Vec::new();

        for proj in &projects {
            let id    = proj["id"].as_str().unwrap_or_default();
            let title = proj["title"].as_str().unwrap_or("Horizon Prize");
            let topic = proj["teaser"].as_str()
                .or_else(|| proj["objective"].as_str())
                .unwrap_or(title);

            if id.is_empty() { continue; }

            // Parse EC contribution — CORDIS returns it as a string or number
            let ec_eur: u64 = proj["ecMaxContribution"]
                .as_str()
                .and_then(|s| s.replace([',', '.', ' ', '€'], "").parse().ok())
                .or_else(|| proj["ecMaxContribution"].as_f64().map(|f| f as u64))
                .or_else(|| proj["totalCost"].as_f64().map(|f| f as u64))
                .unwrap_or(1_000_000); // assume €1M if not specified

            let dataset_root = hex::encode(
                blake3::hash(format!("horizon:{id}").as_bytes()).as_bytes()
            );
            let dataset_url = Some(format!(
                "https://cordis.europa.eu/project/id/{id}"
            ));

            let algorithm = algorithm_from_horizon_topic(topic);
            let reward    = reward_from_ec_contribution_eur(ec_eur);
            let max_time  = 172_800u64; // 48h — Horizon prizes are complex

            let job = ScientificJob::new(
                ExternalSource::HorizonPrize,
                Some(id.to_owned()),
                dataset_root,
                dataset_url,
                algorithm,
                format!("Horizon Prize: {title} (€{ec_eur})"),
                reward,
                max_time,
            );
            jobs.push(job);
        }

        log::info!("[horizon_prize] {} project(s) found", jobs.len());
        Ok(jobs)
    }
}
