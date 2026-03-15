use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Strip HTML tags and decode common entities for clean plain-text slugs.
fn strip_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<'            => in_tag = true,
            '>' if in_tag  => { in_tag = false; out.push(' '); }
            _ if !in_tag   => out.push(c),
            _              => {}
        }
    }
    // collapse multiple spaces
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

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
        // Build a safe slug: only alphanumeric words, max 3, joined by "_"
        let slug = t
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .take(3)
            .collect::<Vec<_>>()
            .join("_");
        Algorithm::Custom(if slug.is_empty() { "research".to_owned() } else { slug })
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
        // Europe PubMed Central (EuropePMC) REST API — EBI-hosted, always public.
        // Returns EC-funded biomedical research publications and datasets.
        // Docs: https://europepmc.org/RestfulWebService
        let url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search\
            ?query=GRANT_AGENCY:%22European+Commission%22+%28genomics+OR+health+OR+biotechnology%29\
            &format=json&pageSize=25&resultType=core\
            &sort=CITED+desc";

        let resp = self.http
            .get(url)
            .header("Accept", "application/json")
            .send()
            .await
            .context("EuropePMC API request")?;

        if !resp.status().is_success() {
            anyhow::bail!("EuropePMC API {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await.context("EuropePMC JSON")?;

        // EuropePMC: { "hitCount": N, "resultList": { "result": [...] } }
        let results = json["resultList"]["result"]
            .as_array()
            .cloned()
            .unwrap_or_default();

        let mut jobs = Vec::new();

        for rec in &results {
            let id    = rec["id"].as_str().unwrap_or_default();
            let title = &strip_html(rec["title"].as_str().unwrap_or("Horizon Research"));
            let desc  = &strip_html(rec["abstractText"].as_str().unwrap_or(title));

            if id.is_empty() { continue; }

            // Grant ID for reward scaling — use grantsList if present
            let grant_id = rec["grantsList"]["grant"]
                .as_array()
                .and_then(|g| g.first())
                .and_then(|g| g["grantId"].as_str())
                .unwrap_or("unknown");

            let dataset_root = hex::encode(
                blake3::hash(format!("horizon:epmc:{id}").as_bytes()).as_bytes()
            );
            let dataset_url = Some(format!(
                "https://europepmc.org/article/{source}/{id}",
                source = rec["source"].as_str().unwrap_or("MED"),
                id     = id
            ));

            let algorithm = algorithm_from_horizon_topic(desc);
            // Horizon prizes €1M–€5M → use max bracket as reward signal
            let reward = reward_from_ec_contribution_eur(1_000_000);

            let job = ScientificJob::new(
                ExternalSource::HorizonPrize,
                Some(format!("{id}:{grant_id}")),
                dataset_root,
                dataset_url,
                algorithm,
                format!("Horizon (EuropePMC): {title}"),
                reward,
                172_800,
            );
            jobs.push(job);
        }

        log::info!("[horizon_prize] {} project(s) found", jobs.len());
        Ok(jobs)
    }
}
