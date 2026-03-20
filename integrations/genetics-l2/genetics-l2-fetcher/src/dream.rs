use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// ── Algorithm mapping ─────────────────────────────────────────────────────────

fn algorithm_from_dream_topic(topic: &str) -> Algorithm {
    let t = topic.to_lowercase();
    if t.contains("network") || t.contains("regulat") || t.contains("pathway") {
        Algorithm::NetworkBiology
    } else if t.contains("gene") || t.contains("expression") || t.contains("transcri") {
        Algorithm::GeneExpression
    } else if t.contains("cancer") || t.contains("tumor") || t.contains("somatic") {
        Algorithm::CancerGenomics
    } else if t.contains("biomarker") || t.contains("diagnosis") {
        Algorithm::BiomarkerDiscovery
    } else if t.contains("drug") || t.contains("synergy") || t.contains("compound") {
        Algorithm::DrugDiscovery
    } else if t.contains("protein") || t.contains("structure") {
        Algorithm::ProteinFolding
    } else if t.contains("variant") || t.contains("mutation") {
        Algorithm::VariantCalling
    } else {
        Algorithm::Custom(topic.to_owned())
    }
}

// ── DreamFetcher — Synapse / dreamchallenges.org ──────────────────────────────

/// Fetches open DREAM Challenges from the Synapse REST API.
///
/// DREAM Challenges are hosted on Synapse (synapse.org).
/// Public endpoint (no auth for listing):
/// https://repo-prod.prod.sagebase.org/repo/v1/challenge?limit=20&offset=0
///
/// For task decomposition: each DREAM challenge sub-task is mapped to a
/// separate `ScientificJob` so multiple workers can tackle different subtasks
/// in parallel, with results ensembled by the coordinator.
pub struct DreamFetcher {
    http:        reqwest::Client,
    /// Optional Synapse personal access token for authenticated requests.
    /// Without a token only public challenges are visible.
    synapse_pat: Option<String>,
}

impl DreamFetcher {
    pub fn new(synapse_pat: Option<String>) -> Self {
        Self {
            http: reqwest::Client::new(),
            synapse_pat,
        }
    }

    fn auth_header(&self) -> Option<(&'static str, String)> {
        self.synapse_pat.as_ref().map(|t| ("Authorization", format!("Bearer {t}")))
    }
}

#[async_trait::async_trait]
impl SourceFetcher for DreamFetcher {
    fn name(&self) -> &str { "dream" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // ── 1. Fetch challenge list from Synapse ──────────────────────────────
        let url = "https://repo-prod.prod.sagebase.org/repo/v1/challenge\
            ?limit=20&offset=0";

        let mut req = self.http.get(url).header("Accept", "application/json");
        if let Some((k, v)) = self.auth_header() {
            req = req.header(k, v);
        }

        let resp = req.send().await.context("Synapse challenge list request")?;

        if !resp.status().is_success() {
            anyhow::bail!("Synapse API {}", resp.status());
        }

        let json: serde_json::Value = resp.json().await.context("Synapse JSON")?;

        let challenges = json["results"]
            .as_array()
            .or_else(|| json["challengeList"].as_array())
            .cloned()
            .unwrap_or_default();

        let mut jobs = Vec::new();

        for ch in &challenges {
            let id = ch["id"].as_str()
                .or_else(|| ch["challengeId"].as_str())
                .unwrap_or_default();
            let project_id = ch["projectId"].as_str().unwrap_or(id);
            let title = ch["title"].as_str().unwrap_or("DREAM Challenge");
            let topic = ch["description"].as_str()
                .or_else(|| ch["tagline"].as_str())
                .unwrap_or(title);

            if id.is_empty() { continue; }

            // ── 2. Decompose into sub-tasks ───────────────────────────────────
            // Each DREAM challenge typically has multiple sub-tasks (sc1, sc2…).
            // We probe the wiki for sub-challenge descriptions; fall back to a
            // single job if the API doesn't expose them.
            let subtasks = fetch_dream_subtasks(&self.http, project_id, self.synapse_pat.as_deref()).await;

            if subtasks.is_empty() {
                // Single-task fallback
                let dataset_root = hex::encode(
                    blake3::hash(format!("dream:{id}:main").as_bytes()).as_bytes()
                );
                let dataset_url = Some(format!(
                    "https://www.synapse.org/#!Synapse:{project_id}"
                ));
                let algorithm = algorithm_from_dream_topic(topic);
                let job = ScientificJob::new(
                    ExternalSource::Dream,
                    Some(format!("{id}:main")),
                    dataset_root,
                    dataset_url,
                    algorithm,
                    format!("DREAM: {title}"),
                    20_000_000, // 20 XEN baseline
                    86_400,     // 24h
                );
                jobs.push(job);
            } else {
                // One job per sub-task — enables parallel ensemble execution
                for (sc_idx, sc_desc) in subtasks.iter().enumerate() {
                    let sc_key = format!("{id}:sc{}", sc_idx + 1);
                    let dataset_root = hex::encode(
                        blake3::hash(format!("dream:{sc_key}").as_bytes()).as_bytes()
                    );
                    let dataset_url = Some(format!(
                        "https://www.synapse.org/#!Synapse:{project_id}/wiki/sc{}", sc_idx + 1
                    ));
                    let algorithm = algorithm_from_dream_topic(sc_desc);
                    let job = ScientificJob::new(
                        ExternalSource::Dream,
                        Some(sc_key.clone()),
                        dataset_root,
                        dataset_url,
                        algorithm,
                        format!("DREAM: {title} — Sub-Challenge {} ({sc_desc})", sc_idx + 1),
                        20_000_000,
                        86_400,
                    );
                    jobs.push(job);
                }
            }
        }

        log::info!("[dream] {} job(s) registered from {} challenge(s)",
            jobs.len(), challenges.len());
        Ok(jobs)
    }
}

// ── Sub-task discovery ────────────────────────────────────────────────────────

/// Attempt to retrieve sub-challenge names from the Synapse project wiki.
/// Returns an empty Vec if the project is private or the endpoint fails.
async fn fetch_dream_subtasks(
    http:        &reqwest::Client,
    project_id:  &str,
    pat:         Option<&str>,
) -> Vec<String> {
    let url = format!(
        "https://repo-prod.prod.sagebase.org/repo/v1/entity/{project_id}/wiki"
    );
    let mut req = http.get(&url).header("Accept", "application/json");
    if let Some(token) = pat {
        req = req.header("Authorization", format!("Bearer {token}"));
    }

    let Ok(resp) = req.send().await else { return vec![]; };
    if !resp.status().is_success() { return vec![]; }
    let Ok(json) = resp.json::<serde_json::Value>().await else { return vec![]; };

    // Wiki may contain sub-challenge sections named "Sub-Challenge 1", "SC1", etc.
    json["results"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|w| w["title"].as_str())
                .filter(|t| {
                    let tl = t.to_lowercase();
                    tl.contains("sub") || tl.starts_with("sc")
                })
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}
