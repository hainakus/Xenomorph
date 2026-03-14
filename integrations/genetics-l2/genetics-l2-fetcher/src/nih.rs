use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

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
