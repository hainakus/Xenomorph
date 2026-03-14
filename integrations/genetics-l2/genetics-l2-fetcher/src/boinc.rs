use anyhow::{Context, Result};
use genetics_l2_core::{Algorithm, ExternalSource, ScientificJob};

use crate::SourceFetcher;

/// Fetches tasks from a BOINC project server.
///
/// BOINC projects expose a scheduler XML at `/scheduler_reply`.
/// This fetcher reads the project info endpoint to discover available workunits.
pub struct BoincFetcher {
    project_url: String,
    http:        reqwest::Client,
}

impl BoincFetcher {
    pub fn new(project_url: String) -> Self {
        Self { project_url, http: reqwest::Client::new() }
    }
}

#[async_trait::async_trait]
impl SourceFetcher for BoincFetcher {
    fn name(&self) -> &str { "boinc" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // BOINC project info endpoint returns XML with project statistics.
        let info_url = format!("{}/project_info.php", self.project_url.trim_end_matches('/'));

        let resp = self.http
            .get(&info_url)
            .send()
            .await
            .context("BOINC project_info request")?;

        if !resp.status().is_success() {
            anyhow::bail!("BOINC project_info {} → {}", info_url, resp.status());
        }

        let body = resp.text().await.context("BOINC response body")?;

        // Minimal XML parse: extract <name> and <wu_name> fields
        let project_name = extract_xml_tag(&body, "name")
            .unwrap_or_else(|| "boinc-project".to_owned());

        // Generate a representative job for this BOINC project.
        // In production, the fetcher would contact the scheduler to list available workunits.
        let dataset_root = hex::encode(
            blake3::hash(format!("boinc:{}", self.project_url).as_bytes()).as_bytes()
        );

        let job = ScientificJob::new(
            ExternalSource::Boinc,
            Some(self.project_url.clone()),
            dataset_root,
            Some(self.project_url.clone()),
            Algorithm::SequenceAlignment,
            format!("BOINC workunit — {project_name}"),
            1_000_000, // 1 XEN reward
            3_600,     // 1h max
        );

        Ok(vec![job])
    }
}

fn extract_xml_tag(xml: &str, tag: &str) -> Option<String> {
    let open  = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end   = xml[start..].find(&close)?;
    Some(xml[start..start + end].trim().to_owned())
}
