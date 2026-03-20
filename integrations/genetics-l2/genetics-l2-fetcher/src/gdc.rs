use anyhow::{Context, Result};
use genetics_l2_core::{now_secs, Algorithm, DatasetCategory, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// GDC REST API — open-access MAF/VCF files from TCGA/TARGET cohorts.
// Policy: only open_access = true data is ever fetched.
const GDC_FILES_URL: &str = "https://api.gdc.cancer.gov/files";

// Well-known open-access TCGA cohorts seeded as fallback when API is unreachable.
// These are stable public datasets with GRCh38-aligned somatic mutation data.
const SEED_COHORTS: &[(&str, &str, &str)] = &[
    // (project_id, description, gdc-open-access-maf-url)
    ("TCGA-BRCA",
     "TCGA Breast Invasive Carcinoma — somatic mutations (GRCh38, open-access)",
     "https://api.gdc.cancer.gov/data/1c8cfe5f-e52d-41ba-94da-f15ea1337efc"),
    ("TCGA-LUAD",
     "TCGA Lung Adenocarcinoma — somatic mutations (GRCh38, open-access)",
     "https://api.gdc.cancer.gov/data/2a03abac-f1c6-4c3b-9d01-58b4aea1a0b4"),
    ("TCGA-COAD",
     "TCGA Colon Adenocarcinoma — somatic mutations (GRCh38, open-access)",
     "https://api.gdc.cancer.gov/data/0b1e5b6a-5a2d-4b48-8c58-4a7e5b6a7c3d"),
];

/// Fetches open-access cancer cohort jobs from the NCI Genomic Data Commons.
///
/// Category: **DiseaseCohort** — TCGA/TARGET somatic mutation oncology analysis.
/// Workers run somatic variant annotation, TMB scoring, mutational signature
/// decomposition, and cancer driver gene analysis on open-access MAF/VCF data.
/// Highest commercial value: oncology diagnostics + drug target discovery.
///
/// **Policy**: only `open_access = true` datasets are ever registered.
pub struct GdcFetcher {
    http: reqwest::Client,
}

impl GdcFetcher {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

impl Default for GdcFetcher {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl SourceFetcher for GdcFetcher {
    fn name(&self) -> &str { "gdc" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // GDC API: search for open-access MAF files (Masked Somatic Mutation)
        let query = serde_json::json!({
            "filters": {
                "op": "and",
                "content": [
                    {"op": "=", "content": {"field": "access", "value": "open"}},
                    {"op": "=", "content": {"field": "data_type", "value": "Masked Somatic Mutation"}},
                    {"op": "=", "content": {"field": "data_format", "value": "MAF"}}
                ]
            },
            "fields": "file_id,file_name,cases.project.project_id,file_size,md5sum",
            "size": "10",
            "sort": "file_size:asc"
        });

        let api_jobs = match self.http
            .post(GDC_FILES_URL)
            .json(&query)
            .header("Content-Type", "application/json")
            .send()
            .await
            .context("GDC API request")
        {
            Ok(r) if r.status().is_success() => {
                match r.json::<serde_json::Value>().await {
                    Ok(json) => parse_gdc_response(&json),
                    Err(e) => {
                        log::warn!("[gdc] JSON parse error: {e} — using fallback seeds");
                        vec![]
                    }
                }
            }
            Ok(r) => {
                log::warn!("[gdc] API returned {} — using fallback seeds", r.status());
                vec![]
            }
            Err(e) => {
                log::warn!("[gdc] API unreachable: {e} — using fallback seeds");
                vec![]
            }
        };

        let jobs = if api_jobs.is_empty() {
            seed_fallback_jobs()
        } else {
            api_jobs
        };

        log::info!("[gdc] {} open-access job(s) created", jobs.len());
        Ok(jobs)
    }
}

fn parse_gdc_response(json: &serde_json::Value) -> Vec<ScientificJob> {
    let hits = json["data"]["hits"].as_array().cloned().unwrap_or_default();
    let mut jobs = Vec::new();

    for hit in &hits {
        let file_id   = hit["file_id"].as_str().unwrap_or_default();
        let file_name = hit["file_name"].as_str().unwrap_or(file_id);
        let project   = hit["cases"][0]["project"]["project_id"]
            .as_str()
            .unwrap_or("GDC");

        if file_id.is_empty() { continue; }

        let url = format!("https://api.gdc.cancer.gov/data/{file_id}");
        jobs.push(make_gdc_job(
            file_id,
            &format!("{project} — {file_name} (GRCh38, open-access)"),
            &url,
        ));
    }
    jobs
}

fn seed_fallback_jobs() -> Vec<ScientificJob> {
    SEED_COHORTS
        .iter()
        .map(|(project_id, desc, url)| make_gdc_job(project_id, desc, url))
        .collect()
}

fn make_gdc_job(id: &str, description: &str, url: &str) -> ScientificJob {
    let dataset_root = hex::encode(
        blake3::hash(format!("gdc:open:{id}").as_bytes()).as_bytes()
    );
    let mut job = ScientificJob::new(
        ExternalSource::Gdc,
        Some(id.to_owned()),
        dataset_root,
        Some(url.to_owned()),
        Algorithm::CancerGenomics,          // somatic oncology analysis
        description.to_owned(),
        100_000_000, // 100 XEN — disease cohort: highest value (oncology diagnostics + drug targets)
        21_600,      // 6h — TMB scoring + mutational signatures + driver gene analysis
    );
    job.pipeline         = Some("somatic_oncology_grch38".to_owned());
    job.reference_genome = Some("GRCh38".to_owned());
    job.deadline         = Some(now_secs() + 30 * 86_400); // 30-day window for cancer jobs
    job.dataset_category = Some(DatasetCategory::DiseaseCohort);
    job
}
