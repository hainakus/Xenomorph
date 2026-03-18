use anyhow::Result;
use genetics_l2_core::{now_secs, Algorithm, DatasetCategory, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// ClinVar weekly VCF releases (GRCh38) — NCBI public FTP.
// These are stable, versioned snapshots synced weekly by NCBI.
// Source: https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/
const CLINVAR_BASE: &str = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38";

/// Fetches clinical variant annotation jobs from NCBI ClinVar.
///
/// Category: **AnnotationLayer** — clinical significance enrichment.
/// Workers join input VCFs against ClinVar classification tables and output
/// per-gene pathogenicity scores + clinical significance summaries.
/// Lightweight compute; highest commercial data value (diagnostic relevance).
pub struct ClinvarFetcher;

impl ClinvarFetcher {
    pub fn new() -> Self { Self }
}

impl Default for ClinvarFetcher {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl SourceFetcher for ClinvarFetcher {
    fn name(&self) -> &str { "clinvar" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // Seed two complementary ClinVar datasets:
        // 1. Full weekly VCF (all variants + all significance labels)
        // 2. Pathogenic-only subset (high-confidence clinical data)
        let datasets: &[(&str, &str, &str)] = &[
            (
                "clinvar_weekly_grch38",
                "ClinVar weekly VCF — all variants GRCh38 (clinical annotation)",
                &format!("{CLINVAR_BASE}/clinvar.vcf.gz"),
            ),
            (
                "clinvar_pathogenic_grch38",
                "ClinVar GRCh38 — pathogenic + likely_pathogenic variants only",
                &format!("{CLINVAR_BASE}/clinvar_papu.vcf.gz"),
            ),
        ];

        let now = now_secs();
        // Compute weekly epoch so jobs refresh every 7 days (week number in year)
        let week_epoch = now / (7 * 86_400);

        let jobs: Vec<ScientificJob> = datasets.iter().map(|(id, desc, url)| {
            // Include week epoch in dataset_root so it refreshes weekly
            let versioned_id = format!("{id}_w{week_epoch}");
            let dataset_root = hex::encode(
                blake3::hash(format!("clinvar:grch38:{versioned_id}").as_bytes()).as_bytes()
            );

            let mut job = ScientificJob::new(
                ExternalSource::ClinVar,
                Some(versioned_id),
                dataset_root,
                Some(url.to_string()),
                Algorithm::ClinicalAnnotation,      // clinical significance layer
                desc.to_string(),
                25_000_000, // 25 XEN — annotation layer: high commercial value (diagnostics)
                3_600,      // 1h — ClinVar VCF is ~250MB; classification join is fast
            );
            job.pipeline         = Some("clinical_significance_grch38".to_owned());
            job.reference_genome = Some("GRCh38".to_owned());
            job.deadline         = Some(now + 7 * 86_400); // expires in 1 week (next snapshot)
            job.dataset_category = Some(DatasetCategory::AnnotationLayer);
            job
        }).collect();

        log::info!("[clinvar] {} job(s) seeded (week epoch {week_epoch})", jobs.len());
        Ok(jobs)
    }
}
