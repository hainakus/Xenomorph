use anyhow::Result;
use genetics_l2_core::{now_secs, Algorithm, DatasetCategory, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// gnomAD v4.1 exome VCFs — Google Cloud public bucket.
// One job per seeded chromosome — workers annotate + compute allele freq metrics.
// Source: https://gnomad.broadinstitute.org/downloads#v4-variants
const GNOMAD_BASE: &str =
    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes";

// Seed a representative set of chromosomes covering major disease genes.
// BRCA1/2 → chr17/13, TP53/KRAS → chr17/12, CFTR → chr7, APOE/PCSK9 → chr19/1
const SEED_CHROMS: &[&str] = &["1", "7", "12", "13", "17", "19"];

/// Fetches variant frequency annotation jobs from gnomAD v4.1.
///
/// Category: **AnnotationLayer** — population allele frequency enrichment.
/// Workers join input VCFs against gnomAD exome AF tables and compute
/// per-gene frequency-weighted scores. Lightweight compute; high data value.
pub struct GnomadFetcher {
    chroms: Vec<String>,
}

impl GnomadFetcher {
    pub fn new() -> Self {
        Self { chroms: SEED_CHROMS.iter().map(|s| s.to_string()).collect() }
    }

    pub fn with_chroms(mut self, chroms: Vec<String>) -> Self {
        self.chroms = chroms;
        self
    }
}

impl Default for GnomadFetcher {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl SourceFetcher for GnomadFetcher {
    fn name(&self) -> &str { "gnomad" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        let mut jobs = Vec::new();

        for chrom in &self.chroms {
            // gnomAD v4.1 exome VCF filename pattern
            let filename = format!(
                "gnomad.exomes.v4.1.sites.chr{chrom}.vcf.bgz"
            );
            let url = format!("{GNOMAD_BASE}/{filename}");

            let id = format!("gnomad_v4.1_exomes_chr{chrom}");
            let dataset_root = hex::encode(
                blake3::hash(format!("gnomad:v4.1:exomes:chr{chrom}").as_bytes()).as_bytes()
            );

            let description = format!(
                "gnomAD v4.1 exomes — chr{chrom} allele frequency annotation (GRCh38)"
            );

            let mut job = ScientificJob::new(
                ExternalSource::Gnomad,
                Some(id),
                dataset_root,
                Some(url),
                Algorithm::FrequencyAnnotation,     // AF enrichment layer
                description,
                15_000_000, // 15 XEN — annotation layer: lightweight compute, high data value
                7_200,      // 2h — AF lookup is faster than full calling
            );
            job.pipeline         = Some("frequency_enrichment_grch38".to_owned());
            job.reference_genome = Some("GRCh38".to_owned());
            job.deadline         = Some(now_secs() + 14 * 86_400);
            job.dataset_category = Some(DatasetCategory::AnnotationLayer);
            jobs.push(job);
        }

        log::info!("[gnomad] {} chromosome job(s) seeded", jobs.len());
        Ok(jobs)
    }
}
