use anyhow::Result;
use genetics_l2_core::{now_secs, Algorithm, DatasetCategory, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// 1000 Genomes Project GRCh38 30x phased VCFs (NYGC resequencing, 3202 samples).
// Stable public FTP URLs — one job per autosome + chrX.
// Source: https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV/
const IGSR_BASE: &str =
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/\
     1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV";

// Chromosomes to seed (subset for MVP — chr1,2,7,17,19 cover major disease genes)
const SEED_CHROMS: &[&str] = &["1", "2", "7", "17", "19", "X"];

/// Fetches VCF annotation jobs from the 1000 Genomes / IGSR public dataset.
///
/// Category: **ReferenceCohort** — gold-standard population reference panel
/// (3202 samples, 30x coverage). Workers build per-chromosome cohort indexes
/// and annotate population-level variant frequencies via Ensembl VEP.
pub struct IgsrFetcher {
    /// Optional: override chromosome list (default SEED_CHROMS).
    chroms: Vec<String>,
}

impl IgsrFetcher {
    pub fn new() -> Self {
        Self { chroms: SEED_CHROMS.iter().map(|s| s.to_string()).collect() }
    }

    pub fn with_chroms(mut self, chroms: Vec<String>) -> Self {
        self.chroms = chroms;
        self
    }
}

impl Default for IgsrFetcher {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl SourceFetcher for IgsrFetcher {
    fn name(&self) -> &str { "igsr" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        let mut jobs = Vec::new();

        for chrom in &self.chroms {
            // File pattern for SNV+INDEL phased VCF
            let filename = format!(
                "1kGP_high_coverage_Illumina.chr{chrom}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
            );
            let url = format!("{IGSR_BASE}/{filename}");

            let id = format!("igsr_grch38_chr{chrom}");
            let dataset_root = hex::encode(
                blake3::hash(format!("igsr:grch38:{chrom}:20220422").as_bytes()).as_bytes()
            );

            let description = format!(
                "1000 Genomes IGSR GRCh38 30x — chr{chrom} SNV/INDEL phased (3202 samples)"
            );

            let mut job = ScientificJob::new(
                ExternalSource::Igsr,
                Some(id),
                dataset_root,
                Some(url),
                Algorithm::CohortBuild,             // reference cohort annotation
                description,
                30_000_000, // 30 XEN — chromosome-scale phased cohort, large
                14_400,     // 4h — chr1/2 are large
            );
            job.pipeline         = Some("cohort_vcf_annotation_grch38".to_owned());
            job.reference_genome = Some("GRCh38".to_owned());
            job.deadline         = Some(now_secs() + 14 * 86_400);
            job.dataset_category = Some(DatasetCategory::ReferenceCohort);
            jobs.push(job);
        }

        log::info!("[igsr] {} chromosome job(s) seeded", jobs.len());
        Ok(jobs)
    }
}
