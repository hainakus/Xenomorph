use anyhow::Result;
use genetics_l2_core::{now_secs, Algorithm, DatasetCategory, ExternalSource, ScientificJob};

use crate::SourceFetcher;

// Stable GRCh38 VCF datasets on NCBI SRA / dbVar used as fallback seeds.
// These are well-characterised samples used as industry-standard benchmarks.
const SEED_DATASETS: &[(&str, &str, &str)] = &[
    // (accession, description, direct-VCF-URL-or-empty)
    ("SRR6399874", "NA12878 30x GRCh38 WGS (Illumina — GIAB benchmark)",
     "https://ftp.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/analysis/NIST_SVs_Integration_v0.6/HG001_GRCh38_GIAB_highconf_CG-IllFB-IllGATKHC-Ion-10X-SOLID_CHROM1-X_v.3.3.2_highconf_nosomaticdel_noCENorHET7.vcf.gz"),
    ("SRR10903401", "NA24385 GRCh38 WGS (GIAB HG002)",
     "https://ftp.ncbi.nlm.nih.gov/giab/ftp/release/AshkenazimTrio/HG002_NA24385_son/NISTv4.2.1/GRCh38/HG002_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"),
    ("SRR10903402", "NA24631 GRCh38 WGS (GIAB HG005)",
     "https://ftp.ncbi.nlm.nih.gov/giab/ftp/release/ChineseTrio/HG005_NA24631_son/NISTv4.2.1/GRCh38/HG005_GRCh38_1_22_v4.2.1_benchmark.vcf.gz"),
];

/// Fetches GRCh38-aligned VCF datasets from NCBI SRA.
///
/// Category: **RawCompute** — raw WGS/WES sequencing data requiring
/// full variant calling pipeline (alignment → GATK/DeepVariant → VCF).
/// Highest compute intensity; benefits from GPU-accelerated alignment.
pub struct SraFetcher {
    http: reqwest::Client,
}

impl SraFetcher {
    pub fn new() -> Self {
        Self { http: reqwest::Client::new() }
    }
}

impl Default for SraFetcher {
    fn default() -> Self { Self::new() }
}

#[async_trait::async_trait]
impl SourceFetcher for SraFetcher {
    fn name(&self) -> &str { "sra" }

    async fn fetch_jobs(&self) -> Result<Vec<ScientificJob>> {
        // Try NCBI E-utilities: search SRA for GRCh38 VCF studies
        let search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi\
            ?db=sra&term=homo+sapiens+vcf+grch38+variant&retmax=8\
            &retmode=json&sort=recently_added";

        let api_ids = match self.http.get(search_url).send().await {
            Ok(r) if r.status().is_success() => {
                r.json::<serde_json::Value>().await
                    .ok()
                    .and_then(|v| v["esearchresult"]["idlist"].as_array().cloned())
                    .unwrap_or_default()
            }
            _ => {
                log::warn!("[sra] NCBI E-utilities unreachable — seeding fallback datasets");
                vec![]
            }
        };

        let mut jobs: Vec<ScientificJob> = Vec::new();

        // API results
        for id_val in &api_ids {
            let Some(uid) = id_val.as_str() else { continue };
            jobs.push(make_sra_job(uid, &format!("NCBI SRA GRCh38 VCF — {uid}"), ""));
        }

        // Fallback seeds (always include so the pipeline has real benchmark data)
        if jobs.is_empty() {
            for (acc, desc, vcf_url) in SEED_DATASETS {
                jobs.push(make_sra_job(acc, desc, vcf_url));
            }
        }

        log::info!("[sra] {} job(s) created", jobs.len());
        Ok(jobs)
    }
}

fn make_sra_job(accession: &str, description: &str, vcf_url: &str) -> ScientificJob {
    let dataset_root = hex::encode(
        blake3::hash(format!("sra:grch38:{accession}").as_bytes()).as_bytes()
    );
    let dataset_url = if vcf_url.is_empty() {
        Some(format!("https://www.ncbi.nlm.nih.gov/sra/{accession}"))
    } else {
        Some(vcf_url.to_owned())
    };

    let mut job = ScientificJob::new(
        ExternalSource::Sra,
        Some(accession.to_owned()),
        dataset_root,
        dataset_url,
        Algorithm::VariantCalling,          // raw WGS → variant calling
        description.to_owned(),
        50_000_000, // 50 XEN — highest reward: raw compute is most GPU/CPU intensive
        14_400,     // 4h — WGS alignment + calling is slow
    );
    job.pipeline         = Some("variant_calling_grch38".to_owned());
    job.reference_genome = Some("GRCh38".to_owned());
    job.deadline         = Some(now_secs() + 7 * 86_400);
    job.dataset_category = Some(DatasetCategory::RawCompute);
    job
}
