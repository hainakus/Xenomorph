use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ── External source ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalSource {
    Kaggle,
    /// NCBI SRA / public genomics datasets.
    Nih,
    /// NIH Prize Challenges (challenges.nih.gov).
    NihChallenge,
    Boinc,
    /// DREAM Challenges (synapse.org / dreamchallenges.org).
    Dream,
    /// EU Horizon Prize Challenges (cordis.europa.eu / EIC).
    HorizonPrize,
    BioContest,
    /// NCBI SRA — raw sequencing / VCF datasets (GRCh38-focused).
    Sra,
    /// IGSR / 1000 Genomes Project (GRCh38 30x phased VCFs).
    Igsr,
    /// Genome Aggregation Database — population variant frequencies.
    Gnomad,
    /// NCI Genomic Data Commons — open-access cancer cohorts (TCGA/TARGET).
    Gdc,
    /// NCBI ClinVar — clinically classified human variant VCFs.
    ClinVar,
    Custom(String),
}

impl std::fmt::Display for ExternalSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kaggle       => write!(f, "kaggle"),
            Self::Nih          => write!(f, "nih"),
            Self::NihChallenge => write!(f, "nih_challenge"),
            Self::Boinc        => write!(f, "boinc"),
            Self::Dream        => write!(f, "dream"),
            Self::HorizonPrize => write!(f, "horizon_prize"),
            Self::BioContest   => write!(f, "bio_contest"),
            Self::Sra          => write!(f, "sra"),
            Self::Igsr         => write!(f, "igsr"),
            Self::Gnomad       => write!(f, "gnomad"),
            Self::Gdc          => write!(f, "gdc"),
            Self::ClinVar      => write!(f, "clinvar"),
            Self::Custom(s)    => write!(f, "{s}"),
        }
    }
}

// ── Dataset category (Proof-of-Useful-Work classification) ───────────────────

/// Classifies a job by the nature of its genomic dataset.
///
/// Each category has a distinct compute profile, reward scale, and
/// pipeline identifier — enabling workers to self-select by capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetCategory {
    /// Raw WGS/WES sequencing data → variant calling from FASTQ/BAM.
    /// Highest compute intensity; GPU-accelerated alignment + calling.
    RawCompute,
    /// Reference population cohort VCFs (1000 Genomes, gnomAD population).
    /// Large-scale, deterministic annotation against reference panels.
    ReferenceCohort,
    /// Annotation enrichment layers — frequency (gnomAD) or clinical (ClinVar) joins.
    /// Lightweight compute; high scientific and commercial data value.
    AnnotationLayer,
    /// Disease cohort somatic mutation data (TCGA/GDC cancer cohorts).
    /// High compute + highest commercial value; oncology-grade analysis.
    DiseaseCohort,
}

impl std::fmt::Display for DatasetCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RawCompute     => write!(f, "raw_compute"),
            Self::ReferenceCohort => write!(f, "reference_cohort"),
            Self::AnnotationLayer => write!(f, "annotation_layer"),
            Self::DiseaseCohort   => write!(f, "disease_cohort"),
        }
    }
}

// ── Algorithm ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Algorithm {
    SequenceAlignment,
    SmithWaterman,
    NeedlemanWunsch,
    VariantCalling,
    GenomeAssembly,
    ProteinFolding,
    RnaExpression,
    Metagenomics,
    MolecularDocking,
    /// Acoustic species identification from audio clips (BirdCLEF, bioacoustics).
    AcousticClassification,
    /// Computational drug discovery / virtual screening (NIH, DrivenData).
    DrugDiscovery,
    /// Cancer genomics — somatic mutation / CNV / fusion analysis.
    CancerGenomics,
    /// Biomarker discovery from omics data.
    BiomarkerDiscovery,
    /// Gene regulatory network inference (DREAM Challenges).
    NetworkBiology,
    /// Bulk / single-cell gene expression prediction (DREAM, GTEx).
    GeneExpression,
    /// Digital health / e-health / health data analytics (Horizon, WHO).
    DigitalHealth,
    /// Biotechnology — synthetic biology, cell engineering, fermentation.
    Biotechnology,
    /// VCF normalization + Ensembl VEP annotation (GRCh38).
    VcfAnnotation,
    /// Cohort building — grouping variants by population / chromosome.
    CohortBuild,
    /// Clinical significance annotation — ClinVar join + pathogenicity scoring.
    ClinicalAnnotation,
    /// Allele frequency enrichment — gnomAD / population frequency join.
    FrequencyAnnotation,
    Custom(String),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceAlignment      => write!(f, "sequence_alignment"),
            Self::SmithWaterman          => write!(f, "smith_waterman"),
            Self::NeedlemanWunsch        => write!(f, "needleman_wunsch"),
            Self::VariantCalling         => write!(f, "variant_calling"),
            Self::GenomeAssembly         => write!(f, "genome_assembly"),
            Self::ProteinFolding         => write!(f, "protein_folding"),
            Self::RnaExpression          => write!(f, "rna_expression"),
            Self::Metagenomics           => write!(f, "metagenomics"),
            Self::MolecularDocking       => write!(f, "molecular_docking"),
            Self::AcousticClassification => write!(f, "acoustic_classification"),
            Self::DrugDiscovery          => write!(f, "drug_discovery"),
            Self::CancerGenomics         => write!(f, "cancer_genomics"),
            Self::BiomarkerDiscovery     => write!(f, "biomarker_discovery"),
            Self::NetworkBiology         => write!(f, "network_biology"),
            Self::GeneExpression         => write!(f, "gene_expression"),
            Self::DigitalHealth          => write!(f, "digital_health"),
            Self::Biotechnology          => write!(f, "biotechnology"),
            Self::VcfAnnotation          => write!(f, "vcf_annotation"),
            Self::CohortBuild            => write!(f, "cohort_build"),
            Self::ClinicalAnnotation     => write!(f, "clinical_annotation"),
            Self::FrequencyAnnotation    => write!(f, "frequency_annotation"),
            Self::Custom(s)              => write!(f, "{s}"),
        }
    }
}

// ── Job status ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    #[default]
    Open,
    Claimed,
    Completed,
    Validated,
    Settled,
    Failed,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Open      => write!(f, "open"),
            Self::Claimed   => write!(f, "claimed"),
            Self::Completed => write!(f, "completed"),
            Self::Validated => write!(f, "validated"),
            Self::Settled   => write!(f, "settled"),
            Self::Failed    => write!(f, "failed"),
        }
    }
}

// ── ScientificJob ─────────────────────────────────────────────────────────────

/// A compute job posted to the genetics-l2 coordinator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScientificJob {
    pub job_id:           String,
    pub source:           ExternalSource,
    /// External competition / dataset reference (e.g. Kaggle slug).
    pub external_ref:     Option<String>,
    /// BLAKE3 Merkle root of the input dataset (from dataset-freeze).
    pub dataset_root:     String,
    /// URL where the input dataset can be downloaded.
    pub dataset_url:      Option<String>,
    pub algorithm:        Algorithm,
    /// Human-readable task description.
    pub task_description: String,
    /// Reward in sompi paid on successful validation.
    pub reward_sompi:     u64,
    /// Max wall-clock execution time in seconds.
    pub max_time_secs:    u64,
    pub status:           JobStatus,
    /// Worker that claimed this job (pubkey hex).
    pub claimed_by:       Option<String>,
    pub created_at:       u64,
    pub claimed_at:       Option<u64>,
    pub completed_at:     Option<u64>,

    // ── Determinism manifest fields (GENOME_MINING_SPEC) ──────────────────────
    /// Pipeline identifier, e.g. "variant_annotation_grch38".
    pub pipeline:         Option<String>,
    /// BLAKE3/SHA256 of the pipeline script served by coordinator.
    pub pipeline_hash:    Option<String>,
    /// Reference genome identifier, e.g. "GRCh38".
    pub reference_genome: Option<String>,
    /// BLAKE3/SHA256 of the pinned reference genome file.
    pub reference_hash:   Option<String>,
    /// BLAKE3/SHA256 of the container image (Docker/Singularity digest).
    pub container_hash:   Option<String>,
    /// BLAKE3/SHA256 of the pipeline config / parameter file.
    pub config_hash:      Option<String>,
    /// Unix timestamp after which job expires.
    pub deadline:         Option<u64>,
    /// PoUW dataset category — determines pipeline routing and reward tier.
    pub dataset_category: Option<DatasetCategory>,
}

impl ScientificJob {
    pub fn new(
        source:           ExternalSource,
        external_ref:     Option<String>,
        dataset_root:     String,
        dataset_url:      Option<String>,
        algorithm:        Algorithm,
        task_description: String,
        reward_sompi:     u64,
        max_time_secs:    u64,
    ) -> Self {
        let now = now_secs();
        let job_id = format!(
            "{}-{}-{:x}",
            source,
            algorithm,
            now & 0xFFFF
        );
        Self {
            job_id,
            source,
            external_ref,
            dataset_root,
            dataset_url,
            algorithm,
            task_description,
            reward_sompi,
            max_time_secs,
            status:           JobStatus::Open,
            claimed_by:       None,
            created_at:       now,
            claimed_at:       None,
            completed_at:     None,
            pipeline:         None,
            pipeline_hash:    None,
            reference_genome: None,
            reference_hash:   None,
            container_hash:   None,
            config_hash:      None,
            deadline:         None,
            dataset_category: None,
        }
    }
}

// ── JobResult ─────────────────────────────────────────────────────────────────

/// A result submitted by a worker after completing a job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub result_id:      String,
    pub job_id:         String,
    /// Worker secp256k1 public key (hex compressed).
    pub worker_pubkey:  String,
    /// BLAKE3 Merkle root of all output files.
    pub result_root:    String,
    /// Algorithm-specific numeric score (e.g. alignment score, accuracy).
    pub score:          f64,
    /// BLAKE3 hash of the execution trace (stdout + stderr).
    pub trace_hash:     Option<String>,
    /// BLAKE3 hash of the Kaggle notebook or git repo used for training.
    pub notebook_or_repo_hash:  Option<String>,
    /// BLAKE3 hash of the container image used (Docker/Singularity digest).
    pub container_hash:         Option<String>,
    /// BLAKE3 hash of the trained model weights file.
    pub weights_hash:           Option<String>,
    /// BLAKE3 hash of the submission bundle (ZIP) sent to AIcrowd/DrivenData.
    pub submission_bundle_hash: Option<String>,
    /// Worker's secp256k1 signature over `result_root`.
    pub worker_sig:     String,
    /// Encrypted result payload (ChaCha20-Poly1305 encrypted with coordinator's public key).
    /// Contains: result_root + score + trace + all hashes.
    /// Only the coordinator (owner) can decrypt and validate.
    pub encrypted_payload: Option<String>,
    /// Ephemeral public key (hex) used for ECDH key exchange to derive encryption key.
    pub ephemeral_pubkey: Option<String>,
    /// Plain CSV of predictions (filename,confidence) — included in encrypted_payload, never stored plain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predictions_csv: Option<String>,
    pub submitted_at:   u64,
}

impl JobResult {
    /// Canonical sign data: `"{job_id}:{result_root}:{score:.6}:{trace_hash}"`
    /// Digest = BLAKE3(sign_data_bytes); signed with worker secp256k1 key.
    /// NOTE: result_root and score are cleared to "" / 0.0 after payload encryption;
    /// verify using decrypted EncryptedResultPayload values, not submitted fields.
    pub fn sign_bytes(&self) -> Vec<u8> {
        format!("{}:{}:{:.6}:{}",
            self.job_id,
            self.result_root,
            self.score,
            self.trace_hash.as_deref().unwrap_or("")
        ).into_bytes()
    }

    pub fn result_hash(&self) -> String {
        hex::encode(blake3::hash(&self.sign_bytes()).as_bytes())
    }
}

// ── ValidationReport ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidationVerdict {
    Valid,
    Invalid,
    Inconclusive,
}

/// Report produced by a validator after partial recomputation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub report_id:         String,
    pub job_id:            String,
    pub result_id:         String,
    /// Validator secp256k1 public key (hex compressed).
    pub validator_pubkey:  String,
    pub verdict:           ValidationVerdict,
    /// Recomputed score for comparison with submitted score.
    pub recomputed_score:  Option<f64>,
    /// Absolute difference between submitted and recomputed score.
    pub score_delta:       Option<f64>,
    pub notes:             Option<String>,
    /// Validator signature over `"{report_id}:{result_id}:{verdict_lowercase}"`.
    /// Digest = BLAKE3(sign_data_bytes); signed with validator secp256k1 key.
    pub validator_sig:     String,
    pub validated_at:      u64,
}

// ── SettlementPayload ─────────────────────────────────────────────────────────

/// On-chain anchor payload for a settled genetics-l2 job.
/// Stored in Xenom `tx.payload` as compact JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettlementPayload {
    pub app:            String,   // "genetics-l2"
    pub v:              u32,      // 1
    pub job_id:         String,
    pub source:         String,
    pub algorithm:      String,
    pub dataset_root:   String,
    pub results_root:   String,   // Merkle root over all validated result_roots
    pub best_score:     f64,
    pub winner_pubkey:  String,
    /// From the winning result — BLAKE3 of notebook or git repo.
    pub notebook_or_repo_hash:  Option<String>,
    /// From the winning result — Docker/Singularity image digest.
    pub container_hash:         Option<String>,
    /// From the winning result — BLAKE3 of trained model weights.
    pub weights_hash:           Option<String>,
    /// From the winning result — BLAKE3 of competition submission bundle.
    pub submission_bundle_hash: Option<String>,
    pub settled_at:     u64,
}

impl SettlementPayload {
    pub const APP_ID: &'static str = "genetics-l2";

    pub fn to_payload_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn from_payload_bytes(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok().filter(|p: &Self| p.app == Self::APP_ID)
    }
}

// ── Payout ────────────────────────────────────────────────────────────────────

/// Payout record created after settlement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Payout {
    pub payout_id:    String,
    pub job_id:       String,
    pub worker_pubkey: String,
    pub amount_sompi: u64,
    pub txid:         Option<String>,
    pub paid_at:      Option<u64>,
}

// ── Helpers ───────────────────────────────────────────────────────────────────

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn merkle_root_hex(leaves: &[String]) -> String {
    if leaves.is_empty() {
        return hex::encode([0u8; 32]);
    }
    let hashes: Vec<[u8; 32]> = leaves
        .iter()
        .map(|l| {
            hex::decode(l)
                .ok()
                .and_then(|v| v.try_into().ok())
                .unwrap_or([0u8; 32])
        })
        .collect();
    hex::encode(bioproof_core::merkle_root(&hashes))
}

// ── Result Encryption (ECIES: secp256k1 ECDH + ChaCha20-Poly1305) ────────────

/// Encrypted result payload structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedResultPayload {
    pub result_root: String,
    pub score: f64,
    pub trace_hash: Option<String>,
    pub notebook_or_repo_hash: Option<String>,
    pub container_hash: Option<String>,
    pub weights_hash: Option<String>,
    pub submission_bundle_hash: Option<String>,
    /// Plain-text CSV of predictions (filename,confidence) — only readable by coordinator.
    pub predictions_csv: Option<String>,
}

impl JobResult {
    /// Encrypt sensitive result data with coordinator's public key.
    /// Uses ECIES: ephemeral keypair + ECDH + ChaCha20-Poly1305.
    /// Returns (encrypted_payload_hex, ephemeral_pubkey_hex).
    pub fn encrypt_payload(
        &self,
        coordinator_pubkey_hex: &str,
    ) -> Result<(String, String), String> {
        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };
        use secp256k1::{ecdh::SharedSecret, PublicKey, Secp256k1, SecretKey};

        let secp = Secp256k1::new();

        // Parse coordinator's public key
        let coordinator_pubkey = PublicKey::from_slice(
            &hex::decode(coordinator_pubkey_hex).map_err(|e| format!("Invalid coordinator pubkey: {e}"))?,
        )
        .map_err(|e| format!("Invalid coordinator pubkey: {e}"))?;

        // Generate ephemeral keypair
        let ephemeral_secret = SecretKey::new(&mut secp256k1::rand::thread_rng());
        let ephemeral_pubkey = PublicKey::from_secret_key(&secp, &ephemeral_secret);

        // ECDH: shared = k_eph * P_coord  — only derivable with k_eph (worker) or k_coord (coordinator)
        let shared = SharedSecret::new(&coordinator_pubkey, &ephemeral_secret);
        let cipher = ChaCha20Poly1305::new_from_slice(&shared.secret_bytes())
            .map_err(|e| format!("Cipher init failed: {e}"))?;

        // Prepare payload
        let payload = EncryptedResultPayload {
            result_root: self.result_root.clone(),
            score: self.score,
            trace_hash: self.trace_hash.clone(),
            notebook_or_repo_hash: self.notebook_or_repo_hash.clone(),
            container_hash: self.container_hash.clone(),
            weights_hash: self.weights_hash.clone(),
            submission_bundle_hash: self.submission_bundle_hash.clone(),
            predictions_csv: self.predictions_csv.clone(),
        };
        let plaintext = serde_json::to_vec(&payload)
            .map_err(|e| format!("Serialize failed: {e}"))?;

        // Encrypt with random nonce
        let nonce_bytes: [u8; 12] = rand::random();
        let nonce = Nonce::from_slice(&nonce_bytes);
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| format!("Encryption failed: {e}"))?;

        // Combine nonce + ciphertext
        let mut encrypted = nonce_bytes.to_vec();
        encrypted.extend_from_slice(&ciphertext);

        Ok((
            hex::encode(encrypted),
            hex::encode(ephemeral_pubkey.serialize()),
        ))
    }

    /// Decrypt result payload using coordinator's private key.
    pub fn decrypt_payload(
        encrypted_hex: &str,
        ephemeral_pubkey_hex: &str,
        coordinator_privkey_hex: &str,
    ) -> Result<EncryptedResultPayload, String> {
        use chacha20poly1305::{
            aead::{Aead, KeyInit},
            ChaCha20Poly1305, Nonce,
        };
        use secp256k1::{ecdh::SharedSecret, PublicKey, SecretKey};

        // Parse keys
        let coordinator_secret = SecretKey::from_slice(
            &hex::decode(coordinator_privkey_hex).map_err(|e| format!("Invalid coordinator privkey: {e}"))?,
        )
        .map_err(|e| format!("Invalid coordinator privkey: {e}"))?;

        let ephemeral_pubkey = PublicKey::from_slice(
            &hex::decode(ephemeral_pubkey_hex).map_err(|e| format!("Invalid ephemeral pubkey: {e}"))?,
        )
        .map_err(|e| format!("Invalid ephemeral pubkey: {e}"))?;

        // ECDH: shared = k_coord * P_eph  — equals k_eph * P_coord; only coordinator holds k_coord
        let shared = SharedSecret::new(&ephemeral_pubkey, &coordinator_secret);
        let cipher = ChaCha20Poly1305::new_from_slice(&shared.secret_bytes())
            .map_err(|e| format!("Cipher init failed: {e}"))?;

        // Parse encrypted data
        let encrypted = hex::decode(encrypted_hex)
            .map_err(|e| format!("Invalid encrypted hex: {e}"))?;
        if encrypted.len() < 12 {
            return Err("Encrypted data too short".to_string());
        }

        let nonce = Nonce::from_slice(&encrypted[..12]);
        let ciphertext = &encrypted[12..];

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| format!("Decryption failed: {e}"))?;

        // Deserialize
        serde_json::from_slice(&plaintext)
            .map_err(|e| format!("Deserialize failed: {e}"))
    }
}
