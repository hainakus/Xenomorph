use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

// ── External source ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExternalSource {
    Kaggle,
    Nih,
    Boinc,
    BioContest,
    Custom(String),
}

impl std::fmt::Display for ExternalSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kaggle     => write!(f, "kaggle"),
            Self::Nih        => write!(f, "nih"),
            Self::Boinc      => write!(f, "boinc"),
            Self::BioContest => write!(f, "bio_contest"),
            Self::Custom(s)  => write!(f, "{s}"),
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
    Custom(String),
}

impl std::fmt::Display for Algorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SequenceAlignment  => write!(f, "sequence_alignment"),
            Self::SmithWaterman      => write!(f, "smith_waterman"),
            Self::NeedlemanWunsch    => write!(f, "needleman_wunsch"),
            Self::VariantCalling     => write!(f, "variant_calling"),
            Self::GenomeAssembly     => write!(f, "genome_assembly"),
            Self::ProteinFolding     => write!(f, "protein_folding"),
            Self::RnaExpression      => write!(f, "rna_expression"),
            Self::Metagenomics       => write!(f, "metagenomics"),
            Self::MolecularDocking   => write!(f, "molecular_docking"),
            Self::Custom(s)          => write!(f, "{s}"),
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
            status:       JobStatus::Open,
            claimed_by:   None,
            created_at:   now,
            claimed_at:   None,
            completed_at: None,
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
    /// Worker's secp256k1 signature over `result_root`.
    pub worker_sig:     String,
    pub submitted_at:   u64,
}

impl JobResult {
    pub fn sign_bytes(&self) -> Vec<u8> {
        format!("{}:{}:{}", self.job_id, self.result_root, self.score)
            .into_bytes()
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
    /// Validator signature over `report_id:result_id:verdict`.
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
