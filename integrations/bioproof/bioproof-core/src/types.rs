use serde::{Deserialize, Serialize};

pub const APP_ID: &str = "bioproof";
pub const APP_VERSION: u32 = 1;

/// Default chunk size for file splitting (4 MiB).
pub const DEFAULT_CHUNK_SIZE: usize = 4 * 1024 * 1024;

// ── ArtifactType ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactType {
    Fastq,
    Bam,
    Cram,
    Vcf,
    Pipeline,
    #[serde(rename = "ai-model")]
    AiModel,
    #[serde(rename = "ai-output")]
    AiOutput,
    Report,
    Other(String),
}

impl std::fmt::Display for ArtifactType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fastq    => write!(f, "fastq"),
            Self::Bam      => write!(f, "bam"),
            Self::Cram     => write!(f, "cram"),
            Self::Vcf      => write!(f, "vcf"),
            Self::Pipeline => write!(f, "pipeline"),
            Self::AiModel  => write!(f, "ai-model"),
            Self::AiOutput => write!(f, "ai-output"),
            Self::Report   => write!(f, "report"),
            Self::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::str::FromStr for ArtifactType {
    type Err = std::convert::Infallible;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "fastq"     => Self::Fastq,
            "bam"       => Self::Bam,
            "cram"      => Self::Cram,
            "vcf"       => Self::Vcf,
            "pipeline"  => Self::Pipeline,
            "ai-model"  => Self::AiModel,
            "ai-output" => Self::AiOutput,
            "report"    => Self::Report,
            other       => Self::Other(other.to_owned()),
        })
    }
}

// ── Manifest ──────────────────────────────────────────────────────────────────

/// Off-chain manifest describing a genomics/AI artefact.
/// Stored alongside the data; the `manifest_hash` is anchored on-chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    /// Caller-supplied stable identifier (e.g. sample barcode).
    pub dataset_id:     String,
    pub artifact_type:  ArtifactType,
    /// Chunk size used when computing `proof_root` (bytes).
    pub chunk_size:     usize,
    /// BLAKE3 hash of the complete file (hex).
    pub file_hash:      String,
    /// BLAKE3 binary Merkle root over per-chunk hashes (hex).
    pub proof_root:     String,
    /// Optional BLAKE3 hash of the pipeline definition file (hex).
    pub pipeline_hash:  Option<String>,
    /// Optional BLAKE3 hash of the AI model weights (hex).
    pub model_hash:     Option<String>,
    /// `proof_root` of the parent artefact for lineage tracking.
    pub parent_root:    Option<String>,
    /// Issuer identifier (lab ID, public key fingerprint, DID, …).
    pub issuer:         String,
    /// Unix timestamp (seconds) when the manifest was created.
    pub created_at:     u64,
}

impl Manifest {
    /// Canonical JSON bytes used as the signing payload.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("manifest is always serialisable")
    }

    /// BLAKE3 hash of the canonical JSON, as raw bytes.
    pub fn hash_bytes(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    /// BLAKE3 hash of the canonical JSON, as lowercase hex.
    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash_bytes())
    }
}

// ── AnchorPayload ─────────────────────────────────────────────────────────────

/// Compact payload embedded in an on-chain OP_RETURN output.
/// Serialises to < 200 bytes of JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnchorPayload {
    pub app:           String,
    pub v:             u32,
    pub proof_root:    String,
    pub manifest_hash: String,
    pub kind:          String,
}

impl AnchorPayload {
    pub fn new(proof_root: &str, manifest_hash: &str, kind: &str) -> Self {
        Self {
            app:           APP_ID.to_owned(),
            v:             APP_VERSION,
            proof_root:    proof_root.to_owned(),
            manifest_hash: manifest_hash.to_owned(),
            kind:          kind.to_owned(),
        }
    }

    /// Serialise to compact JSON bytes (for OP_RETURN script data).
    pub fn to_op_return_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    /// Parse from raw OP_RETURN script data bytes.
    pub fn from_op_return_bytes(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok().filter(|p: &Self| p.app == APP_ID)
    }
}

// ── Certificate ───────────────────────────────────────────────────────────────

// ── Compute Job types ─────────────────────────────────────────────────────────

/// Job type posted to the Xenom compute job market.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobType {
    GenomicsPipeline,
    AiInference,
    AiTraining,
    ProteinFolding,
    MolecularDynamics,
    VariantCalling,
    GenomeAssembly,
    SingleCellRna,
    Metagenomics,
    Custom(String),
}

impl std::fmt::Display for JobType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GenomicsPipeline  => write!(f, "genomics_pipeline"),
            Self::AiInference       => write!(f, "ai_inference"),
            Self::AiTraining        => write!(f, "ai_training"),
            Self::ProteinFolding    => write!(f, "protein_folding"),
            Self::MolecularDynamics => write!(f, "molecular_dynamics"),
            Self::VariantCalling    => write!(f, "variant_calling"),
            Self::GenomeAssembly    => write!(f, "genome_assembly"),
            Self::SingleCellRna     => write!(f, "single_cell_rna"),
            Self::Metagenomics      => write!(f, "metagenomics"),
            Self::Custom(s)         => write!(f, "{s}"),
        }
    }
}

/// A compute job posted to the Xenom job market.
/// Anchored on-chain via `tx.payload` with `app = "bioproof-job"`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeJob {
    /// Unique stable job identifier.
    pub job_id:          String,
    pub job_type:        JobType,
    /// BLAKE3 proof_root of the input dataset.
    pub input_root:      String,
    /// BLAKE3 hash of the pipeline spec (Nextflow/WDL/Snakemake file).
    pub pipeline_hash:   String,
    /// BLAKE3 hash of the container image (for reproducibility).
    pub container_hash:  Option<String>,
    /// BLAKE3 hash of model weights (AI jobs).
    pub model_hash:      Option<String>,
    /// Reward in sompi (smallest XEN unit).
    pub reward_sompi:    u64,
    /// Max wall-clock execution time in seconds.
    pub max_time_secs:   u64,
    /// Issuer/requester public key (hex compressed secp256k1).
    pub requester_pubkey: String,
    /// Unix timestamp when the job was posted.
    pub posted_at:       u64,
}

impl ComputeJob {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("always serialisable")
    }

    pub fn hash_bytes(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash_bytes())
    }
}

/// Proof-of-execution manifest produced by a Scientific Worker.
/// Captures everything needed to verify the computation was done correctly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeJobManifest {
    /// References the original ComputeJob.
    pub job_id:               String,
    pub job_type:             JobType,
    /// BLAKE3 proof_root of input files (must match ComputeJob.input_root).
    pub input_root:           String,
    /// BLAKE3 hash of the pipeline actually executed.
    pub pipeline_hash:        String,
    /// BLAKE3 hash of the container image used.
    pub container_hash:       Option<String>,
    /// BLAKE3 hash of model weights used (AI jobs).
    pub model_hash:           Option<String>,
    /// BLAKE3 Merkle root over all output files.
    pub output_root:          String,
    /// Individual output files: (filename, proof_root).
    pub outputs:              Vec<OutputEntry>,
    /// BLAKE3 hash of the execution log/trace (stdout+stderr).
    pub execution_trace_hash: Option<String>,
    /// Worker's secp256k1 public key (hex compressed).
    pub worker_pubkey:        String,
    /// Unix timestamp when execution finished.
    pub completed_at:         u64,
}

/// One output artefact produced by the compute job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputEntry {
    pub name:       String,
    pub kind:       ArtifactType,
    pub proof_root: String,
    pub file_hash:  String,
    pub size_bytes: u64,
}

impl ComputeJobManifest {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("always serialisable")
    }

    pub fn hash_bytes(&self) -> [u8; 32] {
        *blake3::hash(&self.canonical_bytes()).as_bytes()
    }

    pub fn hash_hex(&self) -> String {
        hex::encode(self.hash_bytes())
    }
}

/// On-chain anchor payload for a completed compute job.
/// Stored in `tx.payload`, extends the base AnchorPayload concept.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobAnchorPayload {
    pub app:           String,         // "bioproof-job"
    pub v:             u32,            // 1
    pub job_id:        String,
    pub manifest_hash: String,         // blake3 of ComputeJobManifest
    pub output_root:   String,         // top-level output Merkle root
    pub worker_pubkey: String,
    pub worker_sig:    String,         // sig over manifest_hash
}

impl JobAnchorPayload {
    pub const APP_ID: &'static str = "bioproof-job";

    pub fn new(
        job_id:        &str,
        manifest_hash: &str,
        output_root:   &str,
        worker_pubkey: &str,
        worker_sig:    &str,
    ) -> Self {
        Self {
            app:           Self::APP_ID.to_owned(),
            v:             1,
            job_id:        job_id.to_owned(),
            manifest_hash: manifest_hash.to_owned(),
            output_root:   output_root.to_owned(),
            worker_pubkey: worker_pubkey.to_owned(),
            worker_sig:    worker_sig.to_owned(),
        }
    }

    pub fn to_payload_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).unwrap_or_default()
    }

    pub fn from_payload_bytes(data: &[u8]) -> Option<Self> {
        serde_json::from_slice(data).ok().filter(|p: &Self| p.app == Self::APP_ID)
    }
}

// ── Worker capability types ───────────────────────────────────────────────────

/// Pipeline execution backends a worker can support.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackend {
    /// Docker container runtime.
    Docker,
    /// Singularity/Apptainer (HPC-compatible rootless containers).
    Singularity,
    /// Nextflow workflow engine.
    Nextflow,
    /// Snakemake workflow engine.
    Snakemake,
    /// Cromwell / WDL workflow engine.
    Cromwell,
    /// Native bash + Python + conda execution (no containerisation).
    Native,
    /// CUDA GPU compute (PyTorch/TensorFlow).
    Cuda,
    /// ROCm GPU compute (AMD).
    Rocm,
}

impl std::fmt::Display for ExecutionBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Docker      => write!(f, "docker"),
            Self::Singularity => write!(f, "singularity"),
            Self::Nextflow    => write!(f, "nextflow"),
            Self::Snakemake   => write!(f, "snakemake"),
            Self::Cromwell    => write!(f, "cromwell"),
            Self::Native      => write!(f, "native"),
            Self::Cuda        => write!(f, "cuda"),
            Self::Rocm        => write!(f, "rocm"),
        }
    }
}

/// GPU device information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub index:       usize,
    pub model:       String,
    pub vram_mb:     u64,
    pub cuda:        bool,
    pub rocm:        bool,
}

/// Capabilities advertised by a Scientific Worker Node.
/// Published to the job market so requesters can match jobs to workers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerCapabilities {
    /// Worker's secp256k1 public key (hex compressed) — its network identity.
    pub worker_pubkey:   String,
    /// Human-readable hostname.
    pub hostname:        String,
    /// Logical CPU count.
    pub cpu_count:       usize,
    /// Total RAM in MiB.
    pub ram_mib:         u64,
    /// Available disk in MiB.
    pub disk_mib:        u64,
    /// GPU devices available.
    pub gpus:            Vec<GpuInfo>,
    /// Execution backends installed on this node.
    pub backends:        Vec<ExecutionBackend>,
    /// Job types this worker accepts.
    pub job_types:       Vec<JobType>,
    /// Maximum number of concurrent jobs.
    pub max_concurrency: usize,
    /// Unix timestamp when these capabilities were last measured.
    pub measured_at:     u64,
}

impl WorkerCapabilities {
    pub fn supports_backend(&self, b: &ExecutionBackend) -> bool {
        self.backends.contains(b)
    }

    pub fn supports_job_type(&self, jt: &JobType) -> bool {
        self.job_types.contains(jt)
    }

    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
}

/// Full verifiable certificate combining off-chain manifest with on-chain proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Certificate {
    pub manifest:      Manifest,
    pub manifest_hash: String,
    /// Hex-encoded DER secp256k1 ECDSA signature over `manifest_hash`.
    pub issuer_sig:    String,
    /// Hex-encoded compressed secp256k1 public key of the issuer.
    pub issuer_pubkey: String,
    /// Xenom transaction ID that carries the anchor (filled after submission).
    pub txid:          Option<String>,
    /// DAA score of the block containing the anchor transaction.
    pub daa_score:     Option<u64>,
    /// Unix timestamp of the anchoring block.
    pub anchored_at:   Option<u64>,
}
