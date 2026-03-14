use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use kaspa_consensus_core::{hashing::header::hash_override_nonce_time, header::Header};
use kaspa_rpc_core::{RpcRawBlock, RpcRawHeader};

const MAX_JOBS: usize = 64;
const JOB_STALE_SECS: u64 = 60;

// ── Job ───────────────────────────────────────────────────────────────────────

/// A single unit of work sent to miners via `mining.notify`.
pub struct Job {
    /// Hex job identifier (16 chars = 8-byte counter).
    pub id: String,
    /// The full block template, kept for block reconstruction on submit.
    pub template: Arc<RpcRawBlock>,

    // Pre-computed stratum notify fields (all lowercase hex strings)

    /// `hash_override_nonce_time(header, 0, 0)` — the commitment the miner works against.
    pub pre_pow_hash_hex: String,
    /// Compact difficulty target from `header.bits`.
    pub bits_hex: String,
    /// Genome epoch seed — selects which genome fragment is hashed.
    pub epoch_seed_hex: String,
    /// Template timestamp in milliseconds (16 hex chars).
    pub timestamp_hex: String,

    /// Decoded `pre_pow_hash` used for KHeavyHash share validation.
    #[allow(dead_code)]
    pub pre_pow_hash: kaspa_hashes::Hash,
    /// `true` once Genome PoW is active (daa_score >= genome_pow_activation_daa_score).
    pub genome_active: bool,
    /// Template DAA score as 16-char hex, sent in `mining.notify` so miners can determine PoW mode.
    pub daa_score_hex: String,

    pub created: Instant,
}

impl Job {
    pub fn new(counter: u64, template: RpcRawBlock, genome_pow_activation_daa_score: u64) -> Self {
        let id = format!("{counter:016x}");

        let header: Header = (&template.header).into();
        let pre_pow_hash = hash_override_nonce_time(&header, 0, 0);

        let pre_pow_hash_hex = bytes_to_hex(&pre_pow_hash.as_bytes());
        let bits_hex = format!("{:08x}", template.header.bits);
        let epoch_seed_hex = bytes_to_hex(&template.header.epoch_seed.as_bytes());
        let timestamp_hex = format!("{:016x}", template.header.timestamp);

        let genome_active = template.header.daa_score >= genome_pow_activation_daa_score;
        let daa_score_hex = format!("{:016x}", template.header.daa_score);

        Self {
            id,
            template: Arc::new(template),
            pre_pow_hash_hex,
            bits_hex,
            epoch_seed_hex,
            timestamp_hex,
            pre_pow_hash,
            genome_active,
            daa_score_hex,
            created: Instant::now(),
        }
    }

    /// Reconstruct the `RpcRawBlock` with the miner-supplied `nonce` ready for submission.
    pub fn build_block(&self, nonce: u64) -> RpcRawBlock {
        let h = &self.template.header;
        RpcRawBlock {
            header: RpcRawHeader {
                version: h.version,
                parents_by_level: h.parents_by_level.clone(),
                hash_merkle_root: h.hash_merkle_root,
                accepted_id_merkle_root: h.accepted_id_merkle_root,
                utxo_commitment: h.utxo_commitment,
                timestamp: h.timestamp,
                bits: h.bits,
                nonce,
                daa_score: h.daa_score,
                blue_work: h.blue_work,
                blue_score: h.blue_score,
                epoch_seed: h.epoch_seed,
                pruning_point: h.pruning_point,
            },
            transactions: self.template.transactions.clone(),
        }
    }

    pub fn is_stale(&self) -> bool {
        self.created.elapsed() > Duration::from_secs(JOB_STALE_SECS)
    }
}

// ── JobManager ────────────────────────────────────────────────────────────────

/// Maintains a window of recent jobs and detects new block templates.
pub struct JobManager {
    jobs: HashMap<String, Arc<Job>>,
    pub current: Option<Arc<Job>>,
    counter: u64,
    last_template_id: Option<kaspa_hashes::Hash>,
    genome_pow_activation_daa_score: u64,
}

impl JobManager {
    pub fn new(genome_pow_activation_daa_score: u64) -> Self {
        Self { jobs: HashMap::new(), current: None, counter: 0, last_template_id: None, genome_pow_activation_daa_score }
    }

    /// Register a new template if it differs from the last one.
    ///
    /// Returns `Some(job)` when the template changed, `None` when unchanged.
    pub fn update(&mut self, template: RpcRawBlock) -> Option<Arc<Job>> {
        let template_id = template.header.accepted_id_merkle_root;
        if self.last_template_id == Some(template_id) {
            return None;
        }
        self.last_template_id = Some(template_id);
        self.counter += 1;

        let job = Arc::new(Job::new(self.counter, template, self.genome_pow_activation_daa_score));
        self.jobs.insert(job.id.clone(), job.clone());
        self.current = Some(job.clone());

        // Prune stale jobs
        if self.jobs.len() > MAX_JOBS {
            self.jobs.retain(|_, j| !j.is_stale());
        }

        Some(job)
    }

    /// Look up a job by its ID.
    pub fn get(&self, job_id: &str) -> Option<Arc<Job>> {
        self.jobs.get(job_id).cloned()
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn bytes_to_hex(bytes: &[u8]) -> String {
    let mut out = vec![0u8; bytes.len() * 2];
    faster_hex::hex_encode(bytes, &mut out).expect("hex encode");
    String::from_utf8(out).expect("valid utf8")
}
