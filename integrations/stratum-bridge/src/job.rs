use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use kaspa_consensus_core::{hashing::header::hash_override_nonce_time, header::Header};
use kaspa_core::warn;
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
    /// `true` once Genome PoW is active (epoch_seed != zero hash).
    pub genome_active: bool,

    pub created: Instant,
}

impl Job {
    pub fn new(counter: u64, template: RpcRawBlock) -> Self {
        let id = format!("{counter:016x}");

        let header: Header = (&template.header).into();
        let pre_pow_hash = hash_override_nonce_time(&header, 0, 0);

        let pre_pow_hash_hex = bytes_to_hex(&pre_pow_hash.as_bytes());
        let bits_hex = format!("{:08x}", template.header.bits);
        let epoch_seed_hex = bytes_to_hex(&template.header.epoch_seed.as_bytes());
        let timestamp_hex = format!("{:016x}", template.header.timestamp);

        let genome_active = template.header.epoch_seed != kaspa_hashes::Hash::default();

        Self {
            id,
            template: Arc::new(template),
            pre_pow_hash_hex,
            bits_hex,
            epoch_seed_hex,
            timestamp_hex,
            pre_pow_hash,
            genome_active,
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
}

impl JobManager {
    pub fn new() -> Self {
        Self { jobs: HashMap::new(), current: None, counter: 0, last_template_id: None }
    }

    /// Returns `true` when the coinbase payload is structurally valid for the
    /// template's PoW mode.
    ///
    /// Once Genome PoW activates the node serialises a V2 payload:
    ///   blue_score(8) + subsidy(8) + spk_ver(2) + spk_len(1) + spk(N) + fitness(4) + extra_data
    ///
    /// `modify_block_template` (triggered server-side when the template cache
    /// was warmed by a *different* miner address) used to call `modify_coinbase_payload`
    /// which dropped the 4 fitness bytes.  The resulting payload then had the first
    /// 4 bytes of the extra_data version string (e.g. "0.15") parsed as fitness ≈ 892M,
    /// far above the fitness_threshold (10_000), which clamps the multiplier to 2.0.
    /// The node then computes expected_subsidy = 2 × base while the payload carries the
    /// original subsidy (computed with the real fitness), producing a WrongSubsidy error.
    ///
    /// This guard is a belt-and-suspenders check: the upstream bug is fixed in
    /// `modify_coinbase_payload`, but we also reject any template where the fitness
    /// bytes look like corrupted data (valid genome fitness is at most 3000).
    fn coinbase_payload_valid(template: &RpcRawBlock) -> bool {
        let Some(coinbase) = template.transactions.first() else { return false; };
        let p = &coinbase.payload;
        // V1 minimum: blue_score(8) + subsidy(8) + spk_ver(2) + spk_len(1) = 19 bytes
        if p.len() < 19 {
            return false;
        }
        let spk_len = p[18] as usize;
        if p.len() < 19 + spk_len {
            return false;
        }
        // For Genome-PoW (V2) blocks the fitness field must immediately follow the SPK.
        // genome_active here uses the same epoch_seed heuristic as the rest of the bridge.
        let genome_active = template.header.epoch_seed != kaspa_hashes::Hash::default();
        if genome_active {
            if p.len() < 19 + spk_len + 4 {
                return false;
            }
            // genome_pow::compute_fitness returns values in [0, 3000].
            // Corrupted payloads (stripped fitness, version-string bytes in its place)
            // produce values like 892_415_536.  Reject anything implausibly large.
            let fitness = u32::from_le_bytes(
                p[19 + spk_len..19 + spk_len + 4].try_into().unwrap(),
            );
            if fitness > 100_000 {
                return false;
            }
            // V2 subsidy depends on the miner's SPK via calc_expected_fitness.
            // modify_block_template (called when the node's template cache was warmed
            // by a *different* miner) correctly updates the payload SPK but leaves
            // outputs 0/1 with the original miner's SPK and fitness-based amounts.
            // verify_coinbase_transaction then recomputes expected outputs for the new
            // SPK → different amounts → BadCoinbaseTransaction.
            // Detect the mismatch by comparing the payload SPK against output 0.
            let payload_spk_ver = u16::from_le_bytes(p[16..18].try_into().unwrap());
            let payload_spk_script = &p[19..19 + spk_len];
            if let Some(out0) = coinbase.outputs.first() {
                let out_spk = &out0.script_public_key;
                if out_spk.version() != payload_spk_ver || out_spk.script() != payload_spk_script {
                    return false;
                }
            }
        }
        true
    }

    /// Register a new template if it differs from the last one.
    ///
    /// Returns `Some(job)` when the template changed, `None` when unchanged.
    pub fn update(&mut self, template: RpcRawBlock) -> Option<Arc<Job>> {
        let template_id = template.header.accepted_id_merkle_root;
        if self.last_template_id == Some(template_id) {
            return None;
        }
        // Reject templates with a corrupted V2 coinbase payload (fitness bytes
        // stripped by a server-side modify_coinbase_payload call).  Mark the
        // template as seen so we don't spam this warning on every poll cycle;
        // the bridge will simply hold the previous job until the virtual state
        // advances and the node builds a fresh, correct template.
        if !Self::coinbase_payload_valid(&template) {
            warn!(
                "Dropping template with invalid V2 coinbase (bad fitness or SPK/output-0 mismatch \
                 from modify_block_template); waiting for fresh template"
            );
            self.last_template_id = Some(template_id);
            return None;
        }
        self.last_template_id = Some(template_id);
        self.counter += 1;

        let job = Arc::new(Job::new(self.counter, template));
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
