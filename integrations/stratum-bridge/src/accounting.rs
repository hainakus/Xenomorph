use std::{
    collections::{HashMap, VecDeque},
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use kaspa_core::{info, warn};
use serde::{Deserialize, Serialize};

// ── Internal share record ─────────────────────────────────────────────────────

struct Share {
    worker: String,
    diff:   f64,
    #[allow(dead_code)]
    at:     Instant,
}

// ── Public types ──────────────────────────────────────────────────────────────

/// Accumulated statistics for a single worker name.
#[derive(Clone, Debug, Default)]
pub struct WorkerStats {
    pub shares_submitted: u64,
    /// Sum of all share difficulties submitted.
    pub total_diff:       f64,
    pub blocks_found:     u64,
    pub last_seen:        Option<Instant>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PayoutStatus {
    /// Waiting for `confirm_depth` DAA-score steps.
    Pending,
    /// Payout transaction submitted successfully.
    Paid { tx_id: String },
    /// Submission failed.
    Failed { reason: String },
}

/// A single payout entry that is persisted to `--payout-file`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PendingPayout {
    pub job_id:          String,
    pub unix_secs:       u64,
    /// DAA score of the mined block (used to determine confirmation depth).
    pub block_daa_score: u64,
    /// (worker_name, proportion 0..1).  Proportions sum to ~1.0.
    pub proportions:     Vec<(String, f64)>,
    pub status:          PayoutStatus,
}

// ── Accounting ────────────────────────────────────────────────────────────────

/// PPLNS share window + payout tracking.
///
/// Thread-safety: wrap in `tokio::sync::Mutex` before sharing.
pub struct Accounting {
    window_size: usize,
    shares:      VecDeque<Share>,
    workers:     HashMap<String, WorkerStats>,
    pending:     Vec<PendingPayout>,
    payout_file: Option<PathBuf>,
}

impl Accounting {
    pub fn new(window_size: usize, payout_file: Option<PathBuf>) -> Self {
        Self {
            window_size,
            shares:      VecDeque::with_capacity(window_size.min(1024)),
            workers:     HashMap::new(),
            pending:     Vec::new(),
            payout_file,
        }
    }

    /// Record a submitted share (called for every valid submission regardless
    /// of whether the node accepted it as a block).
    pub fn record_share(&mut self, worker: &str, diff: f64) {
        self.shares.push_back(Share {
            worker: worker.to_owned(),
            diff,
            at:     Instant::now(),
        });
        while self.shares.len() > self.window_size {
            self.shares.pop_front();
        }

        let e = self.workers.entry(worker.to_owned()).or_default();
        e.shares_submitted += 1;
        e.total_diff       += diff;
        e.last_seen        = Some(Instant::now());
    }

    /// Called when a block is accepted by the node.
    /// `daa_score` is the approximate DAA score of the mined block.
    /// Returns the PPLNS payout distribution.
    pub fn record_block(&mut self, job_id: &str, daa_score: u64) -> PendingPayout {
        let total_diff: f64 = self.shares.iter().map(|s| s.diff).sum();

        let mut prop_map: HashMap<String, f64> = HashMap::new();
        if total_diff > 0.0 {
            for s in &self.shares {
                *prop_map.entry(s.worker.clone()).or_default() += s.diff / total_diff;
            }
        }

        let mut proportions: Vec<(String, f64)> = prop_map.into_iter().collect();
        proportions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        info!(
            "Block found! PPLNS window: {} shares  total_diff={:.2}",
            self.shares.len(),
            total_diff
        );
        for (worker, proportion) in &proportions {
            info!("  payout: {worker}  {:.4}%", proportion * 100.0);
            if let Some(e) = self.workers.get_mut(worker) {
                e.blocks_found += 1;
            }
        }

        let payout = PendingPayout {
            job_id:          job_id.to_owned(),
            unix_secs,
            block_daa_score: daa_score,
            proportions,
            status:          PayoutStatus::Pending,
        };

        self.pending.push(payout.clone());
        self.flush_file();
        payout
    }

    pub fn worker_stats(&self) -> &HashMap<String, WorkerStats> {
        &self.workers
    }

    pub fn pending_payouts(&self) -> &[PendingPayout] {
        &self.pending
    }

    /// Returns payouts whose block has reached `confirm_depth` DAA-score steps
    /// behind the current virtual DAA score, and marks them as ready to pay.
    /// Caller should execute the payout and then call `mark_paid` / `mark_failed`.
    pub fn take_confirmed_payouts(
        &self,
        current_daa_score: u64,
        confirm_depth:     u64,
    ) -> Vec<PendingPayout> {
        self.pending
            .iter()
            .filter(|p| {
                p.status == PayoutStatus::Pending
                    && current_daa_score.saturating_sub(p.block_daa_score) >= confirm_depth
            })
            .cloned()
            .collect()
    }

    pub fn mark_paid(&mut self, job_id: &str, tx_id: String) {
        if let Some(p) = self.pending.iter_mut().find(|p| p.job_id == job_id) {
            p.status = PayoutStatus::Paid { tx_id };
            self.flush_file();
        }
        self.trim_old_payouts(500);
    }

    pub fn mark_failed(&mut self, job_id: &str, reason: String) {
        if let Some(p) = self.pending.iter_mut().find(|p| p.job_id == job_id) {
            p.status = PayoutStatus::Failed { reason };
            self.flush_file();
        }
        self.trim_old_payouts(500);
    }

    /// Remove resolved (Paid/Failed) entries older than the most recent `keep`
    /// entries, and cap total in-memory size. Keeps all `Pending` entries.
    pub fn trim_old_payouts(&mut self, keep: usize) {
        // Partition into pending and resolved
        let (mut pending_entries, mut resolved): (Vec<_>, Vec<_>) = self
            .pending
            .drain(..)
            .partition(|p| p.status == PayoutStatus::Pending);

        // Keep only the most-recent `keep` resolved entries (they're in insertion order)
        if resolved.len() > keep {
            let drop_n = resolved.len() - keep;
            resolved.drain(..drop_n);
        }

        // Rebuild: resolved first (oldest), pending last (newest) — preserves insert order
        self.pending = resolved;
        self.pending.append(&mut pending_entries);
    }

    /// Log a summary of current worker stats.
    pub fn log_stats(&self) {
        if self.workers.is_empty() {
            info!("Pool stats: no miners connected");
            return;
        }
        info!("Pool stats (PPLNS window={} shares):", self.shares.len());
        let mut rows: Vec<(&String, &WorkerStats)> = self.workers.iter().collect();
        rows.sort_by_key(|(w, _)| w.as_str());
        for (worker, s) in rows {
            info!(
                "  {worker}  shares={} total_diff={:.2} blocks={}",
                s.shares_submitted, s.total_diff, s.blocks_found
            );
        }
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn flush_file(&self) {
        let Some(path) = &self.payout_file else { return };
        match serde_json::to_string_pretty(&self.pending) {
            Ok(json) => {
                if let Err(e) = std::fs::write(path, &json) {
                    warn!("Failed to write payout file {}: {e}", path.display());
                }
            }
            Err(e) => warn!("Payout JSON serialize error: {e}"),
        }
    }
}
