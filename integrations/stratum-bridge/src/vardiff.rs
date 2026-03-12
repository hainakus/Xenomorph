use std::{
    collections::VecDeque,
    time::{Duration, Instant},
};

/// Configuration shared across all miner connections.
#[derive(Clone)]
pub struct VarDiffConfig {
    /// Starting share difficulty sent on first authorize.
    pub initial_diff: f64,
    /// Never go below this difficulty.
    pub min_diff: f64,
    /// Never go above this difficulty.
    pub max_diff: f64,
    /// Target number of shares per minute (e.g. 20 = one share every 3 s).
    pub target_shares_per_min: f64,
    /// How often to retarget (seconds).
    pub retarget_secs: u64,
    /// Hysteresis: ignore changes smaller than this fraction of current diff.
    pub hysteresis: f64,
}

impl Default for VarDiffConfig {
    fn default() -> Self {
        Self {
            initial_diff:          1.0,
            min_diff:              0.1,
            max_diff:              1_000_000.0,
            target_shares_per_min: 20.0,
            retarget_secs:         60,
            hysteresis:            0.10,
        }
    }
}

/// Per-miner variable-difficulty state machine.
pub struct VarDiff {
    pub current_diff: f64,
    cfg:              VarDiffConfig,
    last_retarget:    Instant,
    share_times:      VecDeque<Instant>,
    window:           Duration,
}

impl VarDiff {
    pub fn new(cfg: VarDiffConfig) -> Self {
        let window = Duration::from_secs(cfg.retarget_secs * 2);
        let current_diff = cfg.initial_diff;
        Self {
            current_diff,
            cfg,
            last_retarget: Instant::now(),
            share_times:   VecDeque::new(),
            window,
        }
    }

    /// Record a new accepted share.
    ///
    /// Returns `Some(new_diff)` when the miner should be notified of a
    /// difficulty change via `mining.set_difficulty`, `None` otherwise.
    pub fn on_share(&mut self) -> Option<f64> {
        let now = Instant::now();
        self.share_times.push_back(now);

        // Prune entries older than rolling window
        let cutoff = now.checked_sub(self.window).unwrap_or(Instant::now());
        while self.share_times.front().is_some_and(|&t| t < cutoff) {
            self.share_times.pop_front();
        }

        // Only retarget at the configured interval
        if now.duration_since(self.last_retarget) < Duration::from_secs(self.cfg.retarget_secs) {
            return None;
        }
        self.last_retarget = now;

        // Actual shares per minute over the rolling window
        let actual_spm = self.share_times.len() as f64
            / self.window.as_secs_f64()
            * 60.0;

        let ratio = if actual_spm > 0.0 {
            actual_spm / self.cfg.target_shares_per_min
        } else {
            0.5 // no shares in window → halve difficulty
        };

        let new_diff = (self.current_diff * ratio)
            .clamp(self.cfg.min_diff, self.cfg.max_diff);

        // Hysteresis: skip tiny adjustments
        let change = (new_diff - self.current_diff).abs()
            / self.current_diff.max(f64::EPSILON);
        if change < self.cfg.hysteresis {
            return None;
        }

        self.current_diff = new_diff;
        Some(new_diff)
    }
}
