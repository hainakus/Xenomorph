use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{sleep, Duration};

// ── L2 job types ──────────────────────────────────────────────────────────────

/// A scientific compute task piggybacked on a PoW notification.
/// Miners receive this alongside the PoW job and may optionally execute it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct L2Job {
    /// Pool theme: "genetics" | "climate" | "ai" | "materials"
    pub theme:       String,
    /// Coordinator job ID (from genetics-l2 / climate-l2 coordinator)
    pub job_id:      String,
    /// Algorithm or task type (e.g. "sequence_alignment", "variant_calling")
    pub task:        String,
    /// Coordinator files API URL: `{coordinator}/datasets/{job_id}/files`
    /// Miners fetch dataset files directly from this endpoint.
    pub dataset:     String,
    /// Fragment index for partial tasks (shard for large datasets)
    pub fragment:    u64,
    /// Reward in sompi offered for successful completion
    pub reward_sompi: u64,
    /// ISO-8601 timestamp when the job was posted
    pub posted_at:   u64,
}

impl L2Job {
    /// Serialise to a `serde_json::Value` for embedding in `mining.notify` param[6].
    pub fn to_value(&self) -> Value {
        serde_json::json!({
            "theme":       self.theme,
            "job_id":      self.job_id,
            "task":        self.task,
            "dataset":     self.dataset,
            "fragment":    self.fragment,
            "reward_sompi":self.reward_sompi,
        })
    }
}

// ── Shared state ──────────────────────────────────────────────────────────────

pub type L2JobSlot = Arc<RwLock<Option<Arc<L2Job>>>>;

pub fn new_slot() -> L2JobSlot {
    Arc::new(RwLock::new(None))
}

// ── Coordinator poller ────────────────────────────────────────────────────────

/// Continuously polls the L2 coordinator for the next open job and
/// updates the shared `L2JobSlot`.  Runs as a background Tokio task.
pub async fn run_poller(
    theme:           String,
    coordinator_url: String,
    poll_secs:       u64,
    slot:            L2JobSlot,
) {
    let http = reqwest::Client::new();
    let interval = Duration::from_secs(poll_secs.max(1));

    log::info!("L2 job poller started — theme={theme} coordinator={coordinator_url}");

    loop {
        match fetch_next_job(&http, &theme, &coordinator_url).await {
            Ok(Some(job)) => {
                let job_id = job.job_id.clone();
                *slot.write().await = Some(Arc::new(job));
                log::debug!("L2 job updated: {job_id}");
            }
            Ok(None) => {
                log::debug!("L2 coordinator: no open jobs");
            }
            Err(e) => {
                log::warn!("L2 coordinator fetch error: {e:#}");
            }
        }
        sleep(interval).await;
    }
}

async fn fetch_next_job(
    http:            &reqwest::Client,
    theme:           &str,
    coordinator_url: &str,
) -> anyhow::Result<Option<L2Job>> {
    let url = format!("{coordinator_url}/jobs?status=open&limit=1");
    let resp = http
        .get(&url)
        .send()
        .await
        .map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let body: Value = resp.json().await?;
    let jobs = body["jobs"].as_array().cloned().unwrap_or_default();

    let Some(raw) = jobs.into_iter().next() else {
        return Ok(None);
    };

    // Map coordinator ScientificJob fields to L2Job
    let job_id       = raw["job_id"].as_str().unwrap_or("").to_owned();
    let task         = raw["algorithm"].as_str().unwrap_or("").to_owned();
    let reward_sompi = raw["reward_sompi"].as_u64().unwrap_or(0);
    let posted_at    = raw["created_at"].as_u64().unwrap_or(0);
    // Dataset = coordinator files endpoint — miners fetch cached files from here
    let dataset_api  = format!("{coordinator_url}/datasets/{job_id}/files");

    // Use a deterministic fragment index based on the job_id so all miners
    // receive the same fragment number for the same job.
    let fragment = u64::from_be_bytes(
        hex::decode(&format!("{:016x}", job_id.len()))
            .unwrap_or_default()
            .try_into()
            .unwrap_or([0u8; 8])
    );

    if job_id.is_empty() || task.is_empty() {
        return Ok(None);
    }

    Ok(Some(L2Job {
        theme:        theme.to_owned(),
        job_id,
        task,
        dataset:      dataset_api,
        fragment,
        reward_sompi,
        posted_at,
    }))
}
