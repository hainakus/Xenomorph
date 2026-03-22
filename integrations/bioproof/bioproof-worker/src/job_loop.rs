use anyhow::{bail, Context, Result};
use bioproof_core::{blake3_hex, sign_manifest, ComputeJob, ComputeJobManifest, JobAnchorPayload, WorkerCapabilities};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::time::{sleep, Duration};

use crate::executor::{pick_executor, ExecResult};
use crate::proof::{combined_output_root, hash_directory, hash_output_files};

// ── Configuration ─────────────────────────────────────────────────────────────

pub struct WorkerConfig {
    /// Directory watched for incoming job JSON files.
    pub job_inbox: PathBuf,
    /// Root working directory; each job gets its own sub-directory.
    pub work_root: PathBuf,
    /// Worker private key (hex) for signing manifests.
    pub privkey_hex: String,
    /// bioproof-api base URL for result submission (optional).
    #[allow(dead_code)]
    pub api_url: Option<String>,
    /// Xenom node gRPC address for on-chain anchoring.
    pub node_addr: String,
    /// Milliseconds between inbox scans.
    pub poll_ms: u64,
    /// Whether to submit to the chain (false = dry-run).
    pub submit: bool,
    /// Use devnet address prefix + lower coinbase maturity.
    pub devnet: bool,
}

// ── Daemon loop ───────────────────────────────────────────────────────────────

/// Run the worker daemon.  Scans `config.job_inbox` for `*.job.json` files,
/// processes each one, moves it to `<inbox>/done/` or `<inbox>/failed/` and
/// loops.
pub async fn run(caps: Arc<WorkerCapabilities>, cfg: Arc<WorkerConfig>) -> Result<()> {
    log::info!("Worker daemon started");
    log::info!("  inbox:     {}", cfg.job_inbox.display());
    log::info!("  work_root: {}", cfg.work_root.display());
    log::info!("  backends:  {}", caps.backends.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", "));
    log::info!("  job_types: {}", caps.job_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));

    tokio::fs::create_dir_all(&cfg.job_inbox).await?;
    tokio::fs::create_dir_all(cfg.job_inbox.join("done")).await?;
    tokio::fs::create_dir_all(cfg.job_inbox.join("failed")).await?;
    tokio::fs::create_dir_all(&cfg.work_root).await?;

    loop {
        match scan_inbox(&caps, &cfg).await {
            Ok(n) if n > 0 => log::info!("Processed {n} job(s)"),
            Ok(_) => {}
            Err(e) => log::warn!("Inbox scan error: {e:#}"),
        }
        sleep(Duration::from_millis(cfg.poll_ms)).await;
    }
}

async fn scan_inbox(caps: &WorkerCapabilities, cfg: &WorkerConfig) -> Result<usize> {
    let mut rd = tokio::fs::read_dir(&cfg.job_inbox).await?;
    let mut count = 0;

    while let Some(entry) = rd.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if path.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with('.')).unwrap_or(false) {
            continue;
        }

        let bytes = tokio::fs::read(&path).await?;
        let job: ComputeJob = match serde_json::from_slice(&bytes) {
            Ok(j) => j,
            Err(e) => {
                log::warn!("Skipping {}: invalid JSON: {e}", path.display());
                continue;
            }
        };

        if !caps.supports_job_type(&job.job_type) {
            log::info!("Skipping job {} (unsupported type: {})", job.job_id, job.job_type);
            continue;
        }

        log::info!("Claiming job {} [{}]", job.job_id, job.job_type);
        let dest = match process_job(&job, caps, cfg).await {
            Ok(_) => {
                count += 1;
                cfg.job_inbox.join("done").join(path.file_name().unwrap())
            }
            Err(e) => {
                log::error!("Job {} failed: {e:#}", job.job_id);
                cfg.job_inbox.join("failed").join(path.file_name().unwrap())
            }
        };

        let _ = tokio::fs::rename(&path, &dest).await;
    }

    Ok(count)
}

// ── Job processing ────────────────────────────────────────────────────────────

async fn process_job(job: &ComputeJob, caps: &WorkerCapabilities, cfg: &WorkerConfig) -> Result<()> {
    let work_dir = cfg.work_root.join(&job.job_id);
    let input_dir = work_dir.join("input");
    let output_dir = work_dir.join("output");
    let pipeline = work_dir.join("pipeline");

    tokio::fs::create_dir_all(&output_dir).await?;

    // ── 1. Verify input root ──────────────────────────────────────────────────
    log::info!("[{}] Verifying input root…", job.job_id);
    let actual_input_root = hash_directory(&input_dir).await.context("input hashing failed")?;

    if actual_input_root != job.input_root {
        bail!("Input root mismatch: expected={} got={}", job.input_root, actual_input_root);
    }
    log::info!("[{}] Input root OK", job.job_id);

    // ── 2. Pick executor + run pipeline ──────────────────────────────────────
    let preferred_backend = caps.backends.first();
    let executor = pick_executor(preferred_backend);
    log::info!("[{}] Executor: {}", job.job_id, executor.name());

    let ExecResult { trace, success } = executor.execute(job, &pipeline, &input_dir, &output_dir).await.context("executor failed")?;

    let trace_hash = blake3_hex(trace.as_bytes());
    log::info!("[{}] Pipeline exit: ok={success}  trace_hash={trace_hash}", job.job_id);

    if !success {
        // Save trace for debugging even on failure
        let _ = tokio::fs::write(work_dir.join("trace.log"), &trace).await;
        bail!("Pipeline exited non-zero");
    }
    tokio::fs::write(work_dir.join("trace.log"), &trace).await?;

    // ── 3. Hash outputs ───────────────────────────────────────────────────────
    log::info!("[{}] Hashing output files…", job.job_id);
    let outputs = hash_output_files(&output_dir).await?;
    let output_root = combined_output_root(&outputs);
    log::info!("[{}] {} outputs  output_root={}", job.job_id, outputs.len(), output_root);

    // ── 4. Build ComputeJobManifest ───────────────────────────────────────────
    let completed_at = now_secs();
    let manifest = ComputeJobManifest {
        job_id: job.job_id.clone(),
        job_type: job.job_type.clone(),
        input_root: actual_input_root,
        pipeline_hash: job.pipeline_hash.clone(),
        container_hash: job.container_hash.clone(),
        model_hash: job.model_hash.clone(),
        output_root: output_root.clone(),
        outputs,
        execution_trace_hash: Some(trace_hash),
        worker_pubkey: caps.worker_pubkey.clone(),
        completed_at,
    };

    // ── 5. Sign ───────────────────────────────────────────────────────────────
    let manifest_hash = manifest.hash_hex();
    let digest = manifest.hash_bytes();
    let worker_sig = sign_manifest(&digest, &cfg.privkey_hex).context("signing failed")?;
    log::info!("[{}] manifest_hash={manifest_hash}", job.job_id);

    // ── 6. Save manifest ──────────────────────────────────────────────────────
    tokio::fs::write(work_dir.join("manifest.json"), serde_json::to_vec_pretty(&manifest)?).await?;

    // ── 7. Build JobAnchorPayload ─────────────────────────────────────────────
    let anchor = JobAnchorPayload::new(&job.job_id, &manifest_hash, &output_root, &caps.worker_pubkey, &worker_sig);

    // ── 8. Anchor on chain ────────────────────────────────────────────────────
    if cfg.submit {
        submit_anchor(&cfg.node_addr, &anchor.to_payload_bytes(), &cfg.privkey_hex, cfg.devnet).await?;
        log::info!("[{}] Anchor submitted to {}", job.job_id, cfg.node_addr);
    } else {
        log::info!("[{}] Dry-run: anchor not submitted (pass --submit)", job.job_id);
    }

    log::info!("[{}] Done — output_root={output_root}", job.job_id);
    Ok(())
}

// ── Chain submission ──────────────────────────────────────────────────────────

async fn submit_anchor(node_addr: &str, payload_bytes: &[u8], privkey_hex: &str, devnet: bool) -> Result<()> {
    use std::sync::Arc;
    use kaspa_grpc_client::GrpcClient;

    use xenom_anchor_client::tx::{COINBASE_MATURITY, COINBASE_MATURITY_DEVNET, Prefix};

    let url = if node_addr.starts_with("grpc://") { node_addr.to_owned() } else { format!("grpc://{node_addr}") };
    let rpc = Arc::new(GrpcClient::connect(url).await.context("cannot connect to Xenom node")?);
    let keypair = xenom_anchor_client::keypair_from_hex(privkey_hex).context("invalid private key")?;

    let prefix = if devnet { Prefix::Devnet } else { Prefix::Mainnet };
    let maturity = if devnet { COINBASE_MATURITY_DEVNET } else { COINBASE_MATURITY };

    let txid = xenom_anchor_client::submit_anchor(&rpc, &keypair, payload_bytes, xenom_anchor_client::DEFAULT_FEE_PER_INPUT, prefix, maturity)
        .await
        .context("submit_anchor")?;
    log::info!("Anchor tx: {txid}");
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
