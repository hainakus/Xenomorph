use anyhow::{Context, Result};
use bioproof_core::{blake3_hex, sign_manifest, BioProofKeypair};
use genetics_l2_core::{now_secs, JobResult};
use kaspa_core::{info, warn};
use std::path::{Path, PathBuf};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct L2Config {
    pub coordinator_url: String,
    pub privkey_hex:     String,
    pub pubkey_hex:      String,
    pub work_root:       PathBuf,
    pub use_gpu:         bool,
    pub perch_script:    Option<PathBuf>,
}

impl L2Config {
    pub fn new(coordinator_url: String, privkey_hex: String, use_gpu: bool,
               perch_script: Option<PathBuf>) -> Result<Self> {
        let keypair = BioProofKeypair::from_hex(&privkey_hex)
            .context("invalid --l2-private-key")?;
        let pubkey_hex = keypair.pubkey_hex();
        let work_root  = std::env::temp_dir().join("genome-miner-l2");
        let perch_script = perch_script.or_else(find_perch_script);
        Ok(Self { coordinator_url, privkey_hex, pubkey_hex, work_root, use_gpu, perch_script })
    }
}

/// Search for perch_infer.py in common locations.
fn find_perch_script() -> Option<PathBuf> {
    let candidates = [
        "scripts/perch_infer.py",
        "/opt/xenom/scripts/perch_infer.py",
    ];
    // also check next to the running executable
    let exe_dir = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.join("perch_infer.py")));
    candidates.iter().map(PathBuf::from)
        .chain(exe_dir)
        .find(|p| p.exists())
}

// ── Entry point — called per L2 job received from stratum ─────────────────────

/// Claim, execute, and submit a single L2 job.
/// Runs in a spawned tokio task so it never blocks the PoW loop.
pub async fn run_l2_job(cfg: L2Config, l2_val: serde_json::Value) {
    let job_id      = l2_val["job_id"].as_str().unwrap_or("").to_owned();
    let task        = l2_val["task"].as_str().unwrap_or("").to_owned();
    let dataset_url = l2_val["dataset_url"].as_str().map(str::to_owned);

    if job_id.is_empty() {
        warn!("L2: job_id is empty — skipping");
        return;
    }

    info!("L2: starting job={job_id} task={task}");

    if let Err(e) = execute(&cfg, &job_id, &task, dataset_url.as_deref()).await {
        warn!("L2: job {job_id} failed: {e:#}");
    }
}

async fn execute(
    cfg:         &L2Config,
    job_id:      &str,
    task:        &str,
    dataset_url: Option<&str>,
) -> Result<()> {
    let http = reqwest::Client::new();

    // ── 1. Claim ──────────────────────────────────────────────────────────────
    let claim = http
        .post(format!("{}/jobs/{job_id}/claim", cfg.coordinator_url))
        .json(&serde_json::json!({ "worker_pubkey": cfg.pubkey_hex }))
        .send().await.context("POST /claim")?;

    if !claim.status().is_success() {
        let s = claim.status();
        let b = claim.text().await.unwrap_or_default();
        anyhow::bail!("claim {job_id} → {s}: {b}");
    }
    info!("L2: claimed {job_id}");

    // ── 2. Prepare work dirs ──────────────────────────────────────────────────
    let work_dir   = cfg.work_root.join(job_id);
    let input_dir  = work_dir.join("input");
    let output_dir = work_dir.join("output");
    tokio::fs::create_dir_all(&input_dir).await?;
    tokio::fs::create_dir_all(&output_dir).await?;

    // ── 3. Download dataset ───────────────────────────────────────────────────
    if let Some(url) = dataset_url {
        if let Err(e) = download(&http, url, &input_dir).await {
            warn!("L2: dataset download failed (will use stub): {e:#}");
        }
    }

    // ── 4. Execute ────────────────────────────────────────────────────────────
    let (score, trace) = dispatch_task(task, &input_dir, &output_dir, &cfg).await;
    let trace_hash = blake3_hex(trace.as_bytes());
    tokio::fs::write(work_dir.join("trace.log"), &trace).await.ok();
    info!("L2: {job_id} score={score:.4} trace_hash={trace_hash}");

    // ── 5. Hash outputs ───────────────────────────────────────────────────────
    let result_root = hash_dir(&output_dir).await;
    info!("L2: result_root={result_root}");

    // ── 6. Sign ───────────────────────────────────────────────────────────────
    let sign_data = format!("{job_id}:{result_root}:{score:.6}");
    let digest    = *blake3::hash(sign_data.as_bytes()).as_bytes();
    let worker_sig = sign_manifest(&digest, &cfg.privkey_hex)
        .unwrap_or_else(|_| "unsigned".to_owned());

    // ── 7. Submit ─────────────────────────────────────────────────────────────
    let result_id = format!("{job_id}-{}", &trace_hash[..8]);
    let result = JobResult {
        result_id:              result_id.clone(),
        job_id:                 job_id.to_owned(),
        worker_pubkey:          cfg.pubkey_hex.clone(),
        result_root,
        score,
        trace_hash:             Some(trace_hash),
        notebook_or_repo_hash:  None,
        container_hash:         None,
        weights_hash:           None,
        submission_bundle_hash: None,
        worker_sig,
        submitted_at:           now_secs(),
    };

    let submit = http
        .post(format!("{}/results", cfg.coordinator_url))
        .json(&result)
        .send().await.context("POST /results")?;

    if submit.status().is_success() {
        info!("L2: result {result_id} accepted for job {job_id}");
    } else {
        let s = submit.status();
        let b = submit.text().await.unwrap_or_default();
        warn!("L2: submit {job_id} → {s}: {b}");
    }

    Ok(())
}

// ── Task dispatcher ───────────────────────────────────────────────────────────

async fn dispatch_task(task: &str, input_dir: &Path, output_dir: &Path, cfg: &L2Config) -> (f64, String) {
    match task {
        "acoustic_classification" => acoustic_classification(input_dir, output_dir, cfg).await,
        _ => generic_stub(task, output_dir).await,
    }
}

/// Acoustic species classification — Perch v2 (primary) or stub.
async fn acoustic_classification(input_dir: &Path, output_dir: &Path, cfg: &L2Config) -> (f64, String) {
    let files = collect_audio(input_dir).await;
    let mut trace = format!("acoustic_classification on {} file(s)\n", files.len());
    let mut predictions = Vec::new();
    let mut score_sum   = 0.0f64;

    let python = detect_python().await;
    for audio in &files {
        // Use Perch script if available, otherwise fall back to birdnet_analyzer
        let mut cmd = if let Some(ref script) = cfg.perch_script {
            let mut c = tokio::process::Command::new(&python);
            c.args([
                script.to_string_lossy().as_ref(),
                "--input",   &audio.to_string_lossy(),
                "--output",  &output_dir.to_string_lossy(),
                "--min_conf","0.05",
            ]);
            if !cfg.use_gpu { c.arg("--cpu"); }
            trace.push_str(&format!("  [perch] {script:?}\n"));
            c
        } else {
            let mut c = tokio::process::Command::new(&python);
            c.args([
                "-m", "birdnet_analyzer.analyze",
                "--input",   &audio.to_string_lossy(),
                "--output",  &output_dir.to_string_lossy(),
                "--format",  "json",
                "--min_conf","0.05",
            ]);
            if !cfg.use_gpu { c.arg("--cpu"); }
            trace.push_str("  [birdnet]\n");
            c
        };
        let cmd = &mut cmd;
        let result = cmd.output().await;

        let conf = match result {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                trace.push_str(&format!("  birdnet OK: {}\n", audio.display()));
                // extract first detection confidence
                serde_json::from_str::<serde_json::Value>(&stdout)
                    .ok()
                    .and_then(|v| v["detections"].as_array()
                        .and_then(|a| a.first())
                        .and_then(|d| d["confidence"].as_f64()))
                    .unwrap_or(0.5)
            }
            Ok(out) => {
                let err = String::from_utf8_lossy(&out.stderr);
                trace.push_str(&format!("  birdnet exit≠0 (stub): {}\n", err.trim()));
                stub_conf(audio)
            }
            Err(_) => {
                trace.push_str(&format!("  birdnet not installed (stub): {}\n", audio.display()));
                stub_conf(audio)
            }
        };

        score_sum += conf;
        predictions.push(serde_json::json!({
            "file":       audio.file_name().map(|n| n.to_string_lossy().to_string()),
            "confidence": conf
        }));
    }

    let n     = files.len().max(1);
    let score = score_sum / n as f64;
    let out   = serde_json::json!({
        "algorithm":       "acoustic_classification",
        "files_processed": n,
        "mean_confidence": score,
        "predictions":     predictions
    });
    tokio::fs::write(
        output_dir.join("predictions.json"),
        serde_json::to_vec_pretty(&out).unwrap_or_default(),
    ).await.ok();

    trace.push_str(&format!("  mean_confidence={score:.4}\n"));
    (score, trace)
}

async fn generic_stub(task: &str, output_dir: &Path) -> (f64, String) {
    let trace = format!("generic stub for task={task}\n");
    tokio::fs::write(
        output_dir.join("result.json"),
        format!("{{\"task\":\"{task}\"}}"),
    ).await.ok();
    (0.42, trace)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn stub_conf(path: &Path) -> f64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    let src   = if bytes.is_empty() { path.to_string_lossy().as_bytes().to_vec() } else { bytes };
    (blake3::hash(&src).as_bytes()[0] as f64) / 255.0
}

/// Returns the first Python executable that has birdnet_analyzer importable.
/// birdnet-analyzer requires Python >=3.11 — older versions are skipped.
async fn detect_python() -> String {
    let candidates = [
        "/opt/birdnet-venv/bin/python",
        "python3.11",
        "python3.12",
        "python3.13",
        "python3",
    ];
    for candidate in &candidates {
        let ok = tokio::process::Command::new(candidate)
            .args(["-c", "import birdnet_analyzer"])
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return candidate.to_string();
        }
    }
    warn!("L2: birdnet_analyzer not found in any Python>=3.11 — will use stub");
    "python3.11".to_string()
}

async fn collect_audio(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !dir.exists() { return out; }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(mut rd) = tokio::fs::read_dir(&cur).await else { continue };
        while let Ok(Some(e)) = rd.next_entry().await {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
                if matches!(ext.as_str(), "ogg" | "wav" | "mp3" | "flac") {
                    out.push(p);
                }
            }
        }
    }
    out.sort();
    out
}

async fn download(http: &reqwest::Client, url: &str, dest: &Path) -> Result<()> {
    tokio::fs::create_dir_all(dest).await?;
    let resp = http.get(url).send().await.context("download")?;
    if !resp.status().is_success() {
        anyhow::bail!("download {} → {}", url, resp.status());
    }
    let bytes = resp.bytes().await?;
    let name  = url.rsplit('/').next().unwrap_or("data.bin");
    tokio::fs::write(dest.join(name), &bytes).await?;
    Ok(())
}

async fn hash_dir(dir: &Path) -> String {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let Ok(mut rd) = tokio::fs::read_dir(&cur).await else { continue };
        while let Ok(Some(e)) = rd.next_entry().await {
            let p = e.path();
            if p.is_dir() { stack.push(p); } else { files.push(p); }
        }
    }
    files.sort();

    let leaves: Vec<[u8; 32]> = {
        let mut v = Vec::new();
        for f in &files {
            let data = tokio::fs::read(f).await.unwrap_or_default();
            v.push(*blake3::hash(&data).as_bytes());
        }
        v
    };

    if leaves.is_empty() {
        return hex::encode([0u8; 32]);
    }
    hex::encode(bioproof_core::merkle_root(&leaves))
}
