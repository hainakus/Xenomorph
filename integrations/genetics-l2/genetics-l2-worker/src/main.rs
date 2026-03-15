use anyhow::{bail, Context, Result};
use bioproof_core::{blake3_hex, compute_proof, merkle_root, sign_manifest, BioProofKeypair};
use clap::{Arg, Command};
use genetics_l2_core::{now_secs, Algorithm, JobResult, JobStatus, ScientificJob};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tokio::time::{sleep, Duration};

// ── Worker daemon ─────────────────────────────────────────────────────────────

struct WorkerConfig {
    coordinator_url: String,
    work_root:       PathBuf,
    privkey_hex:     String,
    worker_pubkey:   String,
    poll_ms:         u64,
}

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m           = cli().get_matches();
    let privkey     = m.get_one::<String>("private-key").unwrap().clone();
    let coordinator = m.get_one::<String>("coordinator").unwrap().clone();
    let work_root   = PathBuf::from(m.get_one::<String>("work-root").unwrap());
    let poll_ms: u64 = m.get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok()).unwrap_or(5_000);

    let keypair = BioProofKeypair::from_hex(&privkey)
        .context("invalid --private-key")?;
    let worker_pubkey = keypair.pubkey_hex();

    log::info!("Genetics-L2 Worker started");
    log::info!("  pubkey:      {worker_pubkey}");
    log::info!("  coordinator: {coordinator}");
    log::info!("  work_root:   {}", work_root.display());

    tokio::fs::create_dir_all(&work_root).await?;

    let http = reqwest::Client::new();
    let cfg  = WorkerConfig {
        coordinator_url: coordinator,
        work_root,
        privkey_hex:     privkey,
        worker_pubkey,
        poll_ms,
    };

    run_loop(&http, &cfg).await
}

async fn run_loop(http: &reqwest::Client, cfg: &WorkerConfig) -> Result<()> {
    loop {
        match try_claim_and_execute(http, cfg).await {
            Ok(true)  => {} // job processed
            Ok(false) => {} // no open jobs
            Err(e)    => log::warn!("Worker cycle error: {e:#}"),
        }
        sleep(Duration::from_millis(cfg.poll_ms)).await;
    }
}

/// Returns `true` if a job was processed, `false` if the queue was empty.
async fn try_claim_and_execute(
    http: &reqwest::Client,
    cfg:  &WorkerConfig,
) -> Result<bool> {
    // ── 1. Fetch one open job ─────────────────────────────────────────────────
    let resp = http
        .get(format!("{}/jobs?status=open&limit=1", cfg.coordinator_url))
        .send()
        .await
        .context("GET /jobs")?;

    let body: Value = resp.json().await.context("parse /jobs")?;
    let jobs = body["jobs"].as_array().cloned().unwrap_or_default();
    if jobs.is_empty() {
        return Ok(false);
    }

    let job: ScientificJob = serde_json::from_value(jobs[0].clone())
        .context("parse job")?;
    log::info!("Found job: {} [{}]", job.job_id, job.algorithm);

    // ── 2. Claim it ───────────────────────────────────────────────────────────
    let claim_resp = http
        .post(format!("{}/jobs/{}/claim", cfg.coordinator_url, job.job_id))
        .json(&serde_json::json!({ "worker_pubkey": cfg.worker_pubkey }))
        .send()
        .await
        .context("POST /claim")?;

    if !claim_resp.status().is_success() {
        log::debug!("Claim failed for {} (already taken?)", job.job_id);
        return Ok(false);
    }
    log::info!("Claimed job {}", job.job_id);

    // ── 3. Prepare work directory ─────────────────────────────────────────────
    let work_dir   = cfg.work_root.join(&job.job_id);
    let output_dir = work_dir.join("output");
    tokio::fs::create_dir_all(&output_dir).await?;

    // ── 4. Download dataset (if URL provided) ─────────────────────────────────
    if let Some(ref url) = job.dataset_url {
        log::info!("Downloading dataset from {url}…");
        if let Err(e) = download_dataset(http, url, &work_dir.join("input")).await {
            log::warn!("Dataset download failed (proceeding with stub): {e:#}");
        }
    }

    // ── 5. Execute the algorithm ──────────────────────────────────────────────
    log::info!("Running algorithm: {}", job.algorithm);
    let (score, trace) = execute_algorithm(&job, &work_dir).await
        .unwrap_or_else(|e| {
            log::warn!("Algorithm failed: {e:#}");
            (0.0, format!("ERROR: {e}"))
        });

    let trace_hash = blake3_hex(trace.as_bytes());
    tokio::fs::write(work_dir.join("trace.log"), &trace).await.ok();
    log::info!("  score={score:.4}  trace_hash={trace_hash}");

    // ── 6. Hash output files ──────────────────────────────────────────────────
    let result_root = hash_output_dir(&output_dir).await
        .unwrap_or_else(|_| hex::encode([0u8; 32]));
    log::info!("  result_root={result_root}");

    // ── 7. Sign result ────────────────────────────────────────────────────────
    let sign_data = format!("{}:{}:{:.6}", job.job_id, result_root, score);
    let digest    = *blake3::hash(sign_data.as_bytes()).as_bytes();
    let worker_sig = sign_manifest(&digest, &cfg.privkey_hex)
        .unwrap_or_else(|_| "unsigned".to_owned());

    // ── 8. Submit result ──────────────────────────────────────────────────────
    let result_id = format!("{}-{}", job.job_id, &trace_hash[..8]);
    let result = JobResult {
        result_id:    result_id.clone(),
        job_id:       job.job_id.clone(),
        worker_pubkey: cfg.worker_pubkey.clone(),
        result_root:  result_root.clone(),
        score,
        trace_hash:              Some(trace_hash),
        notebook_or_repo_hash:   None,
        container_hash:          None,
        weights_hash:            None,
        submission_bundle_hash:  None,
        worker_sig,
        submitted_at: now_secs(),
    };

    let submit_resp = http
        .post(format!("{}/results", cfg.coordinator_url))
        .json(&result)
        .send()
        .await
        .context("POST /results")?;

    if submit_resp.status().is_success() {
        log::info!("Result {} submitted for job {}", result_id, job.job_id);
    } else {
        let s = submit_resp.status();
        let b = submit_resp.text().await.unwrap_or_default();
        log::warn!("Submit failed for {}: {s} {b}", job.job_id);
    }

    Ok(true)
}

// ── Algorithm execution ───────────────────────────────────────────────────────

/// Dispatch to the appropriate algorithm implementation.
/// Returns `(score, execution_trace)`.
async fn execute_algorithm(
    job:      &ScientificJob,
    work_dir: &Path,
) -> Result<(f64, String)> {
    let input_dir  = work_dir.join("input");
    let output_dir = work_dir.join("output");

    match &job.algorithm {
        Algorithm::SmithWaterman | Algorithm::SequenceAlignment => {
            smith_waterman_stub(&input_dir, &output_dir).await
        }
        Algorithm::VariantCalling => {
            variant_calling_stub(&input_dir, &output_dir).await
        }
        Algorithm::ProteinFolding => {
            protein_folding_stub(&input_dir, &output_dir).await
        }
        Algorithm::RnaExpression => {
            rna_expression_stub(&input_dir, &output_dir).await
        }
        Algorithm::AcousticClassification => {
            acoustic_classification(&input_dir, &output_dir).await
        }
        _ => {
            // Generic: run pipeline script if present, else return stub score
            generic_pipeline_stub(job, &input_dir, &output_dir).await
        }
    }
}

/// Smith-Waterman pairwise alignment stub.
/// In production: calls a SIMD-optimised SW library or BLAST.
async fn smith_waterman_stub(
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let mut score = 0.0f64;
    let mut trace = String::from("smith-waterman alignment\n");

    let files = collect_files(input_dir).await.unwrap_or_default();
    for f in &files {
        let data = tokio::fs::read(f).await.unwrap_or_default();
        // Stub: score = sum of G+C base counts (mock for deterministic output)
        let gc = data.iter().filter(|&&b| b == b'G' || b == b'C').count();
        score += gc as f64 / data.len().max(1) as f64;
        trace.push_str(&format!("  {} → gc_fraction={:.4}\n", f.display(), score));
    }

    // Write stub VCF output
    let vcf_content = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\n";
    tokio::fs::write(output_dir.join("alignment.vcf"), vcf_content).await?;

    Ok((score * 1000.0, trace))
}

async fn variant_calling_stub(
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let files = collect_files(input_dir).await.unwrap_or_default();
    let trace = format!("variant-calling on {} input files\n", files.len());
    let vcf   = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\n";
    tokio::fs::write(output_dir.join("variants.vcf"), vcf).await?;
    Ok((files.len() as f64 * 100.0, trace))
}

async fn protein_folding_stub(
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let files  = collect_files(input_dir).await.unwrap_or_default();
    let trace  = format!("protein-folding on {} sequences\n", files.len());
    let pdb    = "ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00\n";
    tokio::fs::write(output_dir.join("structure.pdb"), pdb).await?;
    Ok((0.85 * files.len() as f64, trace))
}

async fn rna_expression_stub(
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let files  = collect_files(input_dir).await.unwrap_or_default();
    let trace  = format!("rna-expression on {} files\n", files.len());
    let tsv    = "gene_id\tcount\nGENE1\t1234\nGENE2\t567\n";
    tokio::fs::write(output_dir.join("counts.tsv"), tsv).await?;
    Ok((files.len() as f64 * 50.0, trace))
}

/// Acoustic species classification for BirdCLEF-style tasks.
///
/// Production path: calls `birdnet-analyzer` (Python) on each OGG/WAV clip.
/// Stub path (no model present): returns a deterministic hash-based score.
async fn acoustic_classification(
    input_dir:  &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let files = collect_files(input_dir).await.unwrap_or_default();
    let mut trace = format!("acoustic-classification on {} audio file(s)\n", files.len());
    let mut all_predictions: Vec<serde_json::Value> = Vec::new();
    let mut score_acc = 0.0f64;

    for audio_path in &files {
        let ext = audio_path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext.to_lowercase().as_str(), "ogg" | "wav" | "mp3" | "flac") {
            continue;
        }

        // ── Try BirdNET-Analyzer (requires: pip install birdnet-analyzer) ──────
        let birdnet = tokio::process::Command::new("python3")
            .args([
                "-m", "birdnet_analyzer.analyze",
                "--input",  &audio_path.to_string_lossy(),
                "--output", &output_dir.to_string_lossy(),
                "--format", "json",
                "--min_conf", "0.1",
            ])
            .output()
            .await;

        match birdnet {
            Ok(out) if out.status.success() => {
                let stdout = String::from_utf8_lossy(&out.stdout).to_string();
                trace.push_str(&format!("  birdnet OK: {}\n", audio_path.display()));
                // Parse JSON predictions
                if let Ok(preds) = serde_json::from_str::<serde_json::Value>(&stdout) {
                    let conf = preds["detections"]
                        .as_array()
                        .and_then(|a| a.first())
                        .and_then(|d| d["confidence"].as_f64())
                        .unwrap_or(0.0);
                    score_acc += conf;
                    all_predictions.push(preds);
                }
            }
            Ok(out) => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                trace.push_str(&format!("  birdnet WARN {}: {}\n", audio_path.display(), stderr.trim()));
                // Stub score: BLAKE3 of audio bytes → deterministic float in [0,1]
                let bytes = tokio::fs::read(audio_path).await.unwrap_or_default();
                let hash  = blake3::hash(&bytes);
                let stub_conf = (hash.as_bytes()[0] as f64) / 255.0;
                score_acc += stub_conf;
                all_predictions.push(serde_json::json!({
                    "file": audio_path.to_string_lossy(),
                    "stub": true,
                    "confidence": stub_conf
                }));
            }
            Err(_) => {
                // birdnet-analyzer not installed — deterministic stub
                trace.push_str(&format!("  birdnet not found (stub): {}\n", audio_path.display()));
                let bytes = tokio::fs::read(audio_path).await.unwrap_or_default();
                let hash  = blake3::hash(if bytes.is_empty() { audio_path.to_str().unwrap_or("").as_bytes() } else { &bytes });
                let stub_conf = (hash.as_bytes()[0] as f64) / 255.0;
                score_acc += stub_conf;
                all_predictions.push(serde_json::json!({
                    "file": audio_path.to_string_lossy(),
                    "stub": true,
                    "confidence": stub_conf
                }));
            }
        }
    }

    // Write consolidated predictions JSON
    let out_json = serde_json::json!({
        "algorithm": "acoustic_classification",
        "files_processed": files.len(),
        "predictions": all_predictions
    });
    tokio::fs::write(
        output_dir.join("predictions.json"),
        serde_json::to_vec_pretty(&out_json)?,
    ).await?;

    let n = files.len().max(1);
    let score = score_acc / n as f64;
    trace.push_str(&format!("  mean_confidence={score:.4}\n"));
    Ok((score, trace))
}

async fn generic_pipeline_stub(
    job:        &ScientificJob,
    _input_dir: &Path,
    output_dir: &Path,
) -> Result<(f64, String)> {
    let trace = format!("generic pipeline for {} [{}]\n", job.job_id, job.algorithm);
    tokio::fs::write(output_dir.join("result.json"), serde_json::to_vec(job)?).await?;
    Ok((42.0, trace))
}

// ── Dataset download ──────────────────────────────────────────────────────────

async fn download_dataset(
    http:      &reqwest::Client,
    url:       &str,
    dest_dir:  &Path,
) -> Result<()> {
    tokio::fs::create_dir_all(dest_dir).await?;
    let resp = http.get(url).send().await.context("dataset download")?;
    if !resp.status().is_success() {
        bail!("dataset download {} → {}", url, resp.status());
    }
    let bytes = resp.bytes().await.context("dataset read")?;
    let fname = url.rsplit('/').next().unwrap_or("dataset.bin");
    tokio::fs::write(dest_dir.join(fname), &bytes).await?;
    log::info!("  downloaded {} bytes → {}", bytes.len(), dest_dir.display());
    Ok(())
}

// ── File hashing helpers ──────────────────────────────────────────────────────

async fn hash_output_dir(dir: &Path) -> Result<String> {
    let files = collect_files(dir).await?;
    if files.is_empty() {
        return Ok(hex::encode([0u8; 32]));
    }
    let mut leaves = Vec::new();
    for f in &files {
        let data = tokio::fs::read(f).await?;
        let (root, _, _) = compute_proof(&data, 0);
        let mut b = [0u8; 32];
        if let Ok(v) = hex::decode(&root) {
            if let Ok(arr) = v.try_into() { b = arr; }
        }
        leaves.push(b);
    }
    Ok(hex::encode(merkle_root(&leaves)))
}

async fn collect_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.exists() { return Ok(files); }
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        let mut rd = tokio::fs::read_dir(&cur).await?;
        while let Some(e) = rd.next_entry().await? {
            let p = e.path();
            if p.is_dir() { stack.push(p); } else { files.push(p); }
        }
    }
    files.sort();
    Ok(files)
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-worker")
        .about("Genetics L2 worker daemon — polls coordinator for jobs, executes algorithms, submits results")
        .arg(Arg::new("private-key")
            .short('k').long("private-key").value_name("HEX").required(true)
            .help("Worker secp256k1 private key (64 hex chars)"))
        .arg(Arg::new("coordinator")
            .short('c').long("coordinator").value_name("URL")
            .default_value("http://localhost:8091")
            .help("genetics-l2-coordinator base URL"))
        .arg(Arg::new("work-root")
            .short('w').long("work-root").value_name("DIR")
            .default_value("./work")
            .help("Working directory for job inputs/outputs"))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("5000")
            .help("Coordinator poll interval in milliseconds"))
}
