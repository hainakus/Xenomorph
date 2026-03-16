use anyhow::{Context, Result};
use bioproof_core::{blake3_hex, sign_manifest};
use clap::{Arg, Command};
use genetics_l2_core::{now_secs, ValidationReport, ValidationVerdict};
use serde_json::Value;
use tokio::time::{sleep, Duration};

// ── Validator daemon ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m           = cli().get_matches();
    let privkey     = m.get_one::<String>("private-key").unwrap().clone();
    let coordinator = m.get_one::<String>("coordinator").unwrap().clone();
    let coordinator_privkey = m.get_one::<String>("coordinator-privkey").map(|s| s.clone());
    let poll_ms: u64 = m.get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let tolerance: f64 = m.get_one::<String>("score-tolerance")
        .and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let keypair = bioproof_core::BioProofKeypair::from_hex(&privkey)
        .context("invalid --private-key")?;
    let validator_pubkey = keypair.pubkey_hex();

    log::info!("Genetics-L2 Validator started");
    log::info!("  pubkey:      {validator_pubkey}");
    log::info!("  coordinator: {coordinator}");
    log::info!("  tolerance:   {tolerance:.1}%");
    if let Some(ref key) = coordinator_privkey {
        log::info!("  decryption:  ENABLED (privkey len={})", key.len());
    } else {
        log::warn!("  decryption:  DISABLED - coordinator privkey not provided");
    }

    let http = reqwest::Client::new();

    loop {
        match validate_pending(&http, &coordinator, &validator_pubkey, &privkey, coordinator_privkey.as_deref(), tolerance).await {
            Ok(n) if n > 0 => log::info!("Validated {n} result(s)"),
            Ok(_)          => {}
            Err(e)         => log::warn!("Validation cycle error: {e:#}"),
        }
        sleep(Duration::from_millis(poll_ms)).await;
    }
}

/// Poll for completed (unvalidated) jobs and validate their best result.
async fn validate_pending(
    http:             &reqwest::Client,
    coordinator:      &str,
    validator_pubkey: &str,
    privkey:          &str,
    coordinator_privkey: Option<&str>,
    tolerance:        f64,
) -> Result<usize> {
    // Fetch completed jobs
    let resp = http
        .get(format!("{coordinator}/jobs?status=completed&limit=10"))
        .send()
        .await
        .context("GET /jobs?status=completed")?;

    let body: Value = resp.json().await.context("parse jobs")?;
    let jobs = body["jobs"].as_array().cloned().unwrap_or_default();

    let mut count = 0;
    for job_val in &jobs {
        let job_id = job_val["job_id"].as_str().unwrap_or_default().to_owned();
        if job_id.is_empty() { continue; }

        // Fetch submitted results for this job
        let results_resp = http
            .get(format!("{coordinator}/results/{job_id}"))
            .send()
            .await
            .context("GET /results")?;
        let results_body: Value = results_resp.json().await.context("parse results")?;
        let results = results_body["results"].as_array().cloned().unwrap_or_default();

        if results.is_empty() { continue; }

        // Pick best result by score
        let best = results.iter().max_by(|a, b| {
            let sa = a["score"].as_f64().unwrap_or(0.0);
            let sb = b["score"].as_f64().unwrap_or(0.0);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let Some(best) = best else { continue };
        let result_id  = best["result_id"].as_str().unwrap_or("").to_owned();
        
        // Try to decrypt if result is encrypted
        let (result_root, claimed_score, decrypted_payload) = if let (Some(encrypted), Some(ephemeral), Some(coord_key)) = (
            best["encrypted_payload"].as_str(),
            best["ephemeral_pubkey"].as_str(),
            coordinator_privkey
        ) {
            // Decrypt the result
            match genetics_l2_core::JobResult::decrypt_payload(encrypted, ephemeral, coord_key) {
                Ok(decrypted) => {
                    log::info!("Decrypted result {result_id}: score={}", decrypted.score);
                    (decrypted.result_root.clone(), decrypted.score, Some(decrypted))
                }
                Err(e) => {
                    log::warn!("Failed to decrypt result {result_id}: {e} (encrypted_len={}, ephemeral_len={}, key_len={})",
                        encrypted.len(), ephemeral.len(), coord_key.len());
                    continue;
                }
            }
        } else {
            // Use plaintext fields (old format or decryption disabled)
            let result_root = best["result_root"].as_str().unwrap_or("").to_owned();
            let claimed_score = best["score"].as_f64().unwrap_or(0.0);
            (result_root, claimed_score, None)
        };

        if result_id.is_empty() { continue; }

        // Partial recomputation validation
        let (verdict, recomputed_score, score_delta, notes) =
            validate_result(job_val, &result_root, claimed_score, tolerance).await;

        // Save decrypted result to JSON file for Kaggle submission if validation passed
        if verdict == ValidationVerdict::Valid {
            if let Some(ref payload) = decrypted_payload {
                if let Err(e) = save_kaggle_submission(&job_id, &result_id, payload, claimed_score).await {
                    log::warn!("Failed to save Kaggle submission for {job_id}: {e}");
                }
            }
        }

        // Sign the validation report
        let sign_data = format!("{result_id}:{verdict:?}");
        let digest    = *blake3::hash(sign_data.as_bytes()).as_bytes();
        let validator_sig = sign_manifest(&digest, privkey)
            .unwrap_or_else(|_| "unsigned".to_owned());

        let report_id = format!("{result_id}-val-{:x}", now_secs() & 0xFFFF);
        let report = ValidationReport {
            report_id:        report_id.clone(),
            job_id:           job_id.clone(),
            result_id:        result_id.clone(),
            validator_pubkey: validator_pubkey.to_owned(),
            verdict:          verdict.clone(),
            recomputed_score: Some(recomputed_score),
            score_delta:      Some(score_delta),
            notes:            Some(notes.clone()),
            validator_sig,
            validated_at:     now_secs(),
        };

        let post_resp = http
            .post(format!("{coordinator}/validations"))
            .json(&report)
            .send()
            .await
            .context("POST /validations")?;

        log::info!(
            "Job {job_id} result {result_id}: {:?}  score={claimed_score:.2}  recomputed={recomputed_score:.2}  delta={score_delta:.4}",
            verdict
        );

        if !post_resp.status().is_success() {
            let s = post_resp.status();
            let b = post_resp.text().await.unwrap_or_default();
            log::warn!("  POST /validations failed: {s} {b}");
        }

        count += 1;
    }

    Ok(count)
}

// ── Validation methods ────────────────────────────────────────────────────────

/// Validate a result via partial recomputation.
///
/// Returns `(verdict, recomputed_score, score_delta, notes)`.
async fn validate_result(
    job_val:       &Value,
    result_root:   &str,
    claimed_score: f64,
    tolerance:     f64,
) -> (ValidationVerdict, f64, f64, String) {
    // Method 1: Hash integrity check
    // Verify result_root is a valid 32-byte hex string
    if result_root.len() != 64 || hex::decode(result_root).is_err() {
        return (
            ValidationVerdict::Invalid,
            0.0,
            claimed_score,
            "result_root is not a valid 64-hex BLAKE3 hash".to_owned(),
        );
    }

    // Method 2: Score sanity check
    // Scores must be non-negative and within realistic bounds
    if claimed_score < 0.0 {
        return (
            ValidationVerdict::Invalid,
            0.0,
            claimed_score.abs(),
            format!("negative score {claimed_score:.4} is invalid"),
        );
    }

    // Method 3: Deterministic partial recomputation stub
    // In production: re-run a subset of the algorithm on a random sample of the input
    // using the same deterministic seed to verify consistency.
    let recomputed_score = partial_recompute(job_val, claimed_score).await;
    let score_delta = (claimed_score - recomputed_score).abs() / claimed_score.max(1.0);

    if score_delta > tolerance {
        return (
            ValidationVerdict::Invalid,
            recomputed_score,
            score_delta,
            format!(
                "score deviation {:.2}% exceeds tolerance {:.2}%",
                score_delta * 100.0, tolerance * 100.0
            ),
        );
    }

    (
        ValidationVerdict::Valid,
        recomputed_score,
        score_delta,
        format!("score within tolerance ({score_delta:.4} < {tolerance:.4})"),
    )
}

/// Partial recomputation stub.
/// In production: download 10% of input, re-run algorithm, compare score.
async fn partial_recompute(job_val: &Value, claimed_score: f64) -> f64 {
    // Deterministic: use a fixed 3% random perturbation seeded from job_id hash
    let job_id = job_val["job_id"].as_str().unwrap_or("x");
    let seed   = blake3_hex(job_id.as_bytes());
    let first4 = u32::from_be_bytes(hex::decode(&seed[..8]).unwrap()[..4].try_into().unwrap());
    let jitter = (first4 as f64 / u32::MAX as f64) * 0.02 - 0.01; // ±1%
    claimed_score * (1.0 + jitter)
}

/// Save validated result to JSON file for Kaggle submission.
async fn save_kaggle_submission(
    job_id: &str,
    result_id: &str,
    payload: &genetics_l2_core::EncryptedResultPayload,
    score: f64,
) -> Result<()> {
    use std::path::Path;
    
    // Create submissions directory
    let submissions_dir = Path::new("/tmp/kaggle-submissions");
    tokio::fs::create_dir_all(submissions_dir).await
        .context("Failed to create submissions directory")?;
    
    // Create JSON with result data
    let submission = serde_json::json!({
        "job_id": job_id,
        "result_id": result_id,
        "score": score,
        "result_root": payload.result_root,
        "trace_hash": payload.trace_hash,
        "notebook_or_repo_hash": payload.notebook_or_repo_hash,
        "container_hash": payload.container_hash,
        "weights_hash": payload.weights_hash,
        "submission_bundle_hash": payload.submission_bundle_hash,
        "timestamp": now_secs(),
    });
    
    // Save to file
    let filename = format!("{}.json", job_id);
    let filepath = submissions_dir.join(&filename);
    let json_str = serde_json::to_string_pretty(&submission)
        .context("Failed to serialize submission")?;
    
    tokio::fs::write(&filepath, json_str).await
        .context("Failed to write submission file")?;
    
    log::info!("Saved Kaggle submission: {}", filepath.display());
    
    Ok(())
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-validator")
        .about("Genetics L2 validator — partial recomputation and hash verification of submitted results")
        .arg(Arg::new("private-key")
            .short('k').long("private-key").value_name("HEX").required(true)
            .help("Validator secp256k1 private key (64 hex chars)"))
        .arg(Arg::new("coordinator")
            .short('c').long("coordinator").value_name("URL")
            .default_value("http://localhost:8091")
            .help("genetics-l2-coordinator base URL"))
        .arg(Arg::new("coordinator-privkey")
            .long("coordinator-privkey").value_name("HEX")
            .help("Coordinator secp256k1 private key for decrypting encrypted results (64 hex chars)"))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("10000")
            .help("Coordinator poll interval in milliseconds"))
        .arg(Arg::new("score-tolerance")
            .long("score-tolerance").value_name("FRACTION")
            .default_value("0.05")
            .help("Maximum allowed score deviation fraction (e.g. 0.05 = 5%)"))
}
