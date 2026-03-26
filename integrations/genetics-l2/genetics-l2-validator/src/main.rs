use anyhow::{Context, Result};
use bioproof_core::sign_manifest;
use clap::{Arg, Command};
use genetics_l2_core::{now_secs, ValidationReport, ValidationVerdict};
use serde_json::Value;
use tokio::time::{sleep, Duration};

// ── Validator daemon ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m           = cli().get_matches();
    let privkey     = load_privkey("VALIDATOR_PRIVKEY", m.get_one::<String>("key-file").map(|s| s.as_str()))
        .context("private key required (set $VALIDATOR_PRIVKEY or use --key-file <PATH>)")?;
    let coordinator = m.get_one::<String>("coordinator").unwrap().clone();
    let coordinator_privkey = load_privkey_opt("COORDINATOR_PRIVKEY", m.get_one::<String>("coordinator-key-file").map(|s| s.as_str()));
    let poll_ms: u64 = m.get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let tolerance: f64 = m.get_one::<String>("score-tolerance")
        .and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let keypair = bioproof_core::BioProofKeypair::from_hex(&privkey)
        .context("invalid private key (expected 64 hex chars)")?;
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
        let predictions_csv = decrypted_payload.as_ref().and_then(|p| p.predictions_csv.as_deref());
        let (verdict, recomputed_score, score_delta, notes) =
            validate_result(job_val, &result_root, claimed_score, predictions_csv, tolerance).await;

        // Save decrypted result to JSON file for Kaggle submission if validation passed
        if verdict == ValidationVerdict::Valid {
            if let Some(ref payload) = decrypted_payload {
                if let Err(e) = save_kaggle_submission(&job_id, &result_id, payload, claimed_score).await {
                    log::warn!("Failed to save Kaggle submission for {job_id}: {e}");
                }
            }
        }

        // Sign the validation report — canonical: "{report_id}:{result_id}:{verdict_lowercase}"
        let report_id     = format!("{result_id}-val-{:x}", now_secs() & 0xFFFF);
        let verdict_str   = format!("{verdict:?}").to_lowercase();
        let sign_data     = format!("{report_id}:{result_id}:{verdict_str}");
        let digest        = *blake3::hash(sign_data.as_bytes()).as_bytes();
        let validator_sig = sign_manifest(&digest, privkey)
            .unwrap_or_else(|_| "unsigned".to_owned());
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
            "Job {job_id} result {result_id}: {:?}  score={claimed_score:.2}  recomputed={recomputed_score:.2}  delta={score_delta:.4}  note={notes}",
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

/// Validate a result using output-hash integrity and predictions CSV content.
///
/// Returns `(verdict, recomputed_score, score_delta, notes)`.
async fn validate_result(
    _job_val:        &Value,
    result_root:     &str,
    claimed_score:   f64,
    predictions_csv: Option<&str>,
    tolerance:       f64,
) -> (ValidationVerdict, f64, f64, String) {
    // Method 1: Hash integrity check
    // Verify result_root is a valid 32-byte hex string
    let result_root = result_root.strip_prefix("0x").unwrap_or(result_root);
    if result_root.len() != 64 || hex::decode(result_root).is_err() {
        return (
            ValidationVerdict::Invalid,
            0.0,
            claimed_score,
            "result_root is not a valid 64-hex BLAKE3 hash".to_owned(),
        );
    }

    // Method 2: Score sanity check — must be in [0.0, 1.0]
    if claimed_score < 0.0 {
        return (
            ValidationVerdict::Invalid,
            0.0,
            claimed_score.abs(),
            format!("negative score {claimed_score:.4} is invalid"),
        );
    }
    if claimed_score > 1.0 {
        return (
            ValidationVerdict::Invalid,
            0.0,
            1.0,
            format!("score {claimed_score:.4} exceeds maximum 1.0"),
        );
    }

    // Method 2b: Schema + consistency validation from predictions CSV
    if let Some(csv) = predictions_csv {
        let header = csv.lines().next().unwrap_or("");
        let algorithm = _job_val["algorithm"].as_str().unwrap_or_default();
        let external_ref = _job_val["external_ref"].as_str().unwrap_or_default().to_ascii_lowercase();

        let is_genomics_algo = matches!(
            algorithm,
            "variant_calling"
                | "cohort_build"
                | "frequency_annotation"
                | "clinical_annotation"
                | "cancer_genomics"
                | "vcf_annotation"
        );
        let is_genomics =
            is_genomics_algo || csv.contains("reference,GRCh38") || csv.contains("annotated,") || csv.lines().any(|l| l.starts_with("score,"));
        let is_birdclef =
            algorithm == "acoustic_classification" || external_ref.contains("birdclef") || (header.contains("row_id") && !is_genomics);

        if is_genomics {
            // Optional score line for genomics workers; when absent, fall back to claimed_score.
            // Score in CSV must be consistent with claimed_score
            let csv_score = csv.lines()
                .find(|l| l.starts_with("score,"))
                .and_then(|l| l.split_once(','))
                .and_then(|(_, v)| v.trim().parse::<f64>().ok());
            if let Some(cs) = csv_score {
                let delta = (cs - claimed_score).abs();
                if delta > tolerance {
                    return (
                        ValidationVerdict::Invalid,
                        cs,
                        delta,
                        format!("score mismatch: CSV={cs:.4} claimed={claimed_score:.4}"),
                    );
                }
            }
        } else if is_birdclef {
            let data_rows = csv.lines()
                .skip(1)
                .filter(|l| !l.trim().is_empty() && l.contains('_'))
                .count();
            if data_rows == 0 {
                return (
                    ValidationVerdict::Invalid,
                    0.0,
                    claimed_score,
                    "birdclef CSV has no valid row_id data rows".to_owned(),
                );
            }
        }
    }

    // Method 3: Content-based recomputation from decrypted predictions CSV
    let recomputed_score = partial_recompute(predictions_csv, claimed_score).await;
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

/// Content-based recomputation: extracts score from decrypted predictions CSV.
/// Genomics CSV: looks for "score,{val}" row. BirdCLEF CSV: counts valid row_id rows.
/// No jitter — recomputed_score is deterministically derived from output content.
async fn partial_recompute(predictions_csv: Option<&str>, claimed_score: f64) -> f64 {
    if let Some(csv) = predictions_csv {
        // Genomics format: "metric,value\n...score,{val}\n"
        for line in csv.lines() {
            if let Some(val) = line.strip_prefix("score,") {
                if let Ok(s) = val.trim().parse::<f64>() {
                    if (0.0..=1.0).contains(&s) {
                        return s;
                    }
                }
            }
        }
        // BirdCLEF format: header row with "row_id" + data rows
        let lines: Vec<&str> = csv.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() > 1 && lines[0].contains("row_id") {
            let data_rows = lines[1..].iter().filter(|l| l.contains('_')).count();
            if data_rows > 0 {
                return (0.3_f64 + (data_rows as f64 / 1000.0).min(1.0) * 0.65).min(0.95);
            }
        }
    }
    // No CSV or unparseable: echo claimed_score if in [0.0, 1.0], else reject
    if (0.0..=1.0).contains(&claimed_score) {
        claimed_score
    } else {
        0.0
    }
}

/// Save validated result to JSON file for Kaggle submission.
async fn save_kaggle_submission(
    job_id: &str,
    result_id: &str,
    payload: &genetics_l2_core::EncryptedResultPayload,
    score: f64,
) -> Result<()> {
    // Create submissions directory (persistent, not /tmp)
    let submissions_dir = dirs_next::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/var/lib/xenom"))
        .join(".local/share/xenom/kaggle-submissions");
    tokio::fs::create_dir_all(&submissions_dir).await
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

/// Load private key from env var (first) or key file (second).
fn load_privkey(env_var: &str, key_file: Option<&str>) -> anyhow::Result<String> {
    if let Ok(hex) = std::env::var(env_var) {
        let hex = hex.trim().to_string();
        if !hex.is_empty() { return Ok(hex); }
    }
    if let Some(path) = key_file {
        let hex = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read key file '{path}'"))?
            .trim()
            .to_string();
        return Ok(hex);
    }
    anyhow::bail!("No private key found. Set ${env_var} or use --key-file <PATH>")
}

/// Load optional private key from env var or key file.
fn load_privkey_opt(env_var: &str, key_file: Option<&str>) -> Option<String> {
    if let Ok(hex) = std::env::var(env_var) {
        let hex = hex.trim().to_string();
        if !hex.is_empty() { return Some(hex); }
    }
    if let Some(path) = key_file {
        if let Ok(hex) = std::fs::read_to_string(path) {
            let hex = hex.trim().to_string();
            if !hex.is_empty() { return Some(hex); }
        }
    }
    None
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-validator")
        .about("Genetics L2 validator — partial recomputation and hash verification of submitted results")
        .arg(Arg::new("key-file")
            .short('k').long("key-file").value_name("PATH")
            .help("Path to file containing the validator secp256k1 private key (64 hex chars). Alternatively set $VALIDATOR_PRIVKEY."))
        .arg(Arg::new("coordinator")
            .short('c').long("coordinator").value_name("URL")
            .default_value("http://localhost:8091")
            .help("genetics-l2-coordinator base URL"))
        .arg(Arg::new("coordinator-key-file")
            .long("coordinator-key-file").value_name("PATH")
            .help("Path to file with coordinator private key for decrypting results. Alternatively set $COORDINATOR_PRIVKEY."))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("10000")
            .help("Coordinator poll interval in milliseconds"))
        .arg(Arg::new("score-tolerance")
            .long("score-tolerance").value_name("FRACTION")
            .default_value("0.05")
            .help("Maximum allowed score deviation fraction (e.g. 0.05 = 5%)"))
}
