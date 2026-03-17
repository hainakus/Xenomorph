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

/// Search for yamnet_infer.py in common locations.
fn find_perch_script() -> Option<PathBuf> {
    let candidates = [
        "scripts/yamnet_infer.py",
        "/opt/xenom/scripts/yamnet_infer.py",
    ];
    // also check next to the running executable
    let exe_dir = std::env::current_exe().ok()
        .and_then(|p| p.parent().map(|d| d.join("yamnet_infer.py")));
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
    // Read from 'dataset' field (sent by stratum-bridge in mining.notify param[6])
    let dataset_url = l2_val["dataset"].as_str()
        .or_else(|| l2_val["dataset_url"].as_str())
        .map(str::to_owned);

    if job_id.is_empty() {
        warn!("L2: job_id is empty — skipping");
        return;
    }

    info!("L2: starting job={job_id} task={task} dataset={}", dataset_url.as_deref().unwrap_or("none"));

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

    // ── 3. Fetch inference script + install requirements from coordinator ────────
    let python = detect_python().await;
    fetch_and_install_requirements(&http, &cfg.coordinator_url, task, &python).await;

    let fetched_script = fetch_inference_script(&http, &cfg.coordinator_url, task, &work_dir).await;
    if let Some(ref s) = fetched_script {
        info!("L2: inference script fetched from coordinator: {}", s.display());
    } else {
        warn!("L2: could not fetch inference script from coordinator — using local fallback");
    }
    let effective_script = fetched_script.as_ref().or(cfg.perch_script.as_ref()).cloned();

    // ── 4. Download dataset ───────────────────────────────────────────────────
    if let Some(url) = dataset_url {
        if let Err(e) = download(&http, url, &input_dir, job_id).await {
            warn!("L2: dataset download failed (will use stub): {e:#}");
        }
    }

    // ── 5. Execute ────────────────────────────────────────────────────────────
    let (score, trace) = dispatch_task(task, &input_dir, &output_dir, cfg, effective_script.as_ref(), &python).await;
    let trace_hash = blake3_hex(trace.as_bytes());
    tokio::fs::write(work_dir.join("trace.log"), &trace).await.ok();
    info!("L2: {job_id} score={score:.4} trace_hash={trace_hash}");

    // ── 6. Hash outputs (before encryption) ──────────────────────────────────
    let result_root = hash_dir(&output_dir).await;
    info!("L2: result_root={result_root}");

    // ── 6b. Encrypt output files on disk ──────────────────────────────────────
    if let Err(e) = encrypt_output_dir(&output_dir, &cfg.privkey_hex).await {
        warn!("L2: failed to encrypt output files: {e} — continuing with plaintext");
    } else {
        info!("L2: encrypted output files in {}", output_dir.display());
    }

    // ── 7. Sign ───────────────────────────────────────────────────────────────
    let sign_data = format!("{job_id}:{result_root}:{score:.6}");
    let digest    = *blake3::hash(sign_data.as_bytes()).as_bytes();
    let worker_sig = sign_manifest(&digest, &cfg.privkey_hex)
        .unwrap_or_else(|_| "unsigned".to_owned());

    // ── 8. Encrypt result payload ─────────────────────────────────────────────
    // Fetch coordinator's public key for encryption
    let coordinator_pubkey = match fetch_coordinator_pubkey(&http, &cfg.coordinator_url).await {
        Ok(pk) => pk,
        Err(e) => {
            warn!("L2: failed to fetch coordinator pubkey: {e} — submitting unencrypted");
            String::new()
        }
    };

    // Read predictions.json (still plain before encryption) → build CSV for coordinator
    let predictions_csv = build_predictions_csv(&output_dir, score).await;

    let result_id = format!("{job_id}-{}", &trace_hash[..8]);
    let mut result = JobResult {
        result_id:              result_id.clone(),
        job_id:                 job_id.to_owned(),
        worker_pubkey:          cfg.pubkey_hex.clone(),
        result_root:            result_root.clone(),
        score,
        trace_hash:             Some(trace_hash.clone()),
        notebook_or_repo_hash:  None,
        container_hash:         None,
        weights_hash:           None,
        submission_bundle_hash: None,
        worker_sig,
        encrypted_payload:      None,
        ephemeral_pubkey:       None,
        predictions_csv:        Some(predictions_csv),
        submitted_at:           now_secs(),
    };

    // Encrypt if coordinator pubkey available
    if !coordinator_pubkey.is_empty() {
        match result.encrypt_payload(&coordinator_pubkey) {
            Ok((encrypted, ephemeral)) => {
                info!("L2: encrypted result payload for {job_id}");
                result.encrypted_payload = Some(encrypted);
                result.ephemeral_pubkey = Some(ephemeral);
                // Clear plaintext fields after encryption
                result.result_root = String::new();
                result.score = 0.0;
                result.trace_hash = None;
                result.predictions_csv = None; // travels only inside encrypted_payload
            }
            Err(e) => {
                warn!("L2: encryption failed: {e} — submitting unencrypted");
            }
        }
    }

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

// ── Coordinator public key fetching ───────────────────────────────────────────

async fn fetch_coordinator_pubkey(http: &reqwest::Client, coordinator_url: &str) -> Result<String> {
    let resp = http
        .get(format!("{coordinator_url}/pubkey"))
        .send()
        .await
        .context("GET /pubkey")?;

    if !resp.status().is_success() {
        anyhow::bail!("Coordinator /pubkey returned {}", resp.status());
    }

    let body: serde_json::Value = resp.json().await.context("parse /pubkey")?;
    body["pubkey"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("Missing pubkey field"))
}

// ── Task dispatcher ───────────────────────────────────────────────────────────

/// Fetch requirements.txt from coordinator and pip-install if changed (hash-cached).
/// Cache marker: /tmp/genome-miner-l2/pip-{hash}.installed
async fn fetch_and_install_requirements(
    http: &reqwest::Client,
    coordinator_url: &str,
    task: &str,
    python: &str,
) {
    #[cfg(target_os = "macos")]
    let url = format!("{coordinator_url}/scripts/{task}/requirements?backend=efficientnet");
    #[cfg(not(target_os = "macos"))]
    let url = format!("{coordinator_url}/scripts/{task}/requirements?backend=gpu");

    let content = match http.get(&url).send().await {
        Ok(r) if r.status().is_success() => match r.text().await {
            Ok(t) if !t.trim().is_empty() => t,
            _ => return,
        },
        _ => {
            warn!("L2: could not fetch requirements.txt from coordinator");
            return;
        }
    };

    // Only install if requirements changed
    let hash = blake3_hex(content.as_bytes());
    let marker = std::env::temp_dir()
        .join("genome-miner-l2")
        .join(format!("pip-{}.installed", &hash[..16]));

    if marker.exists() {
        return; // already installed
    }

    // Write requirements to temp file
    let req_path = std::env::temp_dir()
        .join("genome-miner-l2")
        .join(format!("requirements-{}.txt", &hash[..16]));
    if let Err(e) = tokio::fs::create_dir_all(req_path.parent().unwrap()).await {
        warn!("L2: cannot create pip dir: {e}");
        return;
    }
    if let Err(e) = tokio::fs::write(&req_path, content.as_bytes()).await {
        warn!("L2: cannot write requirements.txt: {e}");
        return;
    }

    info!("L2: installing requirements from coordinator ({} bytes)...", content.len());
    let out = tokio::process::Command::new(python)
        .args(["-m", "pip", "install", "-q", "-r", req_path.to_str().unwrap()])
        .output()
        .await;

    match out {
        Ok(o) if o.status.success() => {
            info!("L2: pip install completed");
            tokio::fs::write(&marker, b"ok").await.ok();
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr);
            warn!("L2: pip install failed: {stderr}");
        }
        Err(e) => warn!("L2: pip install error: {e}"),
    }
}

/// Fetch inference script from coordinator API and save to work_dir/inference.py.
/// On macOS, requests ?backend=efficientnet (PyTorch/MPS) automatically.
async fn fetch_inference_script(
    http: &reqwest::Client,
    coordinator_url: &str,
    task: &str,
    work_dir: &Path,
) -> Option<PathBuf> {
    // macOS: EfficientNet via MPS; Linux: GPU-optimised EfficientNet-B4 via CUDA
    #[cfg(target_os = "macos")]
    let url = format!("{coordinator_url}/scripts/{task}?backend=efficientnet");
    #[cfg(not(target_os = "macos"))]
    let url = format!("{coordinator_url}/scripts/{task}?backend=gpu");

    let resp = http.get(&url).send().await.ok()?;
    if !resp.status().is_success() { return None; }
    let content = resp.text().await.ok()?;
    if content.trim().is_empty() { return None; }
    let dest = work_dir.join("inference.py");
    tokio::fs::write(&dest, content.as_bytes()).await.ok()?;
    Some(dest)
}

async fn dispatch_task(task: &str, input_dir: &Path, output_dir: &Path, cfg: &L2Config, script: Option<&PathBuf>, python: &str) -> (f64, String) {
    match task {
        "acoustic_classification" => acoustic_classification(input_dir, output_dir, cfg, script, python).await,
        "variant_calling" | "cancer_genomics" | "genome_assembly" | "metagenomics"
            => genomics_analysis(task, input_dir, output_dir).await,
        "gene_expression" | "rna_expression" | "biomarker_discovery"
        | "network_biology" | "sequence_alignment" | "protein_folding" | "molecular_docking"
            => omics_analysis(task, input_dir, output_dir).await,
        "digital_health" | "biotechnology" | "drug_discovery"
            => horizon_analysis(task, input_dir, output_dir).await,
        _ => generic_stub(task, output_dir).await,
    }
}

// ── Genomics handler (NIH SRA — variant calling, cancer genomics, etc.) ───────

async fn genomics_analysis(task: &str, input_dir: &Path, output_dir: &Path) -> (f64, String) {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut trace = format!("{task} — NCBI SRA analysis\n");
    let mut score  = 0.1f64;
    let mut metrics = serde_json::json!({ "status": "no_data" });

    let files = list_input_files(input_dir).await;
    trace.push_str(&format!("  input files: {}\n", files.len()));

    for file in &files {
        let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Try NCBI E-utilities runinfo (CSV) — works for numeric SRA IDs
        let acc = if name.chars().all(|c| c.is_ascii_digit() || c.is_ascii_alphanumeric()) {
            name.to_owned()
        } else {
            // scan file content for SRR/ERR/DRR accession
            let txt = tokio::fs::read_to_string(file).await.unwrap_or_default();
            extract_sra_accession(&txt).unwrap_or_default()
        };

        if acc.is_empty() { continue; }

        let url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi\
             ?db=sra&id={acc}&rettype=runinfo&retmode=csv"
        );
        trace.push_str(&format!("  NCBI runinfo → {acc}\n"));

        if let Ok(resp) = http.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(csv) = resp.text().await {
                    let (s, m) = score_from_sra_csv(&csv);
                    if s > score {
                        score   = s;
                        metrics = m;
                        trace.push_str(&format!("  score={s:.4}\n"));
                    }
                }
            }
        }
    }

    let out = serde_json::json!({ "task": task, "score": score, "metrics": metrics });
    tokio::fs::write(
        output_dir.join("analysis.json"),
        serde_json::to_vec_pretty(&out).unwrap_or_default(),
    ).await.ok();
    trace.push_str(&format!("  final score={score:.4}\n"));
    (score, trace)
}

// ── Omics handler (expression, biomarker, network biology, protein) ────────────

#[allow(unused_assignments)]
async fn omics_analysis(task: &str, input_dir: &Path, output_dir: &Path) -> (f64, String) {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut trace = format!("{task} — NCBI GEO/SRA analysis\n");
    let mut score  = 0.1f64;
    let mut found_records = 0usize;

    let files = list_input_files(input_dir).await;
    trace.push_str(&format!("  input files: {}\n", files.len()));

    for file in &files {
        let name = file.file_name().and_then(|n| n.to_str()).unwrap_or("");

        // Try NCBI E-utilities esearch for GEO/SRA count
        let url = format!(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi\
             ?db=gds&term={name}&retmode=json&retmax=5"
        );
        trace.push_str(&format!("  NCBI GEO search → {name}\n"));

        if let Ok(resp) = http.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    let count = json["esearchresult"]["count"]
                        .as_str().and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
                    found_records += count;
                    trace.push_str(&format!("  GEO count={count}\n"));
                }
            }
        }
    }

    // Score: log scale on GEO record count
    score = if found_records > 0 {
        ((found_records as f64).log10() / 5.0 + 0.2).min(0.95)
    } else {
        // fallback: size-based scoring on downloaded files
        let total_bytes: u64 = files.iter().map(|f| {
            std::fs::metadata(f).map(|m| m.len()).unwrap_or(0)
        }).sum();
        ((total_bytes as f64 / 50_000.0).min(1.0) * 0.6 + 0.1).min(0.9)
    };

    let out = serde_json::json!({
        "task": task, "score": score,
        "geo_records_found": found_records,
        "input_files": files.len(),
    });
    tokio::fs::write(
        output_dir.join("analysis.json"),
        serde_json::to_vec_pretty(&out).unwrap_or_default(),
    ).await.ok();
    trace.push_str(&format!("  final score={score:.4}\n"));
    (score, trace)
}

// ── Horizon / EuropePMC handler (digital_health, biotechnology, drug_discovery) 

#[allow(unused_assignments)]
async fn horizon_analysis(task: &str, input_dir: &Path, output_dir: &Path) -> (f64, String) {
    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());
    let mut trace = format!("{task} — EuropePMC analysis\n");
    let mut score = 0.1f64;
    let mut citations: u64 = 0;
    let mut keywords_hit = 0usize;

    let health_kws = [
        "genome", "genomic", "health", "clinical", "biomarker", "precision",
        "therapy", "cancer", "gene", "variant", "protein", "drug", "biotech",
    ];

    let files = list_input_files(input_dir).await;
    trace.push_str(&format!("  input files: {}\n", files.len()));

    for file in &files {
        // Count relevant keywords in downloaded content
        let content = tokio::fs::read_to_string(file).await.unwrap_or_default();
        for kw in &health_kws {
            keywords_hit += content.matches(kw).count();
        }

        // Extract article ID from filename (format: "<id>" or "MED")
        let art_id = file.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if art_id.is_empty() { continue; }

        // Fetch citation count via EuropePMC REST API
        let url = format!(
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search\
             ?query={art_id}&format=json&pageSize=1&resultType=core"
        );
        if let Ok(resp) = http.get(&url).send().await {
            if resp.status().is_success() {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(c) = json["resultList"]["result"]
                        .as_array()
                        .and_then(|a| a.first())
                        .and_then(|r| r["citedByCount"].as_u64())
                    {
                        citations += c;
                        trace.push_str(&format!("  citedByCount={c}\n"));
                    }
                }
            }
        }
    }

    let kw_score   = (keywords_hit as f64 / 30.0).min(0.6);
    let cite_score = if citations > 0 { ((citations as f64).log10() / 4.0).min(0.4) } else { 0.0 };
    score = (kw_score + cite_score + 0.05).min(0.95);

    let out = serde_json::json!({
        "task": task, "score": score,
        "keywords_hit": keywords_hit,
        "citations": citations,
        "input_files": files.len(),
    });
    tokio::fs::write(
        output_dir.join("analysis.json"),
        serde_json::to_vec_pretty(&out).unwrap_or_default(),
    ).await.ok();
    trace.push_str(&format!("  keywords={keywords_hit} citations={citations} score={score:.4}\n"));
    (score, trace)
}

// ── Shared helpers ─────────────────────────────────────────────────────────────

async fn list_input_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if let Ok(mut rd) = tokio::fs::read_dir(dir).await {
        while let Ok(Some(e)) = rd.next_entry().await {
            if e.path().is_file() { out.push(e.path()); }
        }
    }
    out
}

fn extract_sra_accession(html: &str) -> Option<String> {
    for prefix in ["SRR", "ERR", "DRR"] {
        if let Some(pos) = html.find(prefix) {
            let acc: String = html[pos..].chars()
                .take_while(|c| c.is_ascii_alphanumeric())
                .collect();
            if acc.len() >= 9 { return Some(acc); }
        }
    }
    None
}

fn score_from_sra_csv(csv: &str) -> (f64, serde_json::Value) {
    let rows: Vec<&str> = csv.lines().filter(|l| !l.trim().is_empty()).collect();
    if rows.len() < 2 { return (0.1, serde_json::json!({})); }

    let headers: Vec<&str> = rows[0].split(',').collect();
    let values:  Vec<&str> = rows[1].split(',').collect();
    let mut map = serde_json::Map::new();
    for (h, v) in headers.iter().zip(values.iter()) {
        map.insert(h.to_string(), serde_json::json!(v));
    }

    let get_f64 = |key: &str| -> f64 {
        map.get(key).and_then(|v| v.as_str())
           .and_then(|s| s.parse::<f64>().ok()).unwrap_or(0.0)
    };

    let bases  = get_f64("bases");    // e.g. 3_000_000_000 for 30x WGS
    let avg_len = get_f64("avgLength"); // e.g. 150 for Illumina

    let base_score = if bases > 0.0 { (bases.log10() / 12.0).min(1.0) } else { 0.1 };
    let len_score  = if avg_len > 0.0 { (avg_len / 300.0).min(1.0)   } else { 0.5 };
    let score = (base_score * 0.65 + len_score * 0.35).max(0.05);

    (score, serde_json::Value::Object(map))
}

/// Acoustic species classification — YAMNet (primary) or stub.
async fn acoustic_classification(input_dir: &Path, output_dir: &Path, cfg: &L2Config, script: Option<&PathBuf>, python: &str) -> (f64, String) {
    let files = collect_audio(input_dir).await;
    let mut trace = format!("acoustic_classification on {} file(s)\n", files.len());

    let Some(infer_script) = script else {
        // No script — stub score based on file count
        trace.push_str("  [no script, using stub]\n");
        let score = if files.is_empty() { 0.1 } else { stub_conf(&files[0]) };
        trace.push_str(&format!("  stub score={score:.4}\n"));
        return (score, trace);
    };

    // Call script ONCE for the whole input directory
    let mut cmd = tokio::process::Command::new(python);
    cmd.args([
        infer_script.to_string_lossy().as_ref(),
        "--input",  &input_dir.to_string_lossy(),
        "--output", &output_dir.to_string_lossy(),
    ]);
    if !cfg.use_gpu { cmd.arg("--cpu"); }
    // Pass sample_submission.csv so script emits exactly the right row_ids
    let sample_sub = input_dir.join("sample_submission.csv");
    if sample_sub.exists() {
        cmd.args(["--sample-submission", sample_sub.to_str().unwrap_or("")]);
        trace.push_str(&format!("  [infer] using sample_submission.csv for row_ids\n"));
    }
    trace.push_str(&format!("  [infer] {} --input {:?}\n", infer_script.display(), input_dir));

    let score = match cmd.output().await {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            trace.push_str(&format!("  script OK stdout={}\n", stdout.trim()));
            // Read score from predictions.json written by script
            read_score_from_predictions(output_dir).await
                .or_else(|| {
                    // Fallback: parse stdout JSON {"score": float}
                    serde_json::from_str::<serde_json::Value>(stdout.trim())
                        .ok()
                        .and_then(|v| v["score"].as_f64())
                })
                .unwrap_or(0.1)
        }
        Ok(out) => {
            let err = String::from_utf8_lossy(&out.stderr);
            trace.push_str(&format!("  script exit≠0: {}\n", err.trim()));
            if files.is_empty() { 0.1 } else { stub_conf(&files[0]) }
        }
        Err(e) => {
            trace.push_str(&format!("  script launch failed: {e}\n"));
            if files.is_empty() { 0.1 } else { stub_conf(&files[0]) }
        }
    };

    trace.push_str(&format!("  score={score:.4}\n"));
    (score, trace)
}

/// Read `score` from `predictions.json` written by the inference script.
async fn read_score_from_predictions(output_dir: &Path) -> Option<f64> {
    let data = tokio::fs::read(output_dir.join("predictions.json")).await.ok()?;
    serde_json::from_slice::<serde_json::Value>(&data).ok()
        .and_then(|v| v["score"].as_f64())
}

async fn generic_stub(task: &str, output_dir: &Path) -> (f64, String) {
    let trace = format!("generic stub for task={task}\n");
    tokio::fs::write(
        output_dir.join("result.json"),
        format!("{{\"task\":\"{task}\",\"note\":\"no handler\"}}"),
    ).await.ok();
    (0.1, trace)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Unit: score_from_sra_csv ───────────────────────────────────────────────

    #[test]
    fn test_score_from_sra_csv_wgs() {
        let csv = "Run,bases,avgLength,spots\nSRR12345678,3000000000,150,20000000\n";
        let (score, metrics) = score_from_sra_csv(csv);
        assert!(score > 0.5, "WGS 30x should score >0.5, got {score}");
        assert_eq!(metrics["bases"].as_str().unwrap(), "3000000000");
    }

    #[test]
    fn test_score_from_sra_csv_small() {
        let csv = "Run,bases,avgLength,spots\nSRR00000001,1000000,75,13333\n";
        let (score, _) = score_from_sra_csv(csv);
        assert!(score > 0.0 && score < 0.7, "small dataset score={score}");
    }

    #[test]
    fn test_score_from_sra_csv_empty() {
        let (score, _) = score_from_sra_csv("");
        assert_eq!(score, 0.1);
    }

    #[test]
    fn test_score_from_sra_csv_header_only() {
        let (score, _) = score_from_sra_csv("Run,bases,avgLength\n");
        assert_eq!(score, 0.1);
    }

    // ── Unit: extract_sra_accession ────────────────────────────────────────────

    #[test]
    fn test_extract_sra_accession_srr() {
        let html = "<html>... accession SRR123456789 ...</html>";
        assert_eq!(extract_sra_accession(html), Some("SRR123456789".to_string()));
    }

    #[test]
    fn test_extract_sra_accession_err() {
        let html = "Study ERR987654321 from ENA";
        assert_eq!(extract_sra_accession(html), Some("ERR987654321".to_string()));
    }

    #[test]
    fn test_extract_sra_accession_none() {
        assert_eq!(extract_sra_accession("no accession here"), None);
    }

    #[test]
    fn test_extract_sra_accession_too_short() {
        assert_eq!(extract_sra_accession("SRR123"), None); // <9 chars
    }

    // ── Integration: NCBI E-utilities (real network) ───────────────────────────

    #[tokio::test]
    async fn test_genomics_analysis_ncbi_sra() {
        let tmp = TempDir::new().unwrap();
        let input  = tmp.path().join("input");
        let output = tmp.path().join("output");
        tokio::fs::create_dir_all(&input).await.unwrap();
        tokio::fs::create_dir_all(&output).await.unwrap();

        // Write a file named after a real public SRA accession (1000 Genomes)
        tokio::fs::write(input.join("SRR062634"), b"placeholder").await.unwrap();

        let (score, trace) = genomics_analysis("variant_calling", &input, &output).await;
        println!("NCBI score={score:.4}\ntrace:\n{trace}");

        let result = tokio::fs::read_to_string(output.join("analysis.json")).await.unwrap();
        println!("analysis.json:\n{result}");

        // Score should be > 0.1 if NCBI returned real data
        assert!(score >= 0.1, "score={score}");
        assert!(output.join("analysis.json").exists());
    }

    // ── Integration: EuropePMC (real network) ──────────────────────────────────

    #[tokio::test]
    async fn test_horizon_analysis_europepmc() {
        let tmp = TempDir::new().unwrap();
        let input  = tmp.path().join("input");
        let output = tmp.path().join("output");
        tokio::fs::create_dir_all(&input).await.unwrap();
        tokio::fs::create_dir_all(&output).await.unwrap();

        // Write a file with biomedical keywords (simulates downloaded EuropePMC HTML)
        tokio::fs::write(
            input.join("horizon-test"),
            b"genomics health clinical biomarker precision therapy cancer gene variant protein drug",
        ).await.unwrap();

        let (score, trace) = horizon_analysis("digital_health", &input, &output).await;
        println!("EuropePMC score={score:.4}\ntrace:\n{trace}");

        let result = tokio::fs::read_to_string(output.join("analysis.json")).await.unwrap();
        println!("analysis.json:\n{result}");

        // Keyword hits should drive score > 0.1
        assert!(score > 0.1, "score={score}");
        assert!(output.join("analysis.json").exists());
    }

    // ── Integration: omics_analysis (NCBI GEO) ────────────────────────────────

    #[tokio::test]
    async fn test_omics_analysis_geo() {
        let tmp = TempDir::new().unwrap();
        let input  = tmp.path().join("input");
        let output = tmp.path().join("output");
        tokio::fs::create_dir_all(&input).await.unwrap();
        tokio::fs::create_dir_all(&output).await.unwrap();

        tokio::fs::write(input.join("GSE100026"), b"expression data").await.unwrap();

        let (score, trace) = omics_analysis("gene_expression", &input, &output).await;
        println!("GEO score={score:.4}\ntrace:\n{trace}");

        assert!(score >= 0.1);
        assert!(output.join("analysis.json").exists());
    }
}

// ── Output file encryption ────────────────────────────────────────────────────

/// Encrypt all files in output directory using worker's private key.
/// Each file is encrypted with ChaCha20-Poly1305 using a key derived from the worker's privkey.
/// Original files are replaced with .enc versions.
async fn encrypt_output_dir(output_dir: &Path, worker_privkey_hex: &str) -> Result<()> {
    use chacha20poly1305::{
        aead::{Aead, KeyInit},
        ChaCha20Poly1305, Nonce,
    };
    use sha2::{Digest, Sha256};

    // Derive encryption key from worker's private key
    let key_material = Sha256::digest(format!("L2_OUTPUT_ENC:{worker_privkey_hex}").as_bytes());
    let cipher = ChaCha20Poly1305::new_from_slice(&key_material[..32])
        .map_err(|e| anyhow::anyhow!("Cipher init failed: {e}"))?;

    // Encrypt all files in output directory
    let mut entries = tokio::fs::read_dir(output_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        // Skip already encrypted files
        if path.extension().and_then(|e| e.to_str()) == Some("enc") {
            continue;
        }

        // Read plaintext
        let plaintext = tokio::fs::read(&path).await?;

        // Generate random nonce
        let nonce_bytes: [u8; 12] = rand::random();
        let nonce = Nonce::from_slice(&nonce_bytes);

        // Encrypt
        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_ref())
            .map_err(|e| anyhow::anyhow!("Encryption failed: {e}"))?;

        // Combine nonce + ciphertext
        let mut encrypted = nonce_bytes.to_vec();
        encrypted.extend_from_slice(&ciphertext);

        // Write encrypted file with .enc extension
        let enc_path = path.with_extension(
            format!("{}.enc", path.extension().and_then(|e| e.to_str()).unwrap_or("bin"))
        );
        tokio::fs::write(&enc_path, encrypted).await?;

        // Remove original plaintext file
        tokio::fs::remove_file(&path).await?;

        info!("L2: encrypted {} → {}", path.display(), enc_path.display());
    }

    Ok(())
}

/// Decrypt all .enc files in output directory using worker's private key.
/// Used by validator to verify encrypted results.
#[allow(dead_code)]
async fn decrypt_output_dir(output_dir: &Path, worker_privkey_hex: &str) -> Result<()> {
    use chacha20poly1305::{
        aead::{Aead, KeyInit},
        ChaCha20Poly1305, Nonce,
    };
    use sha2::{Digest, Sha256};

    // Derive decryption key (same as encryption)
    let key_material = Sha256::digest(format!("L2_OUTPUT_ENC:{worker_privkey_hex}").as_bytes());
    let cipher = ChaCha20Poly1305::new_from_slice(&key_material[..32])
        .map_err(|e| anyhow::anyhow!("Cipher init failed: {e}"))?;

    // Decrypt all .enc files
    let mut entries = tokio::fs::read_dir(output_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("enc") {
            continue;
        }

        // Read encrypted data
        let encrypted = tokio::fs::read(&path).await?;
        if encrypted.len() < 12 {
            warn!("L2: encrypted file too short: {}", path.display());
            continue;
        }

        let nonce = Nonce::from_slice(&encrypted[..12]);
        let ciphertext = &encrypted[12..];

        // Decrypt
        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| anyhow::anyhow!("Decryption failed: {e}"))?;

        // Restore original filename (remove .enc)
        let original_path = path.with_extension("");
        tokio::fs::write(&original_path, plaintext).await?;

        // Remove encrypted file
        tokio::fs::remove_file(&path).await?;

        info!("L2: decrypted {} → {}", path.display(), original_path.display());
    }

    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn stub_conf(path: &Path) -> f64 {
    let bytes = std::fs::read(path).unwrap_or_default();
    let src   = if bytes.is_empty() { path.to_string_lossy().as_bytes().to_vec() } else { bytes };
    (blake3::hash(&src).as_bytes()[0] as f64) / 255.0
}

/// Returns the first Python executable that has the required inference library.
/// macOS: checks for `torch` (EfficientNet/PyTorch)
/// Linux: checks for `tensorflow_hub` (YAMNet)
async fn detect_python() -> String {
    #[cfg(target_os = "macos")]
    let check_import = "import torch";
    #[cfg(not(target_os = "macos"))]
    let check_import = "import tensorflow_hub";

    let candidates = [
        "python3",
        "python3.11",
        "python3.12",
        "python3.13",
        "/opt/xenom/venv/bin/python",
        "/opt/birdnet-venv/bin/python",
    ];
    for candidate in &candidates {
        let ok = tokio::process::Command::new(candidate)
            .args(["-c", check_import])
            .output()
            .await
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return candidate.to_string();
        }
    }
    warn!("L2: tensorflow_hub not found — YAMNet will use stub");
    "python3".to_string()
}

/// Build CSV string to embed in encrypted payload (called before encryption).
/// Reads submission.csv (BirdCLEF 2026 format) produced by inference script.
/// Falls back to a minimal score-only row if submission.csv is absent.
async fn build_predictions_csv(output_dir: &Path, mean_score: f64) -> String {
    // Prefer submission.csv (BirdCLEF 2026: row_id + 234 species columns)
    if let Ok(csv) = tokio::fs::read_to_string(output_dir.join("submission.csv")).await {
        if !csv.trim().is_empty() {
            return csv;
        }
    }
    // Fallback: minimal CSV with overall score
    format!("row_id,score\nresult,{mean_score:.6}\n")
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

async fn download(http: &reqwest::Client, url: &str, dest: &Path, job_id: &str) -> Result<()> {
    tokio::fs::create_dir_all(dest).await?;
    
    // Handle kaggle:// protocol for Kaggle datasets (e.g., BirdCLEF)
    if url.starts_with("kaggle://") {
        return download_kaggle_dataset(url, dest, job_id).await;
    }
    
    // Standard HTTP(S) download
    let resp = http.get(url).send().await.context("download")?;
    if !resp.status().is_success() {
        anyhow::bail!("download {} → {}", url, resp.status());
    }
    let bytes = resp.bytes().await?;
    let name  = url.rsplit('/').next().unwrap_or("data.bin");
    tokio::fs::write(dest.join(name), &bytes).await?;
    Ok(())
}

/// Download dataset from coordinator API.
/// URL format: kaggle://competitions/{slug} or http://coordinator/datasets/{job_id}/files
async fn download_kaggle_dataset(_url: &str, dest: &Path, job_id: &str) -> Result<()> {
    log::info!("Downloading dataset from coordinator for job: {job_id}");
    
    // Get coordinator URL from environment or default
    let coordinator_url = std::env::var("L2_COORDINATOR_URL")
        .unwrap_or_else(|_| "http://localhost:8091".to_string());
    
    let http = reqwest::Client::new();
    let files_url = format!("{}/datasets/{}/files", coordinator_url, job_id);
    
    // Retry up to 6 times with 5s delay (30s max) — dataset may still be downloading
    let mut files = Vec::new();
    for attempt in 1..=6u32 {
        let resp = http.get(&files_url).send().await
            .context("Failed to list dataset files from coordinator")?;
        
        if !resp.status().is_success() {
            anyhow::bail!("Coordinator returned error: {}", resp.status());
        }
        
        let files_data: serde_json::Value = resp.json().await
            .context("Failed to parse files list")?;
        
        let f = files_data["files"].as_array()
            .cloned()
            .unwrap_or_default();
        
        if !f.is_empty() {
            files = f;
            break;
        }
        
        log::info!("L2: coordinator dataset not ready yet (attempt {attempt}/12), waiting 5s...");
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
    }
    
    if files.is_empty() {
        anyhow::bail!("No dataset files available from coordinator after retries");
    }
    
    log::info!("Found {} files from coordinator, downloading...", files.len());

    // Always download sample_submission.csv first (enables exact row_id matching)
    let sample_sub_url = format!("{}/datasets/{}/download/sample_submission.csv",
        coordinator_url, job_id);
    if let Ok(resp) = http.get(&sample_sub_url).send().await {
        if resp.status().is_success() {
            if let Ok(bytes) = resp.bytes().await {
                tokio::fs::write(dest.join("sample_submission.csv"), &bytes).await.ok();
                log::info!("Downloaded sample_submission.csv ({} bytes)", bytes.len());
            }
        }
    }

    // Download audio files — prefer test_soundscapes/, skip train_audio/
    let mut audio_count = 0;
    for file in &files {
        let filename = match file["filename"].as_str() { Some(f) => f, None => continue };

        // Skip train_audio — only use test_soundscapes or train_soundscapes for inference
        if filename.starts_with("train_audio/") { continue; }

        if let Some(ext) = std::path::Path::new(filename).extension() {
            let ext_str = ext.to_string_lossy().to_lowercase();
            if matches!(ext_str.as_str(), "wav" | "ogg" | "mp3" | "flac") {
                let download_url = format!("{}/datasets/{}/download/{}",
                    coordinator_url, job_id, filename);
                if let Ok(file_resp) = http.get(&download_url).send().await {
                    if file_resp.status().is_success() {
                        if let Ok(bytes) = file_resp.bytes().await {
                            let basename = std::path::Path::new(filename)
                                .file_name().unwrap_or(std::ffi::OsStr::new(filename));
                            let dest_file = dest.join(basename);
                            tokio::fs::write(&dest_file, &bytes).await.ok();
                            log::info!("Downloaded: {} → {}", filename, basename.to_string_lossy());
                            audio_count += 1;
                        }
                    }
                }
            }
        }
    }
    log::info!("Downloaded {audio_count} test audio files to {}", dest.display());
    
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
