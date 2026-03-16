use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use clap::{Arg, Command};
use genetics_l2_core::{now_secs, JobResult, Payout, ScientificJob, ValidationReport};
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

// ── DB schema ─────────────────────────────────────────────────────────────────

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS jobs (
    job_id          TEXT    PRIMARY KEY,
    source          TEXT    NOT NULL,
    external_ref    TEXT,
    dataset_root    TEXT    NOT NULL,
    dataset_url     TEXT,
    algorithm       TEXT    NOT NULL,
    task_description TEXT   NOT NULL,
    reward_sompi    INTEGER NOT NULL DEFAULT 0,
    max_time_secs   INTEGER NOT NULL DEFAULT 3600,
    status          TEXT    NOT NULL DEFAULT 'open',
    claimed_by      TEXT,
    created_at      INTEGER NOT NULL,
    claimed_at      INTEGER,
    completed_at    INTEGER
);

CREATE TABLE IF NOT EXISTS results (
    result_id              TEXT    PRIMARY KEY,
    job_id                 TEXT    NOT NULL REFERENCES jobs(job_id),
    worker_pubkey          TEXT    NOT NULL,
    result_root            TEXT    NOT NULL,
    score                  REAL    NOT NULL,
    trace_hash             TEXT,
    notebook_or_repo_hash  TEXT,
    container_hash         TEXT,
    weights_hash           TEXT,
    submission_bundle_hash TEXT,
    worker_sig             TEXT    NOT NULL,
    encrypted_payload      TEXT,
    ephemeral_pubkey       TEXT,
    submitted_at           INTEGER NOT NULL,
    verdict                TEXT
);

CREATE TABLE IF NOT EXISTS validation_reports (
    report_id       TEXT    PRIMARY KEY,
    job_id          TEXT    NOT NULL,
    result_id       TEXT    NOT NULL,
    validator_pubkey TEXT   NOT NULL,
    verdict         TEXT    NOT NULL,
    recomputed_score REAL,
    score_delta     REAL,
    notes           TEXT,
    validator_sig   TEXT    NOT NULL,
    validated_at    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS payouts (
    payout_id       TEXT    PRIMARY KEY,
    job_id          TEXT    NOT NULL,
    worker_pubkey   TEXT    NOT NULL,
    amount_sompi    INTEGER NOT NULL,
    txid            TEXT,
    paid_at         INTEGER
);
"#;

// ── App state ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    pool: SqlitePool,
    /// Coordinator's secp256k1 keypair for encrypting/decrypting L2 results.
    coordinator_privkey: String,
    coordinator_pubkey: String,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

// GET /health
async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "service": "genetics-l2-coordinator" }))
}

// GET /pubkey - Returns coordinator's public key for result encryption
async fn get_pubkey(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    Json(serde_json::json!({ "pubkey": s.coordinator_pubkey }))
}

// GET /jobs?status=open&limit=20&offset=0
#[derive(Deserialize)]
struct JobsQuery {
    status: Option<String>,
    limit:  Option<i64>,
    offset: Option<i64>,
}

async fn list_jobs(
    State(s): State<Arc<AppState>>,
    Query(q): Query<JobsQuery>,
) -> impl IntoResponse {
    let status = q.status.as_deref().unwrap_or("open");
    let limit  = q.limit.unwrap_or(50);
    let offset = q.offset.unwrap_or(0);

    let rows = sqlx::query(
        "SELECT job_id, source, external_ref, dataset_root, dataset_url,
                algorithm, task_description, reward_sompi, max_time_secs,
                status, claimed_by, created_at, claimed_at, completed_at
         FROM jobs WHERE status = ?1 ORDER BY created_at ASC LIMIT ?2 OFFSET ?3",
    )
    .bind(status)
    .bind(limit)
    .bind(offset)
    .fetch_all(&s.pool)
    .await;

    match rows {
        Ok(rows) => {
            let jobs: Vec<serde_json::Value> = rows.iter().map(|r| {
                use sqlx::Row;
                serde_json::json!({
                    "job_id":           r.get::<String, _>("job_id"),
                    "source":           r.get::<String, _>("source"),
                    "external_ref":     r.get::<Option<String>, _>("external_ref"),
                    "dataset_root":     r.get::<String, _>("dataset_root"),
                    "dataset_url":      r.get::<Option<String>, _>("dataset_url"),
                    "algorithm":        r.get::<String, _>("algorithm"),
                    "task_description": r.get::<String, _>("task_description"),
                    "reward_sompi":     r.get::<i64, _>("reward_sompi"),
                    "max_time_secs":    r.get::<i64, _>("max_time_secs"),
                    "status":           r.get::<String, _>("status"),
                    "claimed_by":       r.get::<Option<String>, _>("claimed_by"),
                    "created_at":       r.get::<i64, _>("created_at"),
                })
            }).collect();
            (StatusCode::OK, Json(serde_json::json!({ "jobs": jobs }))).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        ).into_response(),
    }
}

// POST /jobs  (used by job-fetcher to register new jobs)
async fn create_job(
    State(s): State<Arc<AppState>>,
    Json(job): Json<ScientificJob>,
) -> impl IntoResponse {
    let initial_status = if job.dataset_url.as_deref()
        .map(|u| u.starts_with("kaggle://"))
        .unwrap_or(false)
    {
        "pending"   // will flip to 'open' once dataset is downloaded
    } else {
        "open"
    };

    let res = sqlx::query(
        "INSERT OR IGNORE INTO jobs
         (job_id, source, external_ref, dataset_root, dataset_url, algorithm,
          task_description, reward_sompi, max_time_secs, status, created_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
    )
    .bind(&job.job_id)
    .bind(job.source.to_string())
    .bind(&job.external_ref)
    .bind(&job.dataset_root)
    .bind(&job.dataset_url)
    .bind(job.algorithm.to_string())
    .bind(&job.task_description)
    .bind(job.reward_sompi as i64)
    .bind(job.max_time_secs as i64)
    .bind(initial_status)
    .bind(job.created_at as i64)
    .execute(&s.pool)
    .await;

    match res {
        Ok(_) => {
            // If job needs a Kaggle dataset, start downloading and flip status open when ready
            if let Some(ref dataset_url) = job.dataset_url {
                if dataset_url.starts_with("kaggle://competitions/") {
                    let job_id = job.job_id.clone();
                    let dataset_url = dataset_url.clone();
                    let pool = s.pool.clone();
                    tokio::spawn(async move {
                        match download_kaggle_dataset(&job_id, &dataset_url).await {
                            Ok(()) => {
                                // Dataset ready — open the job for miners
                                let _ = sqlx::query(
                                    "UPDATE jobs SET status='open' WHERE job_id=?1 AND status='pending'"
                                )
                                .bind(&job_id)
                                .execute(&pool)
                                .await;
                                log::info!("Dataset ready, job {job_id} is now open");
                            }
                            Err(e) => {
                                // Download failed — open anyway so miners can still try with stub
                                log::warn!("Dataset download failed for {job_id}: {e} — opening job anyway");
                                let _ = sqlx::query(
                                    "UPDATE jobs SET status='open' WHERE job_id=?1 AND status='pending'"
                                )
                                .bind(&job_id)
                                .execute(&pool)
                                .await;
                            }
                        }
                    });
                    // Return immediately — job is pending until dataset is ready
                    return (StatusCode::CREATED, Json(serde_json::json!({ "job_id": job.job_id, "status": "pending" }))).into_response();
                }
            }
            (StatusCode::CREATED, Json(serde_json::json!({ "job_id": job.job_id, "status": "open" }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// POST /jobs/:job_id/claim  (worker claims a job)
#[derive(Deserialize)]
struct ClaimBody {
    worker_pubkey: String,
}

async fn claim_job(
    State(s): State<Arc<AppState>>,
    Path(job_id): Path<String>,
    Json(body): Json<ClaimBody>,
) -> impl IntoResponse {
    let now = now_secs() as i64;
    let res = sqlx::query(
        "UPDATE jobs SET status='claimed', claimed_by=?1, claimed_at=?2
         WHERE job_id=?3 AND status='open'",
    )
    .bind(&body.worker_pubkey)
    .bind(now)
    .bind(&job_id)
    .execute(&s.pool)
    .await;

    match res {
        Ok(r) if r.rows_affected() > 0 =>
            (StatusCode::OK, Json(serde_json::json!({ "claimed": true, "job_id": job_id }))).into_response(),
        Ok(_) =>
            (StatusCode::CONFLICT, Json(serde_json::json!({ "error": "job not available" }))).into_response(),
        Err(e) =>
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// POST /results  (worker submits result)
async fn submit_result(
    State(s): State<Arc<AppState>>,
    Json(result): Json<JobResult>,
) -> impl IntoResponse {
    let now = now_secs() as i64;
    let res = sqlx::query(
        "INSERT OR IGNORE INTO results
         (result_id, job_id, worker_pubkey, result_root, score,
          trace_hash, notebook_or_repo_hash, container_hash, weights_hash,
          submission_bundle_hash, worker_sig, encrypted_payload, ephemeral_pubkey, submitted_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
    )
    .bind(&result.result_id)
    .bind(&result.job_id)
    .bind(&result.worker_pubkey)
    .bind(&result.result_root)
    .bind(result.score)
    .bind(&result.trace_hash)
    .bind(&result.notebook_or_repo_hash)
    .bind(&result.container_hash)
    .bind(&result.weights_hash)
    .bind(&result.submission_bundle_hash)
    .bind(&result.worker_sig)
    .bind(&result.encrypted_payload)
    .bind(&result.ephemeral_pubkey)
    .bind(now)
    .execute(&s.pool)
    .await;

    if res.is_ok() {
        // Mark job as completed
        let _ = sqlx::query(
            "UPDATE jobs SET status='completed', completed_at=?1 WHERE job_id=?2 AND status='claimed'",
        )
        .bind(now)
        .bind(&result.job_id)
        .execute(&s.pool)
        .await;
    }

    match res {
        Ok(_)  => (StatusCode::CREATED, Json(serde_json::json!({ "result_id": result.result_id }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// GET /results/:job_id
async fn get_results(
    State(s): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        "SELECT result_id, worker_pubkey, result_root, score, submitted_at, verdict,
                notebook_or_repo_hash, container_hash, weights_hash, submission_bundle_hash,
                encrypted_payload, ephemeral_pubkey
         FROM results WHERE job_id = ?1 ORDER BY score DESC",
    )
    .bind(&job_id)
    .fetch_all(&s.pool)
    .await;

    match rows {
        Ok(rows) => {
            use sqlx::Row;
            let results: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::json!({
                "result_id":              r.get::<String, _>("result_id"),
                "worker_pubkey":          r.get::<String, _>("worker_pubkey"),
                "result_root":            r.get::<String, _>("result_root"),
                "score":                  r.get::<f64, _>("score"),
                "submitted_at":           r.get::<i64, _>("submitted_at"),
                "verdict":                r.get::<Option<String>, _>("verdict"),
                "notebook_or_repo_hash":  r.get::<Option<String>, _>("notebook_or_repo_hash"),
                "container_hash":         r.get::<Option<String>, _>("container_hash"),
                "weights_hash":           r.get::<Option<String>, _>("weights_hash"),
                "submission_bundle_hash": r.get::<Option<String>, _>("submission_bundle_hash"),
                "encrypted_payload":      r.get::<Option<String>, _>("encrypted_payload"),
                "ephemeral_pubkey":       r.get::<Option<String>, _>("ephemeral_pubkey"),
            })).collect();
            (StatusCode::OK, Json(serde_json::json!({ "job_id": job_id, "results": results }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// GET /datasets/:job_id/files - List available dataset files for a job
async fn list_dataset_files(
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    use std::path::Path;
    
    let dataset_dir = Path::new("/tmp/kaggle-datasets").join(&job_id);
    
    if !dataset_dir.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Dataset not found for job",
            "job_id": job_id
        }))).into_response();
    }
    
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&dataset_dir) {
        for entry in entries.flatten() {
            if let Ok(metadata) = entry.metadata() {
                if metadata.is_file() {
                    if let Some(filename) = entry.file_name().to_str() {
                        files.push(serde_json::json!({
                            "filename": filename,
                            "size": metadata.len(),
                        }));
                    }
                }
            }
        }
    }
    
    (StatusCode::OK, Json(serde_json::json!({
        "job_id": job_id,
        "files": files,
        "count": files.len()
    }))).into_response()
}

// GET /datasets/:job_id/download/:filename - Download a specific dataset file
async fn download_dataset_file(
    Path((job_id, filename)): Path<(String, String)>,
) -> impl IntoResponse {
    use std::path::Path;
    use axum::body::Body;
    use axum::http::header;
    
    let dataset_dir = Path::new("/tmp/kaggle-datasets").join(&job_id);
    let file_path = dataset_dir.join(&filename);
    
    // Security: prevent path traversal
    if !file_path.starts_with(&dataset_dir) {
        return (StatusCode::FORBIDDEN, Json(serde_json::json!({
            "error": "Invalid filename"
        }))).into_response();
    }
    
    if !file_path.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "File not found",
            "filename": filename
        }))).into_response();
    }
    
    match tokio::fs::File::open(&file_path).await {
        Ok(file) => {
            let stream = tokio_util::io::ReaderStream::new(file);
            let body = Body::from_stream(stream);
            
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/octet-stream")],
                body
            ).into_response()
        }
        Err(e) => {
            (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({
                "error": format!("Failed to read file: {e}")
            }))).into_response()
        }
    }
}

// POST /validations  (validator posts a validation report)
async fn submit_validation(
    State(s): State<Arc<AppState>>,
    Json(report): Json<ValidationReport>,
) -> impl IntoResponse {
    let verdict_str = format!("{:?}", report.verdict).to_lowercase();
    let res = sqlx::query(
        "INSERT OR IGNORE INTO validation_reports
         (report_id, job_id, result_id, validator_pubkey, verdict,
          recomputed_score, score_delta, notes, validator_sig, validated_at)
         VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10)",
    )
    .bind(&report.report_id)
    .bind(&report.job_id)
    .bind(&report.result_id)
    .bind(&report.validator_pubkey)
    .bind(&verdict_str)
    .bind(report.recomputed_score)
    .bind(report.score_delta)
    .bind(&report.notes)
    .bind(&report.validator_sig)
    .bind(report.validated_at as i64)
    .execute(&s.pool)
    .await;

    // If valid verdict, update result and mark job as validated
    if matches!(report.verdict, genetics_l2_core::ValidationVerdict::Valid) {
        let _ = sqlx::query("UPDATE results SET verdict='valid' WHERE result_id=?1")
            .bind(&report.result_id)
            .execute(&s.pool)
            .await;
        let _ = sqlx::query("UPDATE jobs SET status='validated' WHERE job_id=?1 AND status='completed'")
            .bind(&report.job_id)
            .execute(&s.pool)
            .await;
    }

    match res {
        Ok(_)  => (StatusCode::CREATED, Json(serde_json::json!({ "report_id": report.report_id }))).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// GET /payouts?worker=<pubkey>
#[derive(Deserialize)]
struct PayoutsQuery {
    worker: Option<String>,
}

async fn list_payouts(
    State(s): State<Arc<AppState>>,
    Query(q): Query<PayoutsQuery>,
) -> impl IntoResponse {
    let rows = if let Some(w) = &q.worker {
        sqlx::query(
            "SELECT payout_id, job_id, worker_pubkey, amount_sompi, txid, paid_at
             FROM payouts WHERE worker_pubkey = ?1 ORDER BY paid_at DESC LIMIT 100",
        )
        .bind(w)
        .fetch_all(&s.pool)
        .await
    } else {
        sqlx::query(
            "SELECT payout_id, job_id, worker_pubkey, amount_sompi, txid, paid_at
             FROM payouts ORDER BY paid_at DESC LIMIT 100",
        )
        .fetch_all(&s.pool)
        .await
    };

    match rows {
        Ok(rows) => {
            use sqlx::Row;
            let payouts: Vec<serde_json::Value> = rows.iter().map(|r| serde_json::json!({
                "payout_id":    r.get::<String, _>("payout_id"),
                "job_id":       r.get::<String, _>("job_id"),
                "worker_pubkey":r.get::<String, _>("worker_pubkey"),
                "amount_sompi": r.get::<i64, _>("amount_sompi"),
                "txid":         r.get::<Option<String>, _>("txid"),
                "paid_at":      r.get::<Option<i64>, _>("paid_at"),
            })).collect();
            (StatusCode::OK, Json(serde_json::json!({ "payouts": payouts }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// POST /payouts  (settlement service registers a payout)
async fn create_payout(
    State(s): State<Arc<AppState>>,
    Json(payout): Json<Payout>,
) -> impl IntoResponse {
    let res = sqlx::query(
        "INSERT OR IGNORE INTO payouts
         (payout_id, job_id, worker_pubkey, amount_sompi, txid, paid_at)
         VALUES (?1,?2,?3,?4,?5,?6)",
    )
    .bind(&payout.payout_id)
    .bind(&payout.job_id)
    .bind(&payout.worker_pubkey)
    .bind(payout.amount_sompi as i64)
    .bind(&payout.txid)
    .bind(payout.paid_at.map(|t| t as i64))
    .execute(&s.pool)
    .await;

    match res {
        Ok(_) => {
            let _ = sqlx::query("UPDATE jobs SET status='settled' WHERE job_id=?1")
                .bind(&payout.job_id)
                .execute(&s.pool)
                .await;
            (StatusCode::CREATED, Json(serde_json::json!({ "payout_id": payout.payout_id }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// GET /stats
#[derive(Serialize)]
struct Stats {
    total_jobs:      i64,
    open_jobs:       i64,
    completed_jobs:  i64,
    validated_jobs:  i64,
    settled_jobs:    i64,
    total_results:   i64,
    total_payouts:   i64,
}

async fn stats(State(s): State<Arc<AppState>>) -> impl IntoResponse {
    let count = |status: &str, pool: &SqlitePool| {
        let status = status.to_owned();
        let pool   = pool.clone();
        async move {
            sqlx::query_scalar::<_, i64>(
                "SELECT COUNT(*) FROM jobs WHERE status = ?1",
            )
            .bind(status)
            .fetch_one(&pool)
            .await
            .unwrap_or(0)
        }
    };

    let total_jobs     = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM jobs").fetch_one(&s.pool).await.unwrap_or(0);
    let open_jobs      = count("open",      &s.pool).await;
    let completed_jobs = count("completed", &s.pool).await;
    let validated_jobs = count("validated", &s.pool).await;
    let settled_jobs   = count("settled",   &s.pool).await;
    let total_results  = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM results").fetch_one(&s.pool).await.unwrap_or(0);
    let total_payouts  = sqlx::query_scalar::<_, i64>("SELECT COUNT(*) FROM payouts").fetch_one(&s.pool).await.unwrap_or(0);

    Json(Stats { total_jobs, open_jobs, completed_jobs, validated_jobs, settled_jobs, total_results, total_payouts })
}

// ── Router ────────────────────────────────────────────────────────────────────

fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health",                  get(health))
        .route("/pubkey",                  get(get_pubkey))
        .route("/stats",                   get(stats))
        .route("/jobs",                    get(list_jobs).post(create_job))
        .route("/jobs/:job_id/claim",      post(claim_job))
        .route("/results",                 post(submit_result))
        .route("/results/:job_id",         get(get_results))
        .route("/validations",             post(submit_validation))
        .route("/payouts",                 get(list_payouts).post(create_payout))
        .route("/datasets/:job_id/files",  get(list_dataset_files))
        .route("/datasets/:job_id/download/:filename", get(download_dataset_file))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ── Dataset download ──────────────────────────────────────────────────────────

/// Download Kaggle dataset once per slug (cached), then symlink to job_id dir.
/// Cache: /tmp/kaggle-datasets/_cache/{slug}/
/// Job dir: /tmp/kaggle-datasets/{job_id} -> symlink to cache
async fn download_kaggle_dataset(job_id: &str, dataset_url: &str) -> Result<()> {
    use std::path::Path;

    let slug = dataset_url.strip_prefix("kaggle://competitions/")
        .ok_or_else(|| anyhow::anyhow!("Invalid kaggle:// URL: {dataset_url}"))?;

    let cache_dir = Path::new("/tmp/kaggle-datasets/_cache").join(slug);
    let job_link  = Path::new("/tmp/kaggle-datasets").join(job_id);

    // If job symlink already exists, nothing to do
    if job_link.exists() {
        log::info!("Dataset already linked for job {job_id}");
        return Ok(());
    }

    // If cache already marked ready, just symlink and return immediately
    let ready_marker = cache_dir.join(".ready");
    if ready_marker.exists() {
        std::os::unix::fs::symlink(&cache_dir, &job_link).ok();
        log::info!("Dataset cache hit for {job_id} → {}", cache_dir.display());
        return Ok(());
    }

    // Lock file to prevent concurrent downloads of the same slug
    let lock_file = Path::new("/tmp/kaggle-datasets").join(format!("_{slug}.lock"));
    if lock_file.exists() {
        // Another task is already downloading — wait for it to finish (up to 30 min)
        log::info!("Another download in progress for {slug}, waiting...");
        for _ in 0..360 {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            if ready_marker.exists() {
                std::os::unix::fs::symlink(&cache_dir, &job_link).ok();
                log::info!("Dataset cache ready for {job_id} after wait");
                return Ok(());
            }
        }
        anyhow::bail!("Timed out waiting for dataset download for {slug}");
    }

    // Acquire lock
    tokio::fs::create_dir_all(Path::new("/tmp/kaggle-datasets")).await?;
    tokio::fs::write(&lock_file, job_id.as_bytes()).await?;

    // First time: download to cache
    log::info!("Downloading Kaggle dataset for job {job_id}: {slug} (will cache)");
    tokio::fs::create_dir_all(&cache_dir).await?;

    let output = tokio::process::Command::new("kaggle")
        .args(&["competitions", "download", "-c", slug, "-p", cache_dir.to_str().unwrap()])
        .output()
        .await
        .context("Failed to execute kaggle CLI")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!("kaggle download failed: {stderr}\n{stdout}");
    }

    log::info!("Kaggle dataset downloaded, extracting zip files...");

    // Extract zip files in cache
    let mut entries = tokio::fs::read_dir(&cache_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("zip") {
            log::info!("Extracting {}", path.display());
            let out = tokio::process::Command::new("unzip")
                .args(&["-o", "-q", path.to_str().unwrap(), "-d", cache_dir.to_str().unwrap()])
                .output()
                .await;
            if out.is_ok() {
                tokio::fs::remove_file(&path).await.ok();
            }
        }
    }

    let audio_count = count_audio_files(&cache_dir).await;
    log::info!("Cache ready: {audio_count} audio files in {}", cache_dir.display());

    // Write .ready marker and release lock
    tokio::fs::write(&ready_marker, b"ready").await.ok();
    tokio::fs::remove_file(&lock_file).await.ok();

    // Symlink job_id → cache
    std::os::unix::fs::symlink(&cache_dir, &job_link).ok();
    log::info!("Dataset ready for job {job_id}");

    Ok(())
}

async fn has_audio_files(dir: &std::path::Path) -> bool {
    count_audio_files(dir).await > 0
}

async fn count_audio_files(dir: &std::path::Path) -> usize {
    let Ok(mut entries) = tokio::fs::read_dir(dir).await else { return 0 };
    let mut count = 0;
    while let Ok(Some(entry)) = entries.next_entry().await {
        if let Some(ext) = entry.path().extension() {
            let e = ext.to_string_lossy().to_lowercase();
            if matches!(e.as_str(), "wav" | "ogg" | "mp3" | "flac") {
                count += 1;
            }
        }
    }
    count
}

// ── Keypair management ────────────────────────────────────────────────────────

/// Load or generate coordinator's secp256k1 keypair for result encryption.
/// Keypair is stored in {db_path}.key file.
fn load_or_generate_keypair(db_path: &str) -> Result<(String, String)> {
    use secp256k1::{PublicKey, Secp256k1, SecretKey};
    
    let key_file = format!("{db_path}.key");
    let secp = Secp256k1::new();

    // Try to load existing keypair
    if let Ok(privkey_hex) = std::fs::read_to_string(&key_file) {
        let privkey_hex = privkey_hex.trim().to_string();
        if let Ok(privkey_bytes) = hex::decode(&privkey_hex) {
            if let Ok(secret_key) = SecretKey::from_slice(&privkey_bytes) {
                let public_key = PublicKey::from_secret_key(&secp, &secret_key);
                let pubkey_hex = hex::encode(public_key.serialize());
                log::info!("Loaded existing coordinator keypair from {key_file}");
                return Ok((privkey_hex, pubkey_hex));
            }
        }
    }

    // Generate new keypair
    let secret_key = SecretKey::new(&mut secp256k1::rand::thread_rng());
    let public_key = PublicKey::from_secret_key(&secp, &secret_key);
    
    let privkey_hex = hex::encode(secret_key.secret_bytes());
    let pubkey_hex = hex::encode(public_key.serialize());

    // Save private key to file
    std::fs::write(&key_file, &privkey_hex)
        .context(format!("Failed to write keypair to {key_file}"))?;
    
    log::info!("Generated new coordinator keypair, saved to {key_file}");
    log::warn!("IMPORTANT: Backup {key_file} - it's required to decrypt L2 results!");

    Ok((privkey_hex, pubkey_hex))
}

// ── Main ──────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();

    // Handle decrypt subcommand
    if let Some(decrypt_matches) = m.subcommand_matches("decrypt") {
        let db_path = decrypt_matches.get_one::<String>("db-path").unwrap();
        let result_id = decrypt_matches.get_one::<String>("result-id").unwrap();
        return decrypt_result(db_path, result_id).await;
    }

    // Normal server mode
    let db_path = m.get_one::<String>("db-path").unwrap();
    let listen  = m.get_one::<String>("listen").unwrap();

    let pool = SqlitePool::connect(&format!("sqlite:{db_path}?mode=rwc"))
        .await
        .context("open SQLite")?;
    sqlx::raw_sql(SCHEMA).execute(&pool).await.context("schema init")?;
    log::info!("Database: {db_path}");

    // Generate or load coordinator keypair for result encryption
    let (coordinator_privkey, coordinator_pubkey) = load_or_generate_keypair(&db_path)?;
    log::info!("Coordinator pubkey: {coordinator_pubkey}");

    let state  = Arc::new(AppState { 
        pool,
        coordinator_privkey,
        coordinator_pubkey,
    });
    let router = build_router(state);

    let addr: std::net::SocketAddr = listen.parse().context("invalid --listen address")?;
    log::info!("genetics-l2-coordinator listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, router).await.context("server error")?;
    Ok(())
}

// ── Decrypt subcommand ────────────────────────────────────────────────────────

async fn decrypt_result(db_path: &str, result_id: &str) -> Result<()> {
    use genetics_l2_core::JobResult;
    
    // Load coordinator private key
    let (coordinator_privkey, _) = load_or_generate_keypair(db_path)?;
    
    // Connect to database
    let pool = SqlitePool::connect(&format!("sqlite:{db_path}?mode=rwc"))
        .await
        .context("open SQLite")?;
    
    // Fetch result from database
    let row = sqlx::query(
        "SELECT result_id, job_id, worker_pubkey, result_root, score,
                trace_hash, notebook_or_repo_hash, container_hash, weights_hash,
                submission_bundle_hash, worker_sig, encrypted_payload, ephemeral_pubkey,
                submitted_at, verdict
         FROM results WHERE result_id = ?1"
    )
    .bind(result_id)
    .fetch_optional(&pool)
    .await?;
    
    let row = row.ok_or_else(|| anyhow::anyhow!("Result not found: {result_id}"))?;
    
    use sqlx::Row;
    let encrypted_payload = row.get::<Option<String>, _>("encrypted_payload");
    let ephemeral_pubkey = row.get::<Option<String>, _>("ephemeral_pubkey");
    
    if encrypted_payload.is_none() || ephemeral_pubkey.is_none() {
        println!("Result {} is not encrypted (old format)", result_id);
        println!("result_root: {}", row.get::<String, _>("result_root"));
        println!("score: {}", row.get::<f64, _>("score"));
        println!("trace_hash: {:?}", row.get::<Option<String>, _>("trace_hash"));
        return Ok(());
    }
    
    // Decrypt
    println!("Decrypting result {} with coordinator private key...", result_id);
    
    let encrypted_payload_hex = encrypted_payload.unwrap();
    let ephemeral_pubkey_hex = ephemeral_pubkey.unwrap();
    
    let decrypted = JobResult::decrypt_payload(
        &encrypted_payload_hex,
        &ephemeral_pubkey_hex,
        &coordinator_privkey
    )
    .map_err(|e| anyhow::anyhow!("Failed to decrypt result: {e}"))?;
    
    // Display decrypted data
    println!("\n=== Decrypted Result ===");
    println!("Metadata (from database):");
    println!("  result_id: {}", row.get::<String, _>("result_id"));
    println!("  job_id: {}", row.get::<String, _>("job_id"));
    println!("  worker_pubkey: {}", row.get::<String, _>("worker_pubkey"));
    println!("  submitted_at: {}", row.get::<i64, _>("submitted_at"));
    println!("  verdict: {:?}", row.get::<Option<String>, _>("verdict"));
    println!("\nDecrypted Payload:");
    println!("  result_root: {}", decrypted.result_root);
    println!("  score: {}", decrypted.score);
    println!("  trace_hash: {:?}", decrypted.trace_hash);
    println!("  notebook_or_repo_hash: {:?}", decrypted.notebook_or_repo_hash);
    println!("  container_hash: {:?}", decrypted.container_hash);
    println!("  weights_hash: {:?}", decrypted.weights_hash);
    println!("  submission_bundle_hash: {:?}", decrypted.submission_bundle_hash);
    
    Ok(())
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-coordinator")
        .about("Genetics L2 coordinator — job registry, scheduler, result aggregator")
        .subcommand_required(false)
        .subcommand(
            Command::new("decrypt")
                .about("Decrypt and view a submitted result (coordinator owner only)")
                .arg(Arg::new("db-path")
                    .short('d').long("db-path").value_name("PATH")
                    .default_value("genetics-l2.db")
                    .help("SQLite database path"))
                .arg(Arg::new("result-id")
                    .short('r').long("result-id").value_name("ID")
                    .required(true)
                    .help("Result ID to decrypt"))
        )
        .arg(Arg::new("db-path")
            .short('d').long("db-path").value_name("PATH")
            .default_value("genetics-l2.db")
            .help("SQLite database path"))
        .arg(Arg::new("listen")
            .short('l').long("listen").value_name("ADDR")
            .default_value("0.0.0.0:8091")
            .help("REST API listen address"))
}
