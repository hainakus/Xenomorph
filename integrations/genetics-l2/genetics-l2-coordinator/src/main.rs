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
    /// Directory where inference scripts are stored (served to miners via GET /scripts/:task)
    scripts_dir: std::path::PathBuf,
    /// Persistent base directory for Kaggle dataset caches.
    /// Cache layout: {datasets_dir}/_cache/{slug}/  Job symlinks: {datasets_dir}/{job_id}
    datasets_dir: std::path::PathBuf,
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
    let initial_status = "open"; // always open immediately — dataset downloads in background

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
            // Spawn background download — job is already 'open', miners get it immediately
            if let Some(ref dataset_url) = job.dataset_url {
                if dataset_url.starts_with("kaggle://competitions/") {
                    let job_id = job.job_id.clone();
                    let dataset_url = dataset_url.clone();
                    let datasets_dir = s.datasets_dir.clone();
                    tokio::spawn(async move {
                        if let Err(e) = download_kaggle_dataset(&job_id, &dataset_url, &datasets_dir).await {
                            log::warn!("Background dataset download failed for {job_id}: {e}");
                        }
                    });
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
        "SELECT r.result_id, r.worker_pubkey, r.result_root, r.score, r.submitted_at, r.verdict,
                r.notebook_or_repo_hash, r.container_hash, r.weights_hash,
                r.submission_bundle_hash, r.encrypted_payload, r.ephemeral_pubkey,
                vr.recomputed_score
         FROM results r
         LEFT JOIN validation_reports vr ON vr.result_id = r.result_id AND vr.verdict = 'valid'
         WHERE r.job_id = ?1
         ORDER BY COALESCE(vr.recomputed_score, r.score) DESC",
    )
    .bind(&job_id)
    .fetch_all(&s.pool)
    .await;

    match rows {
        Ok(rows) => {
            use sqlx::Row;
            let results: Vec<serde_json::Value> = rows.iter().map(|r| {
                let submitted_score: f64 = r.get::<f64, _>("score");
                let recomputed_score: Option<f64> = r.get::<Option<f64>, _>("recomputed_score");
                serde_json::json!({
                    "result_id":              r.get::<String, _>("result_id"),
                    "worker_pubkey":          r.get::<String, _>("worker_pubkey"),
                    "result_root":            r.get::<String, _>("result_root"),
                    "score":                  recomputed_score.unwrap_or(submitted_score),
                    "submitted_score":        submitted_score,
                    "recomputed_score":       recomputed_score,
                    "submitted_at":           r.get::<i64, _>("submitted_at"),
                    "verdict":                r.get::<Option<String>, _>("verdict"),
                    "notebook_or_repo_hash":  r.get::<Option<String>, _>("notebook_or_repo_hash"),
                    "container_hash":         r.get::<Option<String>, _>("container_hash"),
                    "weights_hash":           r.get::<Option<String>, _>("weights_hash"),
                    "submission_bundle_hash": r.get::<Option<String>, _>("submission_bundle_hash"),
                    "encrypted_payload":      r.get::<Option<String>, _>("encrypted_payload"),
                    "ephemeral_pubkey":       r.get::<Option<String>, _>("ephemeral_pubkey"),
                })
            }).collect();
            (StatusCode::OK, Json(serde_json::json!({ "job_id": job_id, "results": results }))).into_response()
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))).into_response(),
    }
}

// GET /results/:job_id/csv — decrypt best valid result, return predictions_csv as text/csv
async fn get_result_csv(
    State(s): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    use sqlx::Row;

    // Fetch best valid result (encrypted_payload + ephemeral_pubkey)
    let row = sqlx::query(
        "SELECT encrypted_payload, ephemeral_pubkey FROM results
         WHERE job_id = ?1 AND verdict = 'valid'
         ORDER BY score DESC LIMIT 1",
    )
    .bind(&job_id)
    .fetch_optional(&s.pool)
    .await;

    let row = match row {
        Ok(Some(r)) => r,
        Ok(None) => return (StatusCode::NOT_FOUND,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            "{\"error\":\"no valid result for job\"}".to_owned()).into_response(),
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            format!("{{\"error\":\"{e}\"}}")).into_response(),
    };

    let encrypted: Option<String> = row.get("encrypted_payload");
    let ephemeral: Option<String> = row.get("ephemeral_pubkey");

    let (Some(enc), Some(eph)) = (encrypted, ephemeral) else {
        return (StatusCode::NOT_FOUND,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            "{\"error\":\"no encrypted payload\"}".to_owned()).into_response();
    };

    match genetics_l2_core::JobResult::decrypt_payload(&enc, &eph, &s.coordinator_privkey) {
        Ok(payload) => {
            let csv = payload.predictions_csv
                .unwrap_or_else(|| format!("job_id,score\n{job_id},{}\n", payload.score));
            (
                StatusCode::OK,
                [(axum::http::header::CONTENT_TYPE, "text/csv; charset=utf-8")],
                csv,
            ).into_response()
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            format!("{{\"error\":\"decrypt failed: {e}\"}}"),
        ).into_response(),
    }
}

// GET /datasets/:job_id/files - List available dataset files for a job (recursive)
async fn list_dataset_files(
    State(s): State<Arc<AppState>>,
    Path(job_id): Path<String>,
) -> impl IntoResponse {
    let dataset_dir = s.datasets_dir.join(&job_id);

    if !dataset_dir.exists() {
        return (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": "Dataset not found for job",
            "job_id": job_id
        }))).into_response();
    }
    
    let mut files: Vec<serde_json::Value> = Vec::new();

    // Always include sample_submission.csv so miners can produce exact row_ids
    let sample_sub = dataset_dir.join("sample_submission.csv");
    if sample_sub.exists() {
        let size = sample_sub.metadata().map(|m| m.len()).unwrap_or(0);
        files.push(serde_json::json!({ "filename": "sample_submission.csv", "size": size }));
    }

    // Walk audio: prioritise test_soundscapes → train_soundscapes → train_audio
    let priority_dirs = ["test_soundscapes", "train_soundscapes", "train_audio"];
    'outer: for dir_name in &priority_dirs {
        let subdir = dataset_dir.join(dir_name);
        if !subdir.exists() { continue; }
        let mut stack = vec![subdir];
        while let Some(dir) = stack.pop() {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        stack.push(path);
                    } else if let Some(ext) = path.extension() {
                        let e = ext.to_string_lossy().to_lowercase();
                        if matches!(e.as_str(), "ogg" | "wav" | "mp3" | "flac") {
                            if let Ok(rel) = path.strip_prefix(&dataset_dir) {
                                let size = path.metadata().map(|m| m.len()).unwrap_or(0);
                                files.push(serde_json::json!({
                                    "filename": rel.to_string_lossy(),
                                    "size": size,
                                }));
                                if files.len() >= 101 { break 'outer; }
                            }
                        }
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

// GET /datasets/:job_id/download/*filename - Download a specific dataset file (supports nested paths)
async fn download_dataset_file(
    State(s): State<Arc<AppState>>,
    Path((job_id, filename)): Path<(String, String)>,
) -> impl IntoResponse {
    use axum::body::Body;
    use axum::http::header;
    
    let dataset_dir = s.datasets_dir.join(&job_id);
    // filename may contain slashes for nested paths (e.g. train_audio/Abrupto/file.ogg)
    let file_path = dataset_dir.join(&filename);
    
    // Security: prevent path traversal
    let canonical_dir  = dataset_dir.canonicalize().unwrap_or(dataset_dir.clone());
    let canonical_file = file_path.canonicalize().unwrap_or(file_path.clone());
    if !canonical_file.starts_with(&canonical_dir) {
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

/// Decrypt the winning result's encrypted_payload and save predictions_csv to disk.
async fn save_predictions_csv_on_validation(s: &AppState, job_id: &str, result_id: &str) {
    use sqlx::Row;
    let row = sqlx::query(
        "SELECT encrypted_payload, ephemeral_pubkey FROM results WHERE result_id = ?1",
    )
    .bind(result_id)
    .fetch_optional(&s.pool)
    .await;

    let row = match row { Ok(Some(r)) => r, _ => return };
    let enc: Option<String> = row.get("encrypted_payload");
    let eph: Option<String> = row.get("ephemeral_pubkey");
    let (Some(enc), Some(eph)) = (enc, eph) else { return };

    let payload = match genetics_l2_core::JobResult::decrypt_payload(&enc, &eph, &s.coordinator_privkey) {
        Ok(p) => p,
        Err(e) => { log::warn!("CSV save: decrypt failed for {result_id}: {e}"); return }
    };

    let csv = payload.predictions_csv
        .unwrap_or_else(|| format!("job_id,score\n{job_id},{}\n", payload.score));

    let dir = s.datasets_dir.parent()
        .unwrap_or(&s.datasets_dir)
        .join("kaggle-submissions");
    if tokio::fs::create_dir_all(&dir).await.is_err() { return }
    let path = dir.join(format!("{job_id}.csv"));
    match tokio::fs::write(&path, csv.as_bytes()).await {
        Ok(_)  => log::info!("Saved predictions CSV: {}", path.display()),
        Err(e) => log::warn!("Failed to save predictions CSV: {e}"),
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

        // Decrypt and save predictions CSV to disk
        save_predictions_csv_on_validation(&s, &report.job_id, &report.result_id).await;
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
        .route("/results/:job_id/csv",     get(get_result_csv))
        .route("/validations",             post(submit_validation))
        .route("/payouts",                 get(list_payouts).post(create_payout))
        .route("/datasets/:job_id/files",  get(list_dataset_files))
        .route("/datasets/:job_id/download/*filename", get(download_dataset_file))
        .route("/scripts/:task",            get(get_inference_script))
        .route("/scripts/:task/requirements", get(get_script_requirements))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

// ── Inference script serving ─────────────────────────────────────────────────

/// GET /scripts/:task?backend=yamnet|efficientnet
/// Returns the Python inference script for the given task.
///
/// Task → default script:
///   acoustic_classification / birdclef → yamnet_infer.py  (TensorFlow, Linux/GPU)
///
/// Optional ?backend= query param:
///   ?backend=efficientnet  → efficientnet_infer.py  (PyTorch, macOS/MPS)
///   ?backend=yamnet        → yamnet_infer.py         (TensorFlow Hub)
#[derive(Deserialize)]
struct ScriptQuery {
    backend: Option<String>,
}

async fn get_inference_script(
    State(s): State<Arc<AppState>>,
    Path(task): Path<String>,
    Query(q): Query<ScriptQuery>,
) -> impl IntoResponse {
    let script_name = match q.backend.as_deref() {
        Some("gpu")          => "birdclef_gpu_infer.py",
        Some("efficientnet") => "efficientnet_infer.py",
        Some("yamnet")       => "yamnet_infer.py",
        Some("genome")       => "genome_annotate.py",
        _ => match task.as_str() {
            "acoustic_classification" | "birdclef" => "yamnet_infer.py",
            "variant_calling" | "cancer_genomics" | "genome_assembly"
            | "metagenomics"  | "annotation"        => "genome_annotate.py",
            other => return serve_script_file(&s.scripts_dir, &format!("{other}.py")).await,
        },
    };
    serve_script_file(&s.scripts_dir, script_name).await
}

/// GET /scripts/:task/requirements?backend=yamnet|efficientnet
/// Returns the pip requirements.txt for the given task's inference backend.
async fn get_script_requirements(
    State(s): State<Arc<AppState>>,
    Path(task): Path<String>,
    Query(q): Query<ScriptQuery>,
) -> impl IntoResponse {
    let req_name = match q.backend.as_deref() {
        Some("gpu")          => "requirements-birdclef_gpu.txt",
        Some("efficientnet") => "requirements-efficientnet.txt",
        Some("yamnet")       => "requirements-yamnet.txt",
        Some("genome")       => "requirements-genome.txt",
        _ => match task.as_str() {
            "acoustic_classification" | "birdclef" => "requirements-yamnet.txt",
            "variant_calling" | "cancer_genomics" | "genome_assembly"
            | "metagenomics"  | "annotation"        => "requirements-genome.txt",
            other => return serve_script_file(&s.scripts_dir, &format!("requirements-{other}.txt")).await,
        },
    };
    serve_script_file(&s.scripts_dir, req_name).await
}

async fn serve_script_file(scripts_dir: &std::path::Path, filename: &str) -> axum::response::Response {
    use axum::http::header;
    let path = scripts_dir.join(filename);
    match tokio::fs::read_to_string(&path).await {
        Ok(content) => (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "text/x-python")],
            content,
        ).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, Json(serde_json::json!({
            "error": format!("Script not found: {filename}"),
            "scripts_dir": scripts_dir.display().to_string(),
        }))).into_response(),
    }
}

// ── Dataset download ──────────────────────────────────────────────────────────

/// Download Kaggle dataset once per slug (cached), then symlink to job_id dir.
/// Cache: {datasets_dir}/_cache/{slug}/
/// Job dir: {datasets_dir}/{job_id} -> symlink to cache
async fn download_kaggle_dataset(
    job_id: &str,
    dataset_url: &str,
    datasets_dir: &std::path::Path,
) -> Result<()> {
    let slug = dataset_url.strip_prefix("kaggle://competitions/")
        .ok_or_else(|| anyhow::anyhow!("Invalid kaggle:// URL: {dataset_url}"))?;

    let cache_dir = datasets_dir.join("_cache").join(slug);
    let job_link  = datasets_dir.join(job_id);

    // If job symlink already exists, nothing to do
    if job_link.exists() {
        log::info!("Dataset already linked for job {job_id}");
        return Ok(());
    }

    // Always symlink first — serve whatever is in cache (partial is fine)
    tokio::fs::create_dir_all(&cache_dir).await?;
    std::os::unix::fs::symlink(&cache_dir, &job_link).ok();

    // If cache already has audio files, job can start working immediately
    let audio_now = count_audio_files_recursive(&cache_dir).await;
    if audio_now > 0 {
        log::info!("Partial cache hit for {job_id}: {audio_now} audio files already in cache");
    }

    // Lock file to prevent concurrent full downloads of the same slug
    let lock_file = datasets_dir.join(format!("_{slug}.lock"));
    if lock_file.exists() {
        log::info!("Download already in progress for {slug}, job {job_id} will use partial cache");
        return Ok(());
    }

    let ready_marker = cache_dir.join(".ready");
    if ready_marker.exists() {
        log::info!("Dataset fully cached for {job_id}");
        return Ok(());
    }

    // Acquire lock and download missing files
    tokio::fs::create_dir_all(datasets_dir).await?;
    tokio::fs::write(&lock_file, job_id.as_bytes()).await?;

    log::info!("Downloading Kaggle dataset for job {job_id}: {slug} (--skip-existing)");

    let output = tokio::process::Command::new("kaggle")
        .args(&["competitions", "download", "-c", slug,
                "--path", cache_dir.to_str().unwrap()])
        .output()
        .await
        .context("Failed to execute kaggle CLI")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tokio::fs::remove_file(&lock_file).await.ok();
        anyhow::bail!("kaggle download failed: {stderr}");
    }

    log::info!("Extracting zip files...");

    let mut entries = tokio::fs::read_dir(&cache_dir).await?;
    while let Some(entry) = entries.next_entry().await? {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("zip") {
            log::info!("Extracting {}", path.display());
            let out = tokio::process::Command::new("unzip")
                .args(&["-n", "-q", path.to_str().unwrap(), "-d", cache_dir.to_str().unwrap()])
                .output().await;
            if out.is_ok() {
                tokio::fs::remove_file(&path).await.ok();
            }
        }
    }

    let audio_count = count_audio_files_recursive(&cache_dir).await;
    log::info!("Cache complete: {audio_count} audio files in {}", cache_dir.display());
    tokio::fs::write(&ready_marker, b"ready").await.ok();
    tokio::fs::remove_file(&lock_file).await.ok();
    log::info!("Dataset fully ready for {job_id}");
    Ok(())
}

/// Count audio files recursively (BirdCLEF stores in subdirectories)
fn count_audio_files_recursive_sync(dir: &std::path::Path) -> usize {
    let mut count = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(cur) = stack.pop() {
        if let Ok(entries) = std::fs::read_dir(&cur) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_dir() { stack.push(p); }
                else if let Some(ext) = p.extension() {
                    let e = ext.to_string_lossy().to_lowercase();
                    if matches!(e.as_str(), "wav" | "ogg" | "mp3" | "flac") { count += 1; }
                }
            }
        }
    }
    count
}

async fn count_audio_files_recursive(dir: &std::path::Path) -> usize {
    let dir = dir.to_path_buf();
    tokio::task::spawn_blocking(move || count_audio_files_recursive_sync(&dir))
        .await
        .unwrap_or(0)
}

async fn count_audio_files(dir: &std::path::Path) -> usize {
    count_audio_files_recursive(dir).await
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

    // Find scripts directory: --scripts-dir flag, next to binary, or ./scripts/
    let scripts_dir = m.get_one::<String>("scripts-dir")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::current_exe().ok()
            .and_then(|p| p.parent().map(|d| d.join("scripts"))))
        .unwrap_or_else(|| std::path::PathBuf::from("scripts"));
    log::info!("Scripts dir: {}", scripts_dir.display());

    // Persistent datasets directory: --datasets-dir > env XENOM_DATASETS_DIR > $HOME/.local/share/xenom/kaggle-datasets
    let datasets_dir = m.get_one::<String>("datasets-dir")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var("XENOM_DATASETS_DIR").ok().map(std::path::PathBuf::from))
        .unwrap_or_else(|| {
            dirs_next::home_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/var/lib/xenom"))
                .join(".local/share/xenom/kaggle-datasets")
        });
    tokio::fs::create_dir_all(&datasets_dir).await
        .context("create datasets_dir")?;
    log::info!("Datasets dir: {}", datasets_dir.display());

    let state  = Arc::new(AppState { 
        pool,
        coordinator_privkey,
        coordinator_pubkey,
        scripts_dir,
        datasets_dir,
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
        .arg(Arg::new("scripts-dir")
            .long("scripts-dir").value_name("PATH")
            .help("Directory containing Python inference scripts served to miners (default: ./scripts/ next to binary)"))
        .arg(Arg::new("datasets-dir")
            .long("datasets-dir").value_name("PATH")
            .help("Persistent directory for Kaggle dataset caches (default: $XENOM_DATASETS_DIR or ~/.local/share/xenom/kaggle-datasets)"))
}
