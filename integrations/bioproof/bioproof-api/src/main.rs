use anyhow::{Context, Result};
use axum::{
    extract::{Path, Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use bioproof_core::{verify_manifest_sig, Certificate};
use clap::{Arg, Command};
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;

// ── App state ─────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct AppState {
    pool: Arc<SqlitePool>,
}

// ── Anchor record returned by API ─────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AnchorRecord {
    txid:          String,
    block_hash:    Option<String>,
    daa_score:     Option<i64>,
    block_time:    Option<i64>,
    proof_root:    String,
    manifest_hash: String,
    kind:          String,
    indexed_at:    i64,
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m      = cli().get_matches();
    let listen = m.get_one::<String>("listen").unwrap().clone();
    let db_path = m.get_one::<String>("db-path").unwrap().clone();

    let pool = SqlitePool::connect(&format!("sqlite:{db_path}?mode=ro"))
        .await
        .with_context(|| format!("cannot open database '{db_path}'"))?;

    let state = AppState { pool: Arc::new(pool) };
    let addr: SocketAddr = listen.parse().context("invalid --listen address")?;

    let app = Router::new()
        .route("/api/anchor/:proof_root",        get(get_anchor))
        .route("/api/anchors",                   get(list_anchors))
        .route("/api/lineage/:proof_root",        get(get_lineage))
        .route("/api/verify",                    post(verify_cert))
        .route("/api/health",                    get(health))
        .layer(CorsLayer::permissive())
        .with_state(state);

    log::info!("BioProof API listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "service": "bioproof-api" }))
}

/// GET /api/anchor/:proof_root
async fn get_anchor(
    State(st): State<AppState>,
    Path(proof_root): Path<String>,
) -> impl IntoResponse {
    match fetch_anchor_by_root(&st.pool, &proof_root).await {
        Ok(Some(rec)) => (StatusCode::OK, Json(serde_json::to_value(rec).unwrap())),
        Ok(None)      => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "not found" }))),
        Err(e)        => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// GET /api/anchors?kind=vcf&limit=20&offset=0
#[derive(Deserialize)]
struct ListQuery {
    kind:   Option<String>,
    limit:  Option<i64>,
    offset: Option<i64>,
}

async fn list_anchors(
    State(st): State<AppState>,
    Query(q): Query<ListQuery>,
) -> impl IntoResponse {
    let limit  = q.limit.unwrap_or(50).min(200);
    let offset = q.offset.unwrap_or(0);

    let rows: Result<Vec<AnchorRecord>, _> = if let Some(kind) = q.kind {
        sqlx::query(
            "SELECT txid, block_hash, daa_score, block_time, proof_root, \
             manifest_hash, kind, indexed_at \
             FROM anchors WHERE kind = ?1 ORDER BY daa_score DESC LIMIT ?2 OFFSET ?3",
        )
        .bind(kind)
        .bind(limit)
        .bind(offset)
        .map(row_to_anchor)
        .fetch_all(st.pool.as_ref())
        .await
    } else {
        sqlx::query(
            "SELECT txid, block_hash, daa_score, block_time, proof_root, \
             manifest_hash, kind, indexed_at \
             FROM anchors ORDER BY daa_score DESC LIMIT ?1 OFFSET ?2",
        )
        .bind(limit)
        .bind(offset)
        .map(row_to_anchor)
        .fetch_all(st.pool.as_ref())
        .await
    };

    match rows {
        Ok(rows) => (StatusCode::OK, Json(serde_json::to_value(rows).unwrap())),
        Err(e)   => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// GET /api/lineage/:proof_root
///
/// Returns the chain of anchors reachable by following `parent_root`.
/// The lineage is stored in the manifest (off-chain); the indexer only has
/// `proof_root` → txid mappings.  This endpoint resolves the on-chain entries
/// for each hop in the lineage supplied via query param.
///
/// Since parent_root references are kept off-chain in the manifest, the caller
/// must supply the lineage chain.  This endpoint resolves each proof_root to
/// its on-chain anchor record.
async fn get_lineage(
    State(st): State<AppState>,
    Path(proof_root): Path<String>,
) -> impl IntoResponse {
    // Return all anchors whose proof_root == requested root, plus any that
    // reference it as a parent (future: requires parent_root column in DB).
    match fetch_anchor_by_root(&st.pool, &proof_root).await {
        Ok(Some(rec)) => {
            let lineage = vec![rec];
            (StatusCode::OK, Json(serde_json::to_value(lineage).unwrap()))
        }
        Ok(None) => (StatusCode::NOT_FOUND, Json(serde_json::json!({ "error": "not found" }))),
        Err(e)   => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e.to_string() }))),
    }
}

/// POST /api/verify
///
/// Body: { "cert": <Certificate JSON>, "manifest_only": true|false }
/// Verifies signature and manifest hash (file re-hash not supported here).
#[derive(Deserialize)]
struct VerifyRequest {
    cert:          Certificate,
    manifest_only: Option<bool>,
}

#[derive(Serialize)]
struct VerifyResponse {
    manifest_hash_ok: bool,
    signature_ok:     bool,
    proof_root_ok:    Option<bool>,
    overall:          bool,
    errors:           Vec<String>,
}

async fn verify_cert(
    State(st): State<AppState>,
    Json(req): Json<VerifyRequest>,
) -> impl IntoResponse {
    let cert = req.cert;
    let mut errors = Vec::<String>::new();

    // 1. manifest hash
    let recomputed = cert.manifest.hash_hex();
    let manifest_hash_ok = recomputed == cert.manifest_hash;
    if !manifest_hash_ok {
        errors.push(format!("manifest_hash mismatch: expected={} got={}", cert.manifest_hash, recomputed));
    }

    // 2. signature
    let digest = cert.manifest.hash_bytes();
    let signature_ok = verify_manifest_sig(&digest, &cert.issuer_sig, &cert.issuer_pubkey)
        .unwrap_or(false);
    if !signature_ok {
        errors.push("issuer signature invalid".to_owned());
    }

    // 3. on-chain anchor presence (optional — requires indexer DB)
    let proof_root_ok = if req.manifest_only.unwrap_or(false) {
        None
    } else if let Some(ref txid) = cert.txid {
        match fetch_anchor_by_txid(&st.pool, txid).await {
            Ok(Some(rec)) => {
                let ok = rec.proof_root == cert.manifest.proof_root
                    && rec.manifest_hash == cert.manifest_hash;
                if !ok {
                    errors.push("on-chain anchor payload does not match certificate".to_owned());
                }
                Some(ok)
            }
            Ok(None) => {
                errors.push(format!("txid {txid} not found in indexer"));
                Some(false)
            }
            Err(e) => {
                errors.push(format!("indexer query error: {e}"));
                None
            }
        }
    } else {
        None
    };

    let overall = manifest_hash_ok && signature_ok && proof_root_ok.unwrap_or(true);

    let resp = VerifyResponse { manifest_hash_ok, signature_ok, proof_root_ok, overall, errors };
    let status = if overall { StatusCode::OK } else { StatusCode::UNPROCESSABLE_ENTITY };
    (status, Json(serde_json::to_value(resp).unwrap()))
}

// ── DB helpers ────────────────────────────────────────────────────────────────

fn row_to_anchor(r: sqlx::sqlite::SqliteRow) -> AnchorRecord {
    AnchorRecord {
        txid:          r.get("txid"),
        block_hash:    r.get("block_hash"),
        daa_score:     r.get("daa_score"),
        block_time:    r.get("block_time"),
        proof_root:    r.get("proof_root"),
        manifest_hash: r.get("manifest_hash"),
        kind:          r.get("kind"),
        indexed_at:    r.get("indexed_at"),
    }
}

async fn fetch_anchor_by_root(pool: &SqlitePool, proof_root: &str) -> Result<Option<AnchorRecord>> {
    let rec = sqlx::query(
        "SELECT txid, block_hash, daa_score, block_time, proof_root, \
         manifest_hash, kind, indexed_at FROM anchors WHERE proof_root = ?1 LIMIT 1",
    )
    .bind(proof_root)
    .map(row_to_anchor)
    .fetch_optional(pool)
    .await?;
    Ok(rec)
}

async fn fetch_anchor_by_txid(pool: &SqlitePool, txid: &str) -> Result<Option<AnchorRecord>> {
    let rec = sqlx::query(
        "SELECT txid, block_hash, daa_score, block_time, proof_root, \
         manifest_hash, kind, indexed_at FROM anchors WHERE txid = ?1 LIMIT 1",
    )
    .bind(txid)
    .map(row_to_anchor)
    .fetch_optional(pool)
    .await?;
    Ok(rec)
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("bioproof-api")
        .about("BioProof REST API — query anchors, lineage and verify certificates")
        .arg(Arg::new("listen")
            .short('l').long("listen").value_name("ADDR")
            .default_value("0.0.0.0:8090")
            .help("Listen address for the REST API"))
        .arg(Arg::new("db-path")
            .short('d').long("db-path").value_name("PATH")
            .default_value("bioproof.db")
            .help("SQLite database written by bioproof-indexer"))
}
