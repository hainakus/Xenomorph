use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Json, Response},
    routing::get,
    Router,
};
use kaspa_core::info;
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::api::rpc::RpcApi;
use serde::Serialize;
use tokio::{net::TcpListener, sync::Mutex};
use tower_http::cors::CorsLayer;

use crate::{
    accounting::{Accounting, PayoutStatus},
    db::Db,
};

// ── Embedded SPA ────────────────────────────────────────────────────────────
const INDEX_HTML: &str = include_str!("pool.html");

async fn serve_index() -> impl IntoResponse {
    Response::builder()
        .header(header::CONTENT_TYPE, "text/html; charset=utf-8")
        .body(INDEX_HTML.to_owned())
        .unwrap()
}

// ── Pool-wide factor to convert share-difficulty → hash-attempts ──────────────
// 2^256 / MAX_DIFFICULTY_TARGET_F64 ≈ 2.0  (exact: ~2.00)
// Keeps pool hashrate on the same H/s scale as estimateNetworkHashesPerSecond.
const DIFF_TO_HASHES: f64 = 2.0;

// ── Shared miner entry (live map updated by stratum connections) ───────────────

#[derive(Clone, Debug)]
pub struct MinerApiEntry {
    pub worker:             String,
    pub address:            String,
    pub connected_since:    u64,
    pub last_share_at:      u64,
    pub shares_submitted:   u64,
    pub blocks_found:       u64,
    pub current_difficulty: f64,
    pub hashrate_hps:       f64,
    pub connected:          bool,
}

// ── Shared API state ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ApiState {
    pub accounting:      Arc<Mutex<Accounting>>,
    pub rpc:             Arc<GrpcClient>,
    /// Live per-worker map; key = worker name (for real-time hashrate overlay)
    pub miners:          Arc<Mutex<HashMap<String, MinerApiEntry>>>,
    pub connected_count: Arc<AtomicU32>,
    pub start_unix:      u64,
    pub pool_name:       String,
    pub target_spm:      f64,
    /// SQLite database — `None` if `--db-path` was not provided
    pub db:              Option<Arc<Db>>,
}

impl ApiState {
    pub fn new(
        accounting: Arc<Mutex<Accounting>>,
        rpc:        Arc<GrpcClient>,
        pool_name:  String,
        target_spm: f64,
        db:         Option<Arc<Db>>,
    ) -> Self {
        Self {
            accounting,
            rpc,
            miners:          Arc::new(Mutex::new(HashMap::new())),
            connected_count: Arc::new(AtomicU32::new(0)),
            start_unix:      unix_now(),
            pool_name,
            target_spm,
            db,
        }
    }
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PoolStats {
    name:              String,
    uptime_secs:       u64,
    connected_miners:  u32,
    pool_hashrate_hps: f64,
    total_shares:      u64,
    total_blocks:      u64,
    pending_payouts:   usize,
    paid_payouts:      usize,
}

#[derive(Serialize)]
struct MinerStats {
    worker:             String,
    address:            String,
    connected_since:    u64,
    last_share_at:      u64,
    shares_submitted:   u64,
    blocks_found:       u64,
    current_difficulty: f64,
    hashrate_hps:       f64,
    connected:          bool,
}

#[derive(Serialize)]
struct BlockRecord {
    job_id:          String,
    found_at:        u64,
    block_daa_score: u64,
    top_miner:       Option<String>,
    top_proportion:  Option<f64>,
    status:          String,
    tx_id:           Option<String>,
}

#[derive(Serialize)]
struct PaymentRecord {
    job_id:          String,
    found_at:        u64,
    block_daa_score: u64,
    status:          String,
    tx_id:           Option<String>,
    payouts:         Vec<PayoutShare>,
}

#[derive(Serialize)]
struct PayoutShare {
    worker:     String,
    proportion: f64,
}

#[derive(Serialize)]
struct NetworkInfo {
    virtual_daa_score:    u64,
    network_hashrate_hps: u64,
    sink_hash:            String,
    difficulty:           f64,
}

#[derive(Serialize)]
struct FullStats {
    pool:    PoolStats,
    network: Option<NetworkInfo>,
    miners:  Vec<MinerStats>,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

async fn get_pool(State(s): State<ApiState>) -> Json<PoolStats> {
    Json(build_pool_stats(&s).await)
}

async fn get_miners(State(s): State<ApiState>) -> Json<Vec<MinerStats>> {
    Json(build_miners_list(&s).await)
}

async fn get_miner(
    State(s): State<ApiState>,
    Path(worker): Path<String>,
) -> Result<Json<MinerStats>, StatusCode> {
    // DB lookup first (includes historical / disconnected miners)
    if let Some(db) = &s.db {
        if let Ok(Some(m)) = db.get_miner(&worker).await {
            let live = s.miners.lock().await;
            let stats = db_miner_to_stats(&m, live.get(&m.worker));
            return Ok(Json(stats));
        }
    }
    // Fallback: in-memory only
    let miners = s.miners.lock().await;
    miners
        .values()
        .find(|m| m.worker == worker || m.address == worker)
        .map(|m| Json(entry_to_stats(m)))
        .ok_or(StatusCode::NOT_FOUND)
}

async fn get_blocks(State(s): State<ApiState>) -> Json<Vec<BlockRecord>> {
    if let Some(db) = &s.db {
        if let Ok(blocks) = db.get_blocks(200).await {
            let mut records = Vec::with_capacity(blocks.len());
            for b in &blocks {
                let payouts = db.get_block_payouts(&b.job_id).await.unwrap_or_default();
                let top = payouts
                    .iter()
                    .max_by(|a, b| a.proportion.partial_cmp(&b.proportion).unwrap_or(std::cmp::Ordering::Equal));
                records.push(BlockRecord {
                    job_id:          b.job_id.clone(),
                    found_at:        b.found_at as u64,
                    block_daa_score: b.block_daa_score as u64,
                    top_miner:       top.map(|p| p.worker.clone()),
                    top_proportion:  top.map(|p| p.proportion),
                    status:          b.status.clone(),
                    tx_id:           b.tx_id.clone(),
                });
            }
            return Json(records);
        }
    }
    // Fallback: in-memory accounting
    let acct = s.accounting.lock().await;
    Json(
        acct.pending_payouts()
            .iter()
            .rev()
            .map(|p| {
                let top = p.proportions.iter().max_by(|a, b| {
                    a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let (status, tx_id) = status_pair(&p.status);
                BlockRecord {
                    job_id:          p.job_id.clone(),
                    found_at:        p.unix_secs,
                    block_daa_score: p.block_daa_score,
                    top_miner:       top.map(|(w, _)| w.clone()),
                    top_proportion:  top.map(|(_, v)| *v),
                    status,
                    tx_id,
                }
            })
            .collect(),
    )
}

async fn get_payments(State(s): State<ApiState>) -> Json<Vec<PaymentRecord>> {
    if let Some(db) = &s.db {
        if let Ok(blocks) = db.get_blocks(200).await {
            let mut records = Vec::with_capacity(blocks.len());
            for b in &blocks {
                let payouts = db.get_block_payouts(&b.job_id).await.unwrap_or_default();
                records.push(PaymentRecord {
                    job_id:          b.job_id.clone(),
                    found_at:        b.found_at as u64,
                    block_daa_score: b.block_daa_score as u64,
                    status:          b.status.clone(),
                    tx_id:           b.tx_id.clone(),
                    payouts:         payouts
                        .iter()
                        .map(|p| PayoutShare { worker: p.worker.clone(), proportion: p.proportion })
                        .collect(),
                });
            }
            return Json(records);
        }
    }
    // Fallback: in-memory accounting
    let acct = s.accounting.lock().await;
    Json(
        acct.pending_payouts()
            .iter()
            .rev()
            .map(|p| {
                let (status, tx_id) = status_pair(&p.status);
                PaymentRecord {
                    job_id:          p.job_id.clone(),
                    found_at:        p.unix_secs,
                    block_daa_score: p.block_daa_score,
                    status,
                    tx_id,
                    payouts:         p.proportions
                        .iter()
                        .map(|(w, v)| PayoutShare { worker: w.clone(), proportion: *v })
                        .collect(),
                }
            })
            .collect(),
    )
}

async fn get_network(
    State(s): State<ApiState>,
) -> Result<Json<NetworkInfo>, StatusCode> {
    let dag = s.rpc.get_block_dag_info().await.map_err(|_| StatusCode::SERVICE_UNAVAILABLE)?;
    let hashrate = s.rpc.estimate_network_hashes_per_second(1000, None).await.unwrap_or(0);
    Ok(Json(NetworkInfo {
        virtual_daa_score:    dag.virtual_daa_score,
        network_hashrate_hps: hashrate,
        sink_hash:            dag.sink.to_string(),
        difficulty:           dag.difficulty,
    }))
}

async fn get_stats(State(s): State<ApiState>) -> Json<FullStats> {
    let pool   = build_pool_stats(&s).await;
    let miners = build_miners_list(&s).await;
    let network = match s.rpc.get_block_dag_info().await {
        Ok(dag) => {
            let hashrate = s.rpc.estimate_network_hashes_per_second(1000, None).await.unwrap_or(0);
            Some(NetworkInfo {
                virtual_daa_score:    dag.virtual_daa_score,
                network_hashrate_hps: hashrate,
                sink_hash:            dag.sink.to_string(),
                difficulty:           dag.difficulty,
            })
        }
        Err(_) => None,
    };
    Json(FullStats { pool, network, miners })
}

// ── Server entry ──────────────────────────────────────────────────────────────

pub async fn run_api_server(
    listen: std::net::SocketAddr,
    state:  ApiState,
) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/",                   get(serve_index))
        .route("/index.html",         get(serve_index))
        .route("/api/pool",           get(get_pool))
        .route("/api/miners",         get(get_miners))
        .route("/api/miners/:worker", get(get_miner))
        .route("/api/blocks",         get(get_blocks))
        .route("/api/payments",       get(get_payments))
        .route("/api/network",        get(get_network))
        .route("/api/stats",          get(get_stats))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = TcpListener::bind(listen).await?;
    info!("Pool API listening on http://{listen}");
    axum::serve(listener, app).await?;
    Ok(())
}

// ── Internal helpers ──────────────────────────────────────────────────────────

async fn build_pool_stats(s: &ApiState) -> PoolStats {
    if let Some(db) = &s.db {
        let total_shares  = db.count_shares().await.unwrap_or(0) as u64;
        let total_blocks  = db.count_blocks().await.unwrap_or(0) as u64;
        let pool_hps      = db.total_pool_hashrate().await.unwrap_or(0.0);
        let connected     = db.count_connected_miners().await.unwrap_or(0) as u32;

        // Payout counts still come from in-memory accounting (always accurate)
        let acct    = s.accounting.lock().await;
        let pending = acct.pending_payouts().iter().filter(|p| p.status == PayoutStatus::Pending).count();
        let paid    = acct.pending_payouts().iter().filter(|p| matches!(p.status, PayoutStatus::Paid { .. })).count();

        return PoolStats {
            name:              s.pool_name.clone(),
            uptime_secs:       unix_now().saturating_sub(s.start_unix),
            connected_miners:  connected,
            pool_hashrate_hps: pool_hps,
            total_shares,
            total_blocks,
            pending_payouts:   pending,
            paid_payouts:      paid,
        };
    }

    // Fallback: in-memory
    let acct   = s.accounting.lock().await;
    let miners = s.miners.lock().await;
    let total_shares: u64 = acct.worker_stats().values().map(|w| w.shares_submitted).sum();
    let total_blocks: u64 = acct.worker_stats().values().map(|w| w.blocks_found).sum();
    let pool_hps: f64     = miners.values().filter(|m| m.connected).map(|m| m.hashrate_hps).sum();
    let pending           = acct.pending_payouts().iter().filter(|p| p.status == PayoutStatus::Pending).count();
    let paid              = acct.pending_payouts().iter().filter(|p| matches!(p.status, PayoutStatus::Paid { .. })).count();

    PoolStats {
        name:              s.pool_name.clone(),
        uptime_secs:       unix_now().saturating_sub(s.start_unix),
        connected_miners:  s.connected_count.load(Ordering::Relaxed),
        pool_hashrate_hps: pool_hps,
        total_shares,
        total_blocks,
        pending_payouts:   pending,
        paid_payouts:      paid,
    }
}

async fn build_miners_list(s: &ApiState) -> Vec<MinerStats> {
    if let Some(db) = &s.db {
        if let Ok(db_miners) = db.get_all_miners().await {
            let live = s.miners.lock().await;
            return db_miners.iter().map(|m| db_miner_to_stats(m, live.get(&m.worker))).collect();
        }
    }
    // Fallback
    let miners = s.miners.lock().await;
    miners.values().map(entry_to_stats).collect()
}

/// Merge a DB miner row with an optional live in-memory entry.
/// Live entry provides the most current `connected`, `hashrate_hps`, and `current_difficulty`.
fn db_miner_to_stats(
    m:    &crate::db::DbMiner,
    live: Option<&MinerApiEntry>,
) -> MinerStats {
    MinerStats {
        worker:             m.worker.clone(),
        address:            m.address.clone(),
        connected_since:    live.map(|e| e.connected_since).unwrap_or(m.first_seen as u64),
        last_share_at:      m.last_share as u64,
        shares_submitted:   m.shares_total as u64,
        blocks_found:       m.blocks_total as u64,
        current_difficulty: live.map(|e| e.current_difficulty).unwrap_or(m.current_diff),
        hashrate_hps:       live.map(|e| e.hashrate_hps).unwrap_or(m.hashrate_hps),
        connected:          live.map(|e| e.connected).unwrap_or(m.connected),
    }
}

fn entry_to_stats(m: &MinerApiEntry) -> MinerStats {
    MinerStats {
        worker:             m.worker.clone(),
        address:            m.address.clone(),
        connected_since:    m.connected_since,
        last_share_at:      m.last_share_at,
        shares_submitted:   m.shares_submitted,
        blocks_found:       m.blocks_found,
        current_difficulty: m.current_difficulty,
        hashrate_hps:       m.hashrate_hps,
        connected:          m.connected,
    }
}

fn status_pair(s: &PayoutStatus) -> (String, Option<String>) {
    match s {
        PayoutStatus::Pending              => ("pending".into(), None),
        PayoutStatus::Paid   { tx_id }    => ("paid".into(), Some(tx_id.clone())),
        PayoutStatus::Failed { reason: _ } => ("failed".into(), None),
    }
}

pub fn update_miner_hashrate(entry: &mut MinerApiEntry, diff: f64, target_spm: f64) {
    entry.current_difficulty = diff;
    entry.hashrate_hps       = diff * (target_spm / 60.0) * DIFF_TO_HASHES;
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
