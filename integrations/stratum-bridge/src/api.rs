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
    l2_jobs::L2JobSlot,
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
pub const DIFF_TO_HASHES: f64 = 2.0;

/// EWMA smoothing factor for per-share instant hashrate estimates.
/// α=0.20 gives a ~5-sample rolling average (heavier weight on recent shares).
const EWMA_ALPHA: f64 = 0.20;

/// Miners with no share for longer than this are considered stale (hashrate=0, offline).
const STALE_MINER_SECS: u64 = 300;

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
    /// SQLite database — `None` if `--db-path` was not provided
    pub db:              Option<Arc<Db>>,
    /// Stratum TCP listen address e.g. "0.0.0.0:1444" — exposed in the API
    pub stratum_endpoint: String,
    /// Pool theme: "genetics" | "climate" | "ai" | "materials" | "generic"
    pub theme:           String,
    /// Current L2 job slot — `None` if L2 dispatch is disabled
    pub l2_slot:         Option<L2JobSlot>,
}

impl ApiState {
    pub fn new(
        accounting:       Arc<Mutex<Accounting>>,
        rpc:              Arc<GrpcClient>,
        pool_name:        String,
        db:               Option<Arc<Db>>,
        stratum_endpoint: String,
        theme:            String,
        l2_slot:          Option<L2JobSlot>,
    ) -> Self {
        Self {
            accounting,
            rpc,
            miners:           Arc::new(Mutex::new(HashMap::new())),
            connected_count:  Arc::new(AtomicU32::new(0)),
            start_unix:       unix_now(),
            pool_name,
            db,
            stratum_endpoint,
            theme,
            l2_slot,
        }
    }
}

// ── Response types ────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct PoolStats {
    name:              String,
    stratum_endpoint:  String,
    uptime_secs:       u64,
    connected_miners:  u32,
    pool_hashrate_hps: f64,
    total_shares:      u64,
    total_blocks:      u64,
    pending_payouts:   usize,
    paid_payouts:      usize,
    theme:             String,
    l2_enabled:        bool,
}

#[derive(Serialize)]
struct L2Info {
    theme:        String,
    enabled:      bool,
    job_id:       Option<String>,
    task:         Option<String>,
    dataset:      Option<String>,
    dataset_url:  Option<String>,
    fragment:     Option<u64>,
    reward_sompi: Option<u64>,
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
struct TxRecord {
    id:           i64,
    tx_id:        String,
    submitted_at: u64,
    status:       String,
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
        if let Ok(blocks) = db.get_blocks(30).await {
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

async fn get_payments(State(s): State<ApiState>) -> Json<Vec<TxRecord>> {
    if let Some(db) = &s.db {
        if let Ok(txs) = db.get_transactions(30).await {
            return Json(
                txs.into_iter()
                    .map(|t| TxRecord {
                        id:           t.id,
                        tx_id:        t.tx_id,
                        submitted_at: t.submitted_at as u64,
                        status:       t.status,
                    })
                    .collect(),
            );
        }
    }
    Json(vec![])
}

async fn get_l2(State(s): State<ApiState>) -> Json<L2Info> {
    let theme   = s.theme.clone();
    let enabled = s.l2_slot.is_some();
    if let Some(slot) = &s.l2_slot {
        if let Ok(guard) = slot.try_read() {
            if let Some(job) = guard.as_ref() {
                return Json(L2Info {
                    theme,
                    enabled,
                    job_id:       Some(job.job_id.clone()),
                    task:         Some(job.task.clone()),
                    dataset:      Some(job.dataset.clone()),
                    dataset_url:  job.dataset_url.clone(),
                    fragment:     Some(job.fragment),
                    reward_sompi: Some(job.reward_sompi),
                });
            }
        }
    }
    Json(L2Info { theme, enabled, job_id: None, task: None, dataset: None, dataset_url: None, fragment: None, reward_sompi: None })
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
        .route("/api/l2",             get(get_l2))
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
            stratum_endpoint:  s.stratum_endpoint.clone(),
            uptime_secs:       unix_now().saturating_sub(s.start_unix),
            connected_miners:  connected,
            pool_hashrate_hps: pool_hps,
            total_shares,
            total_blocks,
            pending_payouts:   pending,
            paid_payouts:      paid,
            theme:             s.theme.clone(),
            l2_enabled:        s.l2_slot.is_some(),
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
        stratum_endpoint:  s.stratum_endpoint.clone(),
        uptime_secs:       unix_now().saturating_sub(s.start_unix),
        connected_miners:  s.connected_count.load(Ordering::Relaxed),
        pool_hashrate_hps: pool_hps,
        total_shares,
        total_blocks,
        pending_payouts:   pending,
        paid_payouts:      paid,
        theme:             s.theme.clone(),
        l2_enabled:        s.l2_slot.is_some(),
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
    let last_share = m.last_share as u64;
    let stale      = last_share > 0 && unix_now().saturating_sub(last_share) > STALE_MINER_SECS;
    let hashrate   = if stale { 0.0 } else { live.map(|e| e.hashrate_hps).unwrap_or(m.hashrate_hps) };
    let connected  = if stale { false } else { live.map(|e| e.connected).unwrap_or(m.connected) };
    MinerStats {
        worker:             m.worker.clone(),
        address:            m.address.clone(),
        connected_since:    live.map(|e| e.connected_since).unwrap_or(m.first_seen as u64),
        last_share_at:      last_share,
        shares_submitted:   m.shares_total as u64,
        blocks_found:       m.blocks_total as u64,
        current_difficulty: live.map(|e| e.current_difficulty).unwrap_or(m.current_diff),
        hashrate_hps:       hashrate,
        connected,
    }
}

fn entry_to_stats(m: &MinerApiEntry) -> MinerStats {
    let stale     = m.last_share_at > 0 && unix_now().saturating_sub(m.last_share_at) > STALE_MINER_SECS;
    let hashrate  = if stale { 0.0 } else { m.hashrate_hps };
    let connected = if stale { false } else { m.connected };
    MinerStats {
        worker:             m.worker.clone(),
        address:            m.address.clone(),
        connected_since:    m.connected_since,
        last_share_at:      m.last_share_at,
        shares_submitted:   m.shares_submitted,
        blocks_found:       m.blocks_found,
        current_difficulty: m.current_difficulty,
        hashrate_hps:       hashrate,
        connected,
    }
}

fn status_pair(s: &PayoutStatus) -> (String, Option<String>) {
    match s {
        PayoutStatus::Pending              => ("pending".into(), None),
        PayoutStatus::Paid   { tx_id }    => ("paid".into(), Some(tx_id.clone())),
        PayoutStatus::Failed { reason: _ } => ("failed".into(), None),
    }
}

/// Update per-miner hashrate from the *current* share timing.
///
/// Call this **before** writing `entry.last_share_at = now_secs` so that
/// `entry.last_share_at` still holds the previous share timestamp.
/// On the first share (last_share_at == 0) the hashrate is left at 0.
pub fn update_miner_hashrate(entry: &mut MinerApiEntry, diff: f64, now_secs: u64) {
    entry.current_difficulty = diff;
    if entry.last_share_at > 0 && now_secs > entry.last_share_at {
        let elapsed   = (now_secs - entry.last_share_at) as f64;
        let instant   = diff * DIFF_TO_HASHES / elapsed;
        entry.hashrate_hps = if entry.hashrate_hps > 0.0 {
            EWMA_ALPHA * instant + (1.0 - EWMA_ALPHA) * entry.hashrate_hps
        } else {
            instant
        };
    }
}

fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs()
}
