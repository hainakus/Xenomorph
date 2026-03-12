use std::{
    collections::HashSet,
    net::SocketAddr,
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

use anyhow::{anyhow, Context, Result};
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::{api::rpc::RpcApi, SubmitBlockReport};
use serde_json::Value;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::{tcp::OwnedWriteHalf, TcpListener, TcpStream},
    sync::{watch, Mutex, RwLock},
};

use kaspa_consensus_core::header::Header;
use kaspa_pow::State as KHeavyState;

use crate::{
    accounting::Accounting,
    api::{ApiState, MinerApiEntry, update_miner_hashrate},
    db::Db,
    job::{Job, JobManager},
    proto::{StratumNotification, StratumRequest, StratumResponse},
    vardiff::{VarDiff, VarDiffConfig},
};

/// Approximate maximum difficulty target for Xenom (genesis compact bits ≋ 0x207fffff).
/// share_target = MAX_DIFFICULTY_TARGET_F64 / share_difficulty
/// Higher share_difficulty → smaller target → harder shares.
const MAX_DIFFICULTY_TARGET_F64: f64 = 5.8e76_f64;

/// Convert a compact `bits` field to floating-point difficulty.
/// `difficulty = MAX_DIFFICULTY_TARGET / target`  where
/// `target = (bits & 0xFFFFFF) * 2^(8 * ((bits >> 24) - 3))`.
fn bits_to_diff(bits: u32) -> f64 {
    let exponent  = (bits >> 24) as i32;
    let mantissa  = (bits & 0x00FF_FFFF) as f64;
    let target    = mantissa * 2.0_f64.powi(8 * (exponent - 3));
    if target <= 0.0 { return 1.0; }
    MAX_DIFFICULTY_TARGET_F64 / target
}

// Monotonically-increasing extranonce1 counter.
static EXTRANONCE_COUNTER: AtomicU32 = AtomicU32::new(1);

// ── Submit outcome + error ────────────────────────────────────────────

enum SubmitOutcome {
    /// Node accepted → block found!  Carries the template's DAA score + 1.
    Block { daa_score: u64 },
    /// Valid submission but below block target (normal share)
    Share,
}

/// Typed rejection reason so the caller can map to Stratum error codes.
#[derive(Debug)]
enum ShareError {
    /// Malformed extranonce2, wrong length, etc.  (code 20)
    BadFormat(String),
    /// Job ID unknown or expired.                 (code 21)
    Stale(String),
    /// Exact (job_id, extranonce2) already seen.  (code 22)
    Duplicate,
    /// Hash does not meet share difficulty target. (code 23)
    LowDifficulty { hash: f64, target: f64 },
}

impl std::fmt::Display for ShareError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadFormat(s)             => write!(f, "bad format: {s}"),
            Self::Stale(s)                 => write!(f, "stale job: {s}"),
            Self::Duplicate                => write!(f, "duplicate share"),
            Self::LowDifficulty{hash,target} =>
                write!(f, "low difficulty  hash={hash:.2e} target={target:.2e}"),
        }
    }
}

// ── Public entry point ────────────────────────────────────────────────────────

pub async fn run_server(
    listen_addr:  SocketAddr,
    job_rx:       watch::Receiver<Option<Arc<Job>>>,
    job_mgr:      Arc<RwLock<JobManager>>,
    rpc:          Arc<GrpcClient>,
    vardiff_cfg:  VarDiffConfig,
    accounting:   Arc<Mutex<Accounting>>,
    api_state:    Option<ApiState>,
    db:           Option<Arc<Db>>,
) -> Result<()> {
    let listener = TcpListener::bind(listen_addr).await.context("bind stratum port")?;
    info!("Stratum server listening on {listen_addr}");

    loop {
        let (stream, peer) = listener.accept().await.context("accept")?;
        info!("Miner connected: {peer}");

        let jrx    = job_rx.clone();
        let jmgr   = job_mgr.clone();
        let rpc2   = rpc.clone();
        let vdcfg  = vardiff_cfg.clone();
        let acct   = accounting.clone();
        let api    = api_state.clone();
        let db2    = db.clone();

        tokio::spawn(async move {
            if let Err(e) = handle_miner(stream, peer, jrx, jmgr, rpc2, vdcfg, acct, api, db2).await {
                warn!("Miner {peer} disconnected: {e}");
            } else {
                info!("Miner {peer} disconnected");
            }
        });
    }
}

// ── Per-miner connection ──────────────────────────────────────────────────────

async fn handle_miner(
    stream:      TcpStream,
    peer:        SocketAddr,
    mut job_rx:  watch::Receiver<Option<Arc<Job>>>,
    job_mgr:     Arc<RwLock<JobManager>>,
    rpc:         Arc<GrpcClient>,
    vardiff_cfg: VarDiffConfig,
    accounting:  Arc<Mutex<Accounting>>,
    api_state:   Option<ApiState>,
    db:          Option<Arc<Db>>,
) -> Result<()> {
    // ── API: register connection ──────────────────────────────────────────────
    if let Some(ref api) = api_state {
        api.connected_count.fetch_add(1, Ordering::Relaxed);
    }
    let extranonce1     = EXTRANONCE_COUNTER.fetch_add(1, Ordering::Relaxed);
    let extranonce1_hex = format!("{extranonce1:08x}");
    const EXTRANONCE2_SIZE: usize = 4;

    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    let mut authorized   = false;
    let mut worker_name  = String::from("unknown");
    let mut vardiff      = VarDiff::new(vardiff_cfg);
    // Per-miner duplicate tracker: cleared whenever a new job replaces the old one.
    let mut seen_shares: HashSet<(String, u32)> = HashSet::new();

    // Mark current value as seen so the first `changed()` fires on a genuinely
    // *new* job after authorization, not the already-known current one.
    job_rx.borrow_and_update();

    loop {
        tokio::select! {
            // ── new job broadcast from node poller ───────────────────────
            changed = job_rx.changed() => {
                if changed.is_err() { break; }
                // New job = new nonce space; discard previous duplicate records
                seen_shares.clear();
                if authorized {
                    let job = job_rx.borrow().clone();
                    if let Some(j) = job {
                        send_notify(&mut writer, &j, true).await?;
                    }
                }
            }

            // ── message from miner ───────────────────────────────────────
            line = lines.next_line() => {
                let line = match line? {
                    Some(l) if !l.trim().is_empty() => l,
                    Some(_) => continue,
                    None    => break,  // EOF / disconnect
                };

                let req: StratumRequest = match serde_json::from_str(&line) {
                    Ok(r)  => r,
                    Err(e) => { warn!("{peer}: JSON parse error: {e}"); continue; }
                };

                let id     = req.id.clone();
                let params = req.params.as_array().cloned().unwrap_or_default();

                match req.method.as_str() {

                    // ── subscribe ─────────────────────────────────────────
                    "mining.subscribe" => {
                        let result = serde_json::json!([
                            [
                                ["mining.set_difficulty", &extranonce1_hex],
                                ["mining.notify",         &extranonce1_hex]
                            ],
                            &extranonce1_hex,
                            EXTRANONCE2_SIZE
                        ]);
                        write_line(&mut writer, &StratumResponse::ok(id, result)).await?;
                    }

                    // ── authorize ─────────────────────────────────────────
                    "mining.authorize" => {
                        worker_name = params.first()
                            .and_then(|v| v.as_str())
                            .unwrap_or("unknown")
                            .to_owned();
                        authorized = true;
                        info!("{peer}: authorized as {worker_name}  init_diff={}", vardiff.current_diff);

                        // ── API: register miner ───────────────────────────
                        let conn_now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
                        let miner_address = worker_name.split('.').next().unwrap_or(&worker_name).to_owned();

                        // Warn if worker name doesn't start with a valid xenom: address — payouts will fail
                        if kaspa_addresses::Address::try_from(miner_address.as_str()).is_err() {
                            warn!(
                                "{peer}: worker '{worker_name}' does not start with a valid xenom: address. \
                                 Auto-payouts require username format  xenom:qYOURADDR[.workername]  — \
                                 shares will be recorded but payouts CANNOT be sent to this worker."
                            );
                        }

                        if let Some(ref api) = api_state {
                            let mut e = MinerApiEntry {
                                worker:             worker_name.clone(),
                                address:            miner_address.clone(),
                                connected_since:    conn_now,
                                last_share_at:      0,
                                shares_submitted:   0,
                                blocks_found:       0,
                                current_difficulty: vardiff.current_diff,
                                hashrate_hps:       0.0,
                                connected:          true,
                            };
                            update_miner_hashrate(&mut e, vardiff.current_diff, conn_now);
                            api.miners.lock().await.insert(worker_name.clone(), e);
                        }

                        // ── DB: upsert miner ─────────────────────────────────
                        if let Some(ref d) = db {
                            let d = d.clone();
                            let w = worker_name.clone();
                            let a = miner_address.clone();
                            tokio::spawn(async move {
                                if let Err(e) = d.upsert_miner_connected(&w, &a, conn_now as i64).await {
                                    warn!("DB upsert_miner_connected: {e}");
                                }
                            });
                        }

                        write_line(&mut writer, &StratumResponse::ok(id, Value::Bool(true))).await?;

                        // Seed VarDiff from current block difficulty so the first
                        // set_difficulty is meaningful and hashrate estimates are correct
                        // before VarDiff has had time to converge.
                        if let Some(j) = job_rx.borrow().as_ref() {
                            let block_diff = bits_to_diff(j.template.header.bits);
                            if block_diff > vardiff.current_diff {
                                vardiff.current_diff = block_diff;
                            }
                        }

                        // Send initial difficulty then the current job
                        write_line(&mut writer, &StratumNotification::set_difficulty(vardiff.current_diff)).await?;

                        let job = job_rx.borrow().clone();
                        if let Some(j) = job {
                            send_notify(&mut writer, &j, true).await?;
                        }
                    }

                    // ── submit ────────────────────────────────────────────
                    "mining.submit" => {
                        if !authorized {
                            write_line(&mut writer, &StratumResponse::err(id, 24, "Unauthorized")).await?;
                            continue;
                        }

                        let job_id  = str_param(&params, 1, "job_id")?;
                        let en2_hex = str_param(&params, 2, "extranonce2")?;

                        match process_submit(
                            job_id, en2_hex, extranonce1,
                            &rpc, &job_mgr,
                            &mut seen_shares,
                            vardiff.current_diff,
                        ).await {
                            Ok((outcome, job_bits)) => {
                                let is_block = matches!(outcome, SubmitOutcome::Block { .. });

                                // Effective difficulty = max(block difficulty from bits, VarDiff)
                                // This ensures hashrate is correct even before VarDiff converges.
                                let effective_diff = {
                                    let bd = bits_to_diff(job_bits);
                                    if bd > vardiff.current_diff { bd } else { vardiff.current_diff }
                                };

                                // ── VarDiff retarget ─────────────────────
                                if let Some(new_diff) = vardiff.on_share() {
                                    info!("{peer}/{worker_name}: vardiff → {new_diff:.4}");
                                    write_line(
                                        &mut writer,
                                        &StratumNotification::set_difficulty(new_diff),
                                    ).await?;
                                    // Hashrate update happens below at share-submit time;
                                    // vardiff.current_diff is already new_diff at that point.
                                }

                                // ── PPLNS accounting ─────────────────────
                                let block_payout = {
                                    let mut acct = accounting.lock().await;
                                    acct.record_share(&worker_name, vardiff.current_diff);
                                    if let SubmitOutcome::Block { daa_score } = &outcome {
                                        Some(acct.record_block(job_id, *daa_score))
                                    } else {
                                        None
                                    }
                                };

                                // ── DB: persist block + payouts ─────────
                                if let (Some(ref d), Some(ref payout)) = (&db, &block_payout) {
                                    let d = d.clone();
                                    let p = payout.clone();
                                    tokio::spawn(async move {
                                        if let Err(e) = d.insert_block(&p.job_id, p.unix_secs as i64, p.block_daa_score as i64).await {
                                            warn!("DB insert_block: {e}");
                                        }
                                        if let Err(e) = d.insert_block_payouts(&p.job_id, &p.proportions).await {
                                            warn!("DB insert_block_payouts: {e}");
                                        }
                                    });
                                }

                                // ── API: update share stats ───────────────
                                let share_now = std::time::SystemTime::now()
                                    .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();

                                if let Some(ref api) = api_state {
                                    if let Some(e) = api.miners.lock().await.get_mut(&worker_name) {
                                        // update_miner_hashrate uses e.last_share_at as the
                                        // *previous* share time — call BEFORE overwriting it.
                                        update_miner_hashrate(e, effective_diff, share_now);
                                        e.last_share_at    = share_now;
                                        e.shares_submitted += 1;
                                        if is_block { e.blocks_found += 1; }
                                    }
                                }

                                // ── DB: record share + miner stats ────────
                                if let Some(ref d) = db {
                                    let d           = d.clone();
                                    let worker      = worker_name.clone();
                                    let jid         = job_id.to_owned();
                                    let diff        = effective_diff;
                                    // Read computed hashrate from the entry we just updated
                                    let hashrate    = api_state.as_ref()
                                        .and_then(|api| {
                                            // non-blocking peek — if locked, fall back to 0
                                            api.miners.try_lock().ok()
                                                .and_then(|m| m.get(&worker_name).map(|e| e.hashrate_hps))
                                        })
                                        .unwrap_or(0.0);
                                    let blk         = is_block;
                                    tokio::spawn(async move {
                                        if let Err(e) = d.insert_share(&worker, &jid, diff, share_now as i64).await {
                                            warn!("DB insert_share: {e}");
                                        }
                                        if let Err(e) = d.upsert_miner_share(&worker, share_now as i64, diff, hashrate).await {
                                            warn!("DB upsert_miner_share: {e}");
                                        }
                                        if blk {
                                            if let Err(e) = d.upsert_miner_block(&worker).await {
                                                warn!("DB upsert_miner_block: {e}");
                                            }
                                        }
                                    });
                                }

                                if is_block {
                                    info!("{peer}/{worker_name}: block ACCEPTED (job {job_id})");
                                } else {
                                    info!("{peer}/{worker_name}: share OK diff={:.4} (job {job_id})", vardiff.current_diff);
                                }
                                write_line(&mut writer, &StratumResponse::ok(id, Value::Bool(true))).await?;
                            }

                            // ── Typed share rejections ─────────────────
                            Err(ShareError::Duplicate) => {
                                warn!("{peer}/{worker_name}: DUPLICATE share (job {job_id})");
                                write_line(&mut writer, &StratumResponse::err(id, 22, "Duplicate share")).await?;
                            }
                            Err(ShareError::LowDifficulty { hash, target }) => {
                                warn!("{peer}/{worker_name}: LOW DIFF hash={hash:.2e} target={target:.2e}");
                                write_line(&mut writer, &StratumResponse::err(id, 23, "Low difficulty")).await?;
                            }
                            Err(ShareError::Stale(ref s)) => {
                                warn!("{peer}/{worker_name}: STALE {s}");
                                write_line(&mut writer, &StratumResponse::err(id, 21, "Stale job")).await?;
                            }
                            Err(ShareError::BadFormat(ref s)) => {
                                warn!("{peer}/{worker_name}: BAD FORMAT {s}");
                                write_line(&mut writer, &StratumResponse::err(id, 20, "Bad format")).await?;
                            }
                        }
                    }

                    // ── extranonce subscribe (optional miner feature) ─────
                    "mining.extranonce.subscribe" => {
                        write_line(&mut writer, &StratumResponse::ok(id, Value::Bool(true))).await?;
                    }

                    // ── keepalive ─────────────────────────────────────────
                    "eth_submitHashrate" | "mining.ping" => {
                        write_line(&mut writer, &StratumResponse::ok(id, Value::Bool(true))).await?;
                    }

                    other => {
                        warn!("{peer}: unknown method '{other}'");
                        write_line(&mut writer, &StratumResponse::err(id, 21, "Unknown method")).await?;
                    }
                }
            }
        }
    }

    // ── API: mark miner disconnected ─────────────────────────────────────────
    if let Some(ref api) = api_state {
        api.connected_count.fetch_sub(1, Ordering::Relaxed);
        if let Some(e) = api.miners.lock().await.get_mut(&worker_name) {
            e.connected = false;
        }
    }

    // ── DB: mark miner disconnected ──────────────────────────────────────────
    if let Some(ref d) = db {
        let d = d.clone();
        let w = worker_name.clone();
        tokio::spawn(async move {
            if let Err(e) = d.set_miner_disconnected(&w).await {
                warn!("DB set_miner_disconnected: {e}");
            }
        });
    }

    Ok(())
}

// ── submit processing ──────────────────────────────────────────────────

async fn process_submit(
    job_id:      &str,
    en2_hex:     &str,
    extranonce1: u32,
    rpc:         &Arc<GrpcClient>,
    job_mgr:     &Arc<RwLock<JobManager>>,
    seen_shares: &mut HashSet<(String, u32)>,
    share_diff:  f64,
) -> Result<(SubmitOutcome, u32), ShareError> {
    // ── 1. Format: extranonce2 must be exactly 8 hex chars (4 bytes) ──────
    if en2_hex.len() != 8 {
        return Err(ShareError::BadFormat(
            format!("extranonce2 must be 8 hex chars, got {}", en2_hex.len())
        ));
    }
    let mut en2_bytes = [0u8; 4];
    faster_hex::hex_decode(en2_hex.as_bytes(), &mut en2_bytes)
        .map_err(|e| ShareError::BadFormat(format!("extranonce2 decode: {e}")))?;
    let extranonce2 = u32::from_be_bytes(en2_bytes);

    // Full 64-bit nonce = extranonce1 (high 32) || extranonce2 (low 32)
    // The high 32 bits are ALWAYS our assigned extranonce1 — miners cannot
    // escape this assignment since we construct the nonce here.
    let nonce: u64 = ((extranonce1 as u64) << 32) | (extranonce2 as u64);

    // ── 2. Stale: look up the job ──────────────────────────────────
    let job = job_mgr
        .read()
        .await
        .get(job_id)
        .ok_or_else(|| ShareError::Stale(format!("unknown/stale job {job_id}")))?;

    // ── 3. Duplicate: same (job_id, extranonce2) already seen by this miner ──
    if !seen_shares.insert((job_id.to_owned(), extranonce2)) {
        return Err(ShareError::Duplicate);
    }

    // ── 4. Share difficulty check ─────────────────────────────────
    //
    // For KHeavyHash (pre-Genome-PoW) we can compute the actual hash locally.
    // For Genome PoW the hash requires the packed genome dataset (739 MB);
    // the bridge doesn’t hold it so we rely on the node’s full validation
    // and accept that misbehaving miners will be limited by VarDiff.
    if !job.genome_active {
        let header: Header = (&job.template.header).into();
        let kh_state = KHeavyState::new(&header);
        let (_, pow_hash) = kh_state.check_pow(nonce);
        let share_target  = MAX_DIFFICULTY_TARGET_F64 / share_diff;
        let hash_f64      = pow_hash.as_f64();
        if hash_f64 > share_target {
            return Err(ShareError::LowDifficulty { hash: hash_f64, target: share_target });
        }
    }

    // ── 5. Submit to node for full Genome PoW / block validation ────────
    let block = job.build_block(nonce);
    let resp  = rpc
        .submit_block(block, false)
        .await
        .map_err(|e| ShareError::BadFormat(format!("submit_block RPC: {e}")))?;

    let bits = job.template.header.bits;
    if matches!(resp.report, SubmitBlockReport::Success) {
        // Approximate DAA score: template parent score + 1
        let daa_score = job.template.header.daa_score.saturating_add(1);
        Ok((SubmitOutcome::Block { daa_score }, bits))
    } else {
        Ok((SubmitOutcome::Share, bits))
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

async fn write_line<T: serde::Serialize>(writer: &mut OwnedWriteHalf, msg: &T) -> Result<()> {
    let mut line = serde_json::to_string(msg).context("serialize")?;
    line.push('\n');
    writer.write_all(line.as_bytes()).await.context("write")?;
    Ok(())
}

async fn send_notify(writer: &mut OwnedWriteHalf, job: &Job, clean: bool) -> Result<()> {
    let notif = StratumNotification::notify(
        &job.id, &job.pre_pow_hash_hex, &job.bits_hex,
        &job.epoch_seed_hex, &job.timestamp_hex, clean,
    );
    write_line(writer, &notif).await
}

fn str_param<'a>(params: &'a [Value], idx: usize, name: &str) -> Result<&'a str> {
    params
        .get(idx)
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow!("missing param '{name}' at index {idx}"))
}
