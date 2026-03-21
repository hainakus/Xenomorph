use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{anyhow, bail};
use kaspa_core::{info, warn};
use kaspa_hashes::Hash;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    net::TcpStream,
    sync::mpsc,
    time::sleep,
};

use crate::tui::DashStats;

// ── Public types ──────────────────────────────────────────────────────────────

/// Work unit received from the stratum bridge via `mining.notify`.
pub struct StratumJob {
    pub job_id:       String,
    pub pre_pow_hash: Hash,
    pub bits:         u32,
    pub epoch_seed:   Hash,
    pub timestamp:    u64,
    pub clean_jobs:   bool,
    pub extranonce1:  u32,
    /// Optional L2 compute task piggybacked on this PoW job (param[6]).
    pub l2_job:       Option<serde_json::Value>,
    /// Template DAA score from param[7].  Used with --genome-activation-daa-score to
    /// select the correct PoW algorithm regardless of epoch_seed value.
    pub daa_score:    u64,
    /// Calculated target based on difficulty or bits.
    pub target:       kaspa_math::Uint256,
}

/// Solution to be submitted to the stratum bridge via `mining.submit`.
pub struct StratumSolution {
    pub job_id:      String,
    pub extranonce2: u32,
    pub ntime:       u64,
    pub nonce:       u64,
}

// ── Client ────────────────────────────────────────────────────────────────────

pub struct StratumClient {
    /// `host:port` (scheme stripped)
    pub addr:       String,
    pub worker:     String,
    pub password:   String,
    pub difficulty: Arc<Mutex<f64>>,
}

impl StratumClient {
    /// Build from a URL like `stratum+tcp://host:1444` or plain `host:1444`.
    pub fn new(url: &str, worker: &str, password: &str) -> Self {
        let addr = url
            .trim_start_matches("stratum+tcp://")
            .trim_start_matches("stratum://")
            .to_owned();
        Self {
            addr,
            worker: worker.to_owned(),
            password: password.to_owned(),
            difficulty: Arc::new(Mutex::new(1.0)),
        }
    }

    /// Spawn-safe run loop — reconnects automatically on failure.
    ///
    /// * Sends parsed jobs via `job_tx`
    /// * Reads solutions to submit from `sol_rx`
    pub async fn run(
        self,
        job_tx:  mpsc::Sender<StratumJob>,
        sol_rx:  mpsc::Receiver<StratumSolution>,
        dash:    Arc<std::sync::Mutex<DashStats>>,
    ) {
        let mut sol_rx = sol_rx;
        loop {
            match self.connect_once(&job_tx, &mut sol_rx, &dash).await {
                Ok(()) => info!("Stratum: connection closed"),
                Err(e) => {
                    warn!("Stratum: {e} — reconnecting in 5s");
                    dash.lock().unwrap().push_log(format!("Stratum disconnected: {e}"));
                }
            }
            sleep(Duration::from_secs(5)).await;
        }
    }

    async fn connect_once(
        &self,
        job_tx:  &mpsc::Sender<StratumJob>,
        sol_rx:  &mut mpsc::Receiver<StratumSolution>,
        dash:    &Arc<std::sync::Mutex<DashStats>>,
    ) -> anyhow::Result<()> {
        info!("Stratum: connecting to {}", self.addr);
        let stream = TcpStream::connect(&self.addr).await?;
        let (reader, mut writer) = stream.into_split();
        let mut lines = BufReader::new(reader).lines();

        let mut extranonce1: u32 = 0;
        let mut msg_id: u64 = 2; // subscribe=1, authorize=2 use hardcoded ids; loop starts at 3

        macro_rules! send_json {
            ($v:expr) => {{
                let mut s = serde_json::to_string(&$v)?;
                s.push('\n');
                writer.write_all(s.as_bytes()).await?;
            }};
        }

        send_json!(serde_json::json!({
            "id": 1, "method": "mining.subscribe", "params": ["genome-miner/1.0", null]
        }));
        send_json!(serde_json::json!({
            "id": 2, "method": "mining.authorize", "params": [&self.worker, &self.password]
        }));
        info!("Stratum: subscribe + authorize sent");

        loop {
            tokio::select! {
                line = lines.next_line() => {
                    let line = match line? {
                        Some(l) if !l.trim().is_empty() => l,
                        Some(_) => continue,
                        None    => bail!("Stratum TCP connection closed"),
                    };

                    let msg: serde_json::Value = serde_json::from_str(&line)
                        .map_err(|e| anyhow!("JSON parse: {e}"))?;

                    if let Some(method) = msg["method"].as_str() {
                        // ── Server notifications ────────────────────────
                        match method {
                            "mining.notify" => {
                                info!("Stratum: received mining.notify: {}", msg);
                                let en1 = extranonce1;
                                let diff = *self.difficulty.lock().unwrap();
                                if let Some(job) = parse_notify(&msg, en1, diff) {
                                    let l2_info = if job.l2_job.is_some() { " L2=yes" } else { "" };
                                    let log = format!(
                                        "Stratum job={} bits={:#010x} target={:x} clean={}{}",
                                        job.job_id, job.bits, job.target, job.clean_jobs, l2_info
                                    );
                                    info!("{log}");
                                    {
                                        let mut s = dash.lock().unwrap();
                                        s.bits         = job.bits;
                                        s.genome_active = job.epoch_seed != Hash::default();
                                        s.connected    = true;
                                        s.push_log(log);
                                    }
                                    if job_tx.send(job).await.is_err() {
                                        break;
                                    }
                                } else {
                                    warn!("Stratum: failed to parse mining.notify");
                                }
                            }
                            "mining.set_difficulty" => {
                                if let Some(d) = msg["params"][0].as_f64() {
                                    *self.difficulty.lock().unwrap() = d;
                                    let target = calculate_target(d);
                                    info!("Stratum: pool difficulty={d} (target: {target:x})");
                                }
                            }
                            other => {
                                info!("Stratum: notification '{other}' (ignored)");
                            }
                        }
                    } else {
                        // ── Responses to our requests ────────────────────
                        let resp_id = msg["id"].as_u64().unwrap_or(0);
                        match resp_id {
                            1 => {
                                // subscribe response: [[[sub_type,sub_id],...], extranonce1_hex, en2_size]
                                if let Some(arr) = msg["result"].as_array() {
                                    if arr.len() >= 2 {
                                        let en1_hex = arr[1].as_str().unwrap_or("00000000");
                                        if let Some(en1) = parse_extranonce1(en1_hex) {
                                            extranonce1 = en1;
                                            info!("Stratum: subscribed extranonce1={en1_hex}");
                                            dash.lock().unwrap().push_log(
                                                format!("Stratum connected {} extranonce1={en1_hex}", self.addr)
                                            );
                                        }
                                    }
                                }
                            }
                            2 => {
                                let ok = msg["result"].as_bool().unwrap_or(false);
                                if ok {
                                    info!("Stratum: authorized as {}", self.worker);
                                } else {
                                    warn!("Stratum: auth failed: {:?}", msg["error"]);
                                }
                            }
                            _ => {
                                // submit response
                                let ok = msg["result"].as_bool().unwrap_or(false);
                                if ok {
                                    info!("Stratum: share ACCEPTED");
                                    let mut s = dash.lock().unwrap();
                                    s.accepted += 1;
                                    s.push_log("Stratum: share accepted".to_owned());
                                } else {
                                    warn!("Stratum: share REJECTED {:?}", msg["error"]);
                                    let mut s = dash.lock().unwrap();
                                    s.rejected += 1;
                                    s.push_log(format!("Stratum: share rejected {:?}", msg["error"]));
                                }
                            }
                        }
                    }
                }

                sol = sol_rx.recv() => {
                    let sol = match sol {
                        Some(s) => s,
                        None    => break,   // all GPU tasks exited
                    };
                    msg_id += 1;
                    let en2_hex = format!("{:08x}", sol.extranonce2);
                    let ntime_hex = format!("{:016x}", sol.ntime);
                    let nonce_hex = format!("{:016x}", sol.nonce);
                    let submit = serde_json::json!({
                        "id": msg_id,
                        "method": "mining.submit",
                        "params": [&self.worker, sol.job_id, en2_hex, ntime_hex, nonce_hex]
                    });
                    let mut line = serde_json::to_string(&submit)?;
                    line.push('\n');
                    writer.write_all(line.as_bytes()).await?;
                    info!("Stratum: submitted en2={en2_hex} ntime={ntime_hex} nonce={nonce_hex} for job={}", sol.job_id);
                }
            }
        }

        Ok(())
    }
}

// ── Parse helpers ─────────────────────────────────────────────────────────────

fn parse_notify(msg: &serde_json::Value, extranonce1: u32, difficulty: f64) -> Option<StratumJob> {
    let params = msg["params"].as_array()?;
    if params.len() < 5 {
        return None;
    }
    let job_id    = params[0].as_str()?.to_owned();
    let pph_hex   = params[1].as_str()?;
    let bits_hex  = params[2].as_str()?;
    let eseed_hex = params[3].as_str()?;
    let ts_hex    = params[4].as_str()?;
    let clean     = params.get(5).and_then(|v| v.as_bool()).unwrap_or(false);
    let l2_job    = params.get(6).and_then(|v| {
        if v.is_null() { None } else { Some(v.clone()) }
    });

    let pre_pow_hash = hex_to_hash32(pph_hex)?;
    let bits         = u32::from_str_radix(bits_hex, 16).ok()?;
    let epoch_seed   = hex_to_hash32(eseed_hex)?;
    let timestamp    = u64::from_str_radix(ts_hex, 16).ok()?;
    let daa_score    = params.get(7)
        .and_then(|v| v.as_str())
        .and_then(|s| u64::from_str_radix(s, 16).ok())
        .unwrap_or(0);

    let target = calculate_target(difficulty);

    Some(StratumJob {
        job_id,
        pre_pow_hash,
        bits,
        epoch_seed,
        timestamp,
        clean_jobs: clean,
        extranonce1,
        l2_job,
        daa_score,
        target,
    })
}

fn calculate_target(difficulty: f64) -> kaspa_math::Uint256 {
    if difficulty <= 0.0 {
        return kaspa_math::Uint256::MAX;
    }

    // Kaspa Stratum difficulty 1.0 = target 0x0000ffff00000000000000000000000000000000000000000000000000000000
    // which is (2^16 - 1) * 2^208.
    // target = (2^16 - 1) * 2^208 / difficulty.

    let base_mantissa = 0xffffu128;
    let base_exponent = 208i16;

    // Use f64 for the division to handle the difficulty scaling
    let target_f64 = (base_mantissa as f64) * (2.0f64.powi(base_exponent as i32)) / difficulty;

    // Convert back to Uint256
    if target_f64 >= 2.0f64.powi(256) {
        return kaspa_math::Uint256::MAX;
    }
    if target_f64 < 1.0 {
        return kaspa_math::Uint256::ZERO;
    }

    // Extract mantissa and exponent from the resulting f64
    let bits = target_f64.to_bits();
    let exponent = (((bits >> 52) & 0x7FF) as i16) - 1023;
    let mantissa = (bits & 0xF_FFFF_FFFF_FFFF) | 0x10_0000_0000_0000;

    let mut res = kaspa_math::Uint256::from_u128(mantissa as u128);
    let shift = exponent - 52;
    if shift >= 0 {
        res = res.wrapping_shl(shift as u32);
    } else {
        res = res.overflowing_shr((-shift) as u32).0;
    }
    res
}

fn hex_to_hash32(hex: &str) -> Option<Hash> {
    if hex.len() != 64 {
        return None;
    }
    let mut bytes = [0u8; 32];
    faster_hex::hex_decode(hex.as_bytes(), &mut bytes).ok()?;
    Some(Hash::from_bytes(bytes))
}

fn parse_extranonce1(hex: &str) -> Option<u32> {
    if hex.len() != 8 {
        return None;
    }
    let mut bytes = [0u8; 4];
    faster_hex::hex_decode(hex.as_bytes(), &mut bytes).ok()?;
    Some(u32::from_be_bytes(bytes))
}
