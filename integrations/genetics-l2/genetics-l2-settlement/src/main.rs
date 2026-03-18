use anyhow::{Context, Result};
use clap::{Arg, Command};
use genetics_l2_core::{merkle_root_hex, now_secs, Payout, SettlementPayload};
use kaspa_addresses::Prefix;
use serde_json::Value;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

// ── Settlement daemon ─────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m           = cli().get_matches();
    let coordinator = m.get_one::<String>("coordinator").unwrap().clone();
    let node_addr   = m.get_one::<String>("node").unwrap().clone();
    let poll_ms: u64 = m.get_one::<String>("poll-ms")
        .and_then(|s| s.parse().ok()).unwrap_or(15_000);
    let dry_run     = !m.get_flag("submit");
    let network_prefix = if m.get_flag("devnet") {
        Prefix::Devnet
    } else if m.get_flag("testnet") {
        Prefix::Testnet
    } else {
        Prefix::Mainnet
    };

    // Auto-default to EVM anchoring for devnet/testnet/simnet (can be overridden with --evm-node)
    let evm_node: Option<String> = m.get_one::<String>("evm-node").cloned()
        .or_else(|| {
            match network_prefix {
                Prefix::Devnet | Prefix::Testnet | Prefix::Simnet => Some("http://127.0.0.1:8545".to_string()),
                Prefix::Mainnet => None,
            }
        });

    log::info!("Genetics-L2 Settlement started");
    log::info!("  coordinator: {coordinator}");
    log::info!("  node:        {node_addr}");
    log::info!("  network:     {network_prefix:?}");
    log::info!("  dry_run:     {dry_run}");
    if let Some(ref e) = evm_node {
        log::info!("  evm-node:    {e} (anchor target)");
    } else {
        log::info!("  evm-node:    none (mainnet: will use coinbase extra_data)");
    }

    let privkey_hex: Option<String> = load_privkey_opt("SETTLEMENT_PRIVKEY", m.get_one::<String>("key-file").map(|s| s.as_str()));
    let fee_sompi: u64 = m.get_one::<String>("fee-sompi")
        .and_then(|s| s.parse().ok())
        .unwrap_or(xenom_anchor_client::DEFAULT_FEE_PER_INPUT);

    if !dry_run && privkey_hex.is_none() {
        anyhow::bail!("--submit requires a private key: set $SETTLEMENT_PRIVKEY or use --key-file <PATH>");
    }

    let keypair: Option<secp256k1::Keypair> = privkey_hex
        .as_deref()
        .map(xenom_anchor_client::keypair_from_hex)
        .transpose()
        .context("invalid private key (expected 64 hex chars)")?;

    if let Some(ref kp) = keypair {
        log::info!("  funding: {}",
            xenom_anchor_client::address_from_keypair(kp, network_prefix));
    }

    // Set up shared RPC client for settlement daemon
    let rpc: Option<std::sync::Arc<kaspa_grpc_client::GrpcClient>> = if !dry_run {
        let url = if node_addr.starts_with("grpc://") {
            node_addr.clone()
        } else {
            format!("grpc://{node_addr}")
        };
        let client = kaspa_grpc_client::GrpcClient::connect(url)
            .await
            .context("cannot connect to Xenom node")?;
        log::info!("  connected to node: {node_addr}");
        Some(std::sync::Arc::new(client))
    } else {
        None
    };

    let quorum: usize = m.get_one::<String>("quorum")
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    let score_tolerance: f64 = m.get_one::<String>("score-tolerance")
        .and_then(|s| s.parse().ok()).unwrap_or(0.05);

    let http = reqwest::Client::new();

    loop {
        match settle_validated_jobs(&http, &coordinator, rpc.as_ref(), keypair.as_ref(), fee_sompi, dry_run, network_prefix, evm_node.as_deref(), quorum, score_tolerance).await {
            Ok(n) if n > 0 => log::info!("Settled {n} job(s)"),
            Ok(_)          => {}
            Err(e)         => log::warn!("Settlement cycle error: {e:#}"),
        }
        sleep(Duration::from_millis(poll_ms)).await;
    }
}

/// Load private key from env var or key file. Returns None if neither provided.
/// Never reads from CLI args to avoid exposure in `ps aux`.
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

// ── Settlement logic ──────────────────────────────────────────────────────────

async fn settle_validated_jobs(
    http:            &reqwest::Client,
    coordinator:     &str,
    rpc:             Option<&std::sync::Arc<kaspa_grpc_client::GrpcClient>>,
    keypair:         Option<&secp256k1::Keypair>,
    fee_sompi:       u64,
    dry_run:         bool,
    prefix:          Prefix,
    evm_node:        Option<&str>,
    quorum:          usize,
    score_tolerance: f64,
) -> Result<usize> {
    // Fetch validated (not yet settled) jobs
    let resp = http
        .get(format!("{coordinator}/jobs?status=validated&limit=10"))
        .send()
        .await
        .context("GET /jobs?status=validated")?;

    let body: Value = resp.json().await.context("parse jobs")?;
    let jobs = body["jobs"].as_array().cloned().unwrap_or_default();

    let mut count = 0;
    for job_val in &jobs {
        let job_id       = job_val["job_id"].as_str().unwrap_or("").to_owned();
        let source       = job_val["source"].as_str().unwrap_or("").to_owned();
        let algorithm    = job_val["algorithm"].as_str().unwrap_or("").to_owned();
        let dataset_root = job_val["dataset_root"].as_str().unwrap_or("").to_owned();
        let reward_sompi = job_val["reward_sompi"].as_i64().unwrap_or(0) as u64;

        if job_id.is_empty() { continue; }

        // Fetch all valid results for this job
        let results_resp = http
            .get(format!("{coordinator}/results/{job_id}"))
            .send()
            .await
            .context("GET /results")?;
        let results_body: Value = results_resp.json().await.context("parse results")?;
        let results = results_body["results"].as_array().cloned().unwrap_or_default();

        let valid_results: Vec<&Value> = results.iter()
            .filter(|r| r["verdict"].as_str() == Some("valid"))
            .collect();

        if valid_results.len() < quorum {
            log::debug!("Job {job_id}: {}/{quorum} valid results — waiting for quorum",
                valid_results.len());
            continue;
        }

        // ── Build results_root ────────────────────────────────────────────────
        let result_root_hashes: Vec<String> = valid_results.iter()
            .filter_map(|r| r["result_root"].as_str().map(str::to_owned))
            .collect();
        let results_root = merkle_root_hex(&result_root_hashes);

        // ── Quorum: winner = result closest to median score ───────────────────
        let mut scores: Vec<f64> = valid_results.iter()
            .map(|r| r["score"].as_f64().unwrap_or(0.0))
            .collect();
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if scores.len() % 2 == 0 {
            (scores[scores.len() / 2 - 1] + scores[scores.len() / 2]) / 2.0
        } else {
            scores[scores.len() / 2]
        };

        // Consensus cluster: results within score_tolerance of median
        let consensus: Vec<&Value> = valid_results.iter()
            .filter(|r| (r["score"].as_f64().unwrap_or(0.0) - median).abs() <= score_tolerance)
            .copied()
            .collect();

        if consensus.is_empty() {
            log::debug!("Job {job_id}: no consensus cluster (median={median:.4}), skipping");
            continue;
        }

        // Winner = result whose score is closest to the median
        let winner = consensus.iter()
            .min_by(|a, b| {
                let da = (a["score"].as_f64().unwrap_or(0.0) - median).abs();
                let db = (b["score"].as_f64().unwrap_or(0.0) - median).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let Some(winner) = winner else { continue };
        let best_score = winner["score"].as_f64().unwrap_or(0.0);
        log::info!(
            "Settling job {job_id}: winner={} score={best_score:.4} median={median:.4} quorum={}/{quorum} results_root={results_root}",
            winner["worker_pubkey"].as_str().unwrap_or("").get(..12).unwrap_or(""),
            valid_results.len()
        );
        let winner_pubkey          = winner["worker_pubkey"].as_str().unwrap_or("").to_owned();
        let notebook_or_repo_hash  = winner["notebook_or_repo_hash"].as_str().map(str::to_owned);
        let container_hash         = winner["container_hash"].as_str().map(str::to_owned);
        let weights_hash           = winner["weights_hash"].as_str().map(str::to_owned);
        let submission_bundle_hash = winner["submission_bundle_hash"].as_str().map(str::to_owned);

        // ── Build SettlementPayload ───────────────────────────────────────────
        let payload = SettlementPayload {
            app:                    SettlementPayload::APP_ID.to_owned(),
            v:                      1,
            job_id:                 job_id.clone(),
            source:                 source.clone(),
            algorithm:              algorithm.clone(),
            dataset_root:           dataset_root.clone(),
            results_root:           results_root.clone(),
            best_score,
            winner_pubkey:          winner_pubkey.clone(),
            notebook_or_repo_hash,
            container_hash,
            weights_hash,
            submission_bundle_hash,
            settled_at:             now_secs(),
        };
        let payload_bytes = payload.to_payload_bytes();
        log::info!("  settlement payload: {} bytes", payload_bytes.len());

        // ── Anchor on Xenom chain ─────────────────────────────────────────────
        // On devnet/testnet: anchor via xenom_anchor RPC on the EVM L2 node.
        // On mainnet: the bridge includes the settlement hash in coinbase extra_data.
        let txid = if dry_run {
            log::info!("  dry-run: skipping chain submission");
            None
        } else if (prefix == Prefix::Devnet || prefix == Prefix::Testnet) && evm_node.is_some() {
            match evm_anchor(http, evm_node.unwrap(), &payload_bytes).await {
                Ok(id) => { log::info!("  EVM anchor id={id}"); Some(id) }
                Err(e) => { log::warn!("  EVM anchor failed: {e:#}"); None }
            }
        } else if prefix == Prefix::Devnet || prefix == Prefix::Testnet {
            log::info!("  devnet/testnet: no --evm-node set, skipping anchor");
            None
        } else if let (Some(rpc_client), Some(kp)) = (rpc, keypair) {
            match xenom_anchor_client::submit_anchor(rpc_client, kp, &payload_bytes, fee_sompi, prefix).await {
                Ok(id)  => { log::info!("  anchored txid={id}"); Some(id) }
                Err(e)  => { log::warn!("  anchor failed: {e:#}"); None }
            }
        } else {
            log::warn!("  no RPC/keypair — skipping chain submission");
            None
        };

        // ── Score-based reward: amount = reward_sompi × score (0.0..1.0) ─────
        // Minimum floor of 1_000 sompi for any valid non-zero score.
        const MIN_SOMPI: u64 = 1_000;
        let scored_sompi: u64 = if best_score > 0.0 {
            let raw = (reward_sompi as f64 * best_score.clamp(0.0, 1.0)).round() as u64;
            raw.max(MIN_SOMPI)
        } else {
            0
        };
        log::info!("  score-based reward: {reward_sompi} × {best_score:.4} = {scored_sompi} sompi");

        // ── Register payout with coordinator ─────────────────────────────────
        let payout = Payout {
            payout_id:     Uuid::new_v4().to_string(),
            job_id:        job_id.clone(),
            worker_pubkey: winner_pubkey.clone(),
            amount_sompi:  scored_sompi,
            txid:          txid.clone(),
            paid_at:       txid.as_ref().map(|_| now_secs()),
        };

        let payout_resp = http
            .post(format!("{coordinator}/payouts"))
            .json(&payout)
            .send()
            .await
            .context("POST /payouts")?;

        if payout_resp.status().is_success() {
            log::info!("  payout {} registered: {} sompi → {}",
                payout.payout_id, scored_sompi, &winner_pubkey[..12.min(winner_pubkey.len())]);
        } else {
            let s = payout_resp.status();
            let b = payout_resp.text().await.unwrap_or_default();
            log::warn!("  payout registration failed: {s} {b}");
        }

        count += 1;
    }

    Ok(count)
}

// ── EVM anchor helper ────────────────────────────────────────────────────────

async fn evm_anchor(http: &reqwest::Client, evm_node: &str, data: &[u8]) -> Result<String> {
    let url = if evm_node.starts_with("http") {
        evm_node.to_owned()
    } else {
        format!("http://{evm_node}")
    };
    let payload_hex = format!("0x{}", hex::encode(data));
    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "xenom_anchor",
        "params": [payload_hex],
        "id": 1
    });
    let resp: serde_json::Value = http
        .post(&url)
        .json(&body)
        .send()
        .await
        .context("xenom_anchor HTTP POST")?
        .json()
        .await
        .context("xenom_anchor parse response")?;
    if let Some(err) = resp.get("error") {
        anyhow::bail!("xenom_anchor RPC error: {err}");
    }
    resp["result"]
        .as_str()
        .map(str::to_owned)
        .ok_or_else(|| anyhow::anyhow!("xenom_anchor: missing result field"))
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genetics-l2-settlement")
        .about("Genetics L2 settlement — creates results_root and anchors validated jobs on Xenom")
        .arg(Arg::new("coordinator")
            .short('c').long("coordinator").value_name("URL")
            .default_value("http://localhost:8091")
            .help("genetics-l2-coordinator base URL"))
        .arg(Arg::new("node")
            .short('n').long("node").value_name("ADDR")
            .default_value("grpc://localhost:36669")
            .help("Xenom node gRPC address"))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("15000")
            .help("Poll interval in milliseconds"))
        .arg(Arg::new("submit")
            .long("submit")
            .action(clap::ArgAction::SetTrue)
            .help("Anchor settlement on-chain (default: dry-run)"))
        .arg(Arg::new("key-file")
            .short('k').long("key-file").value_name("PATH")
            .help("Path to file containing the secp256k1 private key (64 hex chars). Alternatively set $SETTLEMENT_PRIVKEY. Required with --submit."))
        .arg(Arg::new("fee-sompi")
            .long("fee-sompi").value_name("N")
            .default_value("2000")
            .help("Relay fee per input in sompi (default: 2000)"))
        .arg(Arg::new("devnet")
            .long("devnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use devnet address prefix (xenomdev:). Auto-enables EVM anchoring at http://127.0.0.1:8545"))
        .arg(Arg::new("testnet")
            .long("testnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use testnet address prefix (xenomtest:). Auto-enables EVM anchoring at http://127.0.0.1:8545"))
        .arg(Arg::new("evm-node")
            .long("evm-node").value_name("URL")
            .help("Override default EVM L2 JSON-RPC URL. Defaults to http://127.0.0.1:8545 for devnet/testnet, none for mainnet (uses coinbase extra_data)"))
        .arg(Arg::new("quorum")
            .long("quorum").value_name("N")
            .default_value("2")
            .help("Minimum number of valid results required before settling a job"))
        .arg(Arg::new("score-tolerance")
            .long("score-tolerance").value_name("F")
            .default_value("0.05")
            .help("Max score distance from median to be included in the consensus cluster"))
}
