use anyhow::{Context, Result};
use clap::{Arg, Command};
use genetics_l2_core::{merkle_root_hex, now_secs, Payout, SettlementPayload};
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

    log::info!("Genetics-L2 Settlement started");
    log::info!("  coordinator: {coordinator}");
    log::info!("  node:        {node_addr}");
    log::info!("  dry_run:     {dry_run}");

    let privkey_hex: Option<String> = m.get_one::<String>("private-key").cloned();
    let fee_sompi: u64 = m.get_one::<String>("fee-sompi")
        .and_then(|s| s.parse().ok())
        .unwrap_or(xenom_anchor_client::DEFAULT_FEE_PER_INPUT);

    if !dry_run && privkey_hex.is_none() {
        anyhow::bail!("--submit requires --private-key <HEX>");
    }

    let keypair: Option<secp256k1::Keypair> = privkey_hex
        .as_deref()
        .map(xenom_anchor_client::keypair_from_hex)
        .transpose()
        .context("--private-key")?;

    if let Some(ref kp) = keypair {
        log::info!("  funding: {}", xenom_anchor_client::address_from_keypair(kp));
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

    let http = reqwest::Client::new();

    loop {
        match settle_validated_jobs(&http, &coordinator, rpc.as_ref(), keypair.as_ref(), fee_sompi, dry_run).await {
            Ok(n) if n > 0 => log::info!("Settled {n} job(s)"),
            Ok(_)          => {}
            Err(e)         => log::warn!("Settlement cycle error: {e:#}"),
        }
        sleep(Duration::from_millis(poll_ms)).await;
    }
}

// ── Settlement logic ──────────────────────────────────────────────────────────

async fn settle_validated_jobs(
    http:        &reqwest::Client,
    coordinator: &str,
    rpc:         Option<&std::sync::Arc<kaspa_grpc_client::GrpcClient>>,
    keypair:     Option<&secp256k1::Keypair>,
    fee_sompi:   u64,
    dry_run:     bool,
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

        if valid_results.is_empty() {
            log::debug!("Job {job_id}: no valid results yet, skipping");
            continue;
        }

        // ── Build results_root ────────────────────────────────────────────────
        let result_root_hashes: Vec<String> = valid_results.iter()
            .filter_map(|r| r["result_root"].as_str().map(str::to_owned))
            .collect();
        let results_root = merkle_root_hex(&result_root_hashes);

        // ── Find winner (highest score) ───────────────────────────────────────
        let winner = valid_results.iter()
            .max_by(|a, b| {
                let sa = a["score"].as_f64().unwrap_or(0.0);
                let sb = b["score"].as_f64().unwrap_or(0.0);
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();

        let Some(winner) = winner else { continue };
        let winner_pubkey          = winner["worker_pubkey"].as_str().unwrap_or("").to_owned();
        let best_score             = winner["score"].as_f64().unwrap_or(0.0);
        let notebook_or_repo_hash  = winner["notebook_or_repo_hash"].as_str().map(str::to_owned);
        let container_hash         = winner["container_hash"].as_str().map(str::to_owned);
        let weights_hash           = winner["weights_hash"].as_str().map(str::to_owned);
        let submission_bundle_hash = winner["submission_bundle_hash"].as_str().map(str::to_owned);

        log::info!(
            "Settling job {job_id}: winner={} score={best_score:.2} results_root={results_root}",
            &winner_pubkey[..12.min(winner_pubkey.len())]
        );

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
        let txid = if dry_run {
            log::info!("  dry-run: skipping chain submission");
            None
        } else if let (Some(rpc_client), Some(kp)) = (rpc, keypair) {
            match xenom_anchor_client::submit_anchor(rpc_client, kp, &payload_bytes, fee_sompi).await {
                Ok(id)  => { log::info!("  anchored txid={id}"); Some(id) }
                Err(e)  => { log::warn!("  anchor failed: {e:#}"); None }
            }
        } else {
            log::warn!("  no RPC/keypair — skipping chain submission");
            None
        };

        // ── Register payout with coordinator ─────────────────────────────────
        let payout = Payout {
            payout_id:     Uuid::new_v4().to_string(),
            job_id:        job_id.clone(),
            worker_pubkey: winner_pubkey.clone(),
            amount_sompi:  reward_sompi,
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
                payout.payout_id, reward_sompi, &winner_pubkey[..12.min(winner_pubkey.len())]);
        } else {
            let s = payout_resp.status();
            let b = payout_resp.text().await.unwrap_or_default();
            log::warn!("  payout registration failed: {s} {b}");
        }

        count += 1;
    }

    Ok(count)
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
        .arg(Arg::new("private-key")
            .short('k').long("private-key").value_name("HEX")
            .help("secp256k1 private key (64 hex chars) for the funding/signing address. Required with --submit."))
        .arg(Arg::new("fee-sompi")
            .long("fee-sompi").value_name("N")
            .default_value("2000")
            .help("Relay fee per input in sompi (default: 2000)"))
}
