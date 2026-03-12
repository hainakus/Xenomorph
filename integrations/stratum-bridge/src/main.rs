mod accounting;
mod api;
mod db;
mod job;
mod payments;
mod proto;
mod stratum;
mod vardiff;

use std::{net::SocketAddr, path::PathBuf, str::FromStr, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use clap::{Arg, Command};
use kaspa_addresses::{Address, Prefix, Version};
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_rpc_core::{api::rpc::RpcApi, model::message::GetBlockTemplateRequest};
use tokio::{
    sync::{watch, Mutex, RwLock},
    time::sleep,
};

use crate::{
    accounting::Accounting,
    api::ApiState,
    db::Db,
    job::JobManager,
    payments::{execute_payout, PaymentConfig, RetryablePayoutError},
    vardiff::VarDiffConfig,
};

fn cli() -> Command {
    Command::new("xenom-stratum-bridge")
        .about("Stratum bridge for Xenom Genome PoW mining\n\
                Connects to a Xenom node via gRPC and exposes a Stratum v1 server for miners.\n\
                \n\
                Stratum mining.notify params (Xenom extension):\n\
                  [job_id, pre_pow_hash, bits, epoch_seed, timestamp_ms, clean_jobs]\n\
                \n\
                mining.submit params:\n\
                  [worker, job_id, extranonce2_hex]  (extranonce2 = 4 bytes / 8 hex chars)\n\
                  Full nonce = extranonce1 (high 32 bits) || extranonce2 (low 32 bits)")
        // ── connectivity ──────────────────────────────────────────────────────
        .arg(Arg::new("rpcserver")
            .long("rpcserver").short('s').value_name("HOST:PORT")
            .default_value("localhost:36669")
            .help("Xenom node gRPC endpoint"))
        .arg(Arg::new("listen")
            .long("listen").short('l').value_name("ADDR:PORT")
            .default_value("0.0.0.0:1444")
            .help("Stratum TCP listen address"))
        .arg(Arg::new("mining-address")
            .long("mining-address").short('a').value_name("ADDRESS").required(true)
            .help("Xenom pool reward address for coinbase output"))
        .arg(Arg::new("poll-interval-ms")
            .long("poll-interval-ms").value_name("MS")
            .default_value("200").value_parser(clap::value_parser!(u64))
            .help("Block template poll interval in milliseconds"))
        // ── VarDiff ───────────────────────────────────────────────────────────
        .arg(Arg::new("vardiff-initial")
            .long("vardiff-initial").value_name("FLOAT")
            .default_value("1").value_parser(clap::value_parser!(f64))
            .help("Starting share difficulty per miner"))
        .arg(Arg::new("vardiff-min")
            .long("vardiff-min").value_name("FLOAT")
            .default_value("0.1").value_parser(clap::value_parser!(f64))
            .help("Minimum share difficulty"))
        .arg(Arg::new("vardiff-max")
            .long("vardiff-max").value_name("FLOAT")
            .default_value("1000000").value_parser(clap::value_parser!(f64))
            .help("Maximum share difficulty"))
        .arg(Arg::new("vardiff-target-spm")
            .long("vardiff-target-spm").value_name("FLOAT")
            .default_value("20").value_parser(clap::value_parser!(f64))
            .help("Target shares per minute per miner (default: 20 = 1 share/3 s)"))
        .arg(Arg::new("vardiff-retarget-secs")
            .long("vardiff-retarget-secs").value_name("N")
            .default_value("60").value_parser(clap::value_parser!(u64))
            .help("VarDiff retarget interval in seconds"))
        // ── PPLNS accounting ──────────────────────────────────────────────────
        .arg(Arg::new("pplns-window")
            .long("pplns-window").value_name("N")
            .default_value("10000").value_parser(clap::value_parser!(usize))
            .help("PPLNS share window size (number of recent shares used for payout calculation)"))
        .arg(Arg::new("payout-file")
            .long("payout-file").value_name("PATH")
            .help("Write pending PPLNS payout records to this JSON file whenever a block is found"))
        .arg(Arg::new("stats-interval-secs")
            .long("stats-interval-secs").value_name("N")
            .default_value("300").value_parser(clap::value_parser!(u64))
            .help("Log pool stats every N seconds (0 = disable)"))
        // ── Auto-payout ───────────────────────────────────────────────────────
        .arg(Arg::new("pool-private-key")
            .long("pool-private-key").value_name("HEX")
            .help("Pool wallet private key (hex, 32 bytes). Enables automatic payouts after confirmation. Node must be started with --utxoindex."))
        .arg(Arg::new("confirm-depth")
            .long("confirm-depth").value_name("N")
            .default_value("1000").value_parser(clap::value_parser!(u64))
            .help("DAA-score depth required before paying out a mined block (default: 1000)"))
        .arg(Arg::new("min-payout-sompi")
            .long("min-payout-sompi").value_name("N")
            .default_value("100000").value_parser(clap::value_parser!(u64))
            .help("Minimum per-miner payout in sompi (default: 100000 = 0.001 XENOM)"))
        .arg(Arg::new("pool-fee-percent")
            .long("pool-fee-percent").value_name("FLOAT")
            .default_value("1.0").value_parser(clap::value_parser!(f64))
            .help("Pool operator fee percentage (default: 1.0%)"))
        .arg(Arg::new("fee-per-output")
            .long("fee-per-output").value_name("N")
            .default_value("2000").value_parser(clap::value_parser!(u64))
            .help("Estimated tx fee per output in sompi (default: 2000)"))
        // ── REST API ───────────────────────────────────────────────────────────
        .arg(Arg::new("api-listen")
            .long("api-listen").value_name("ADDR:PORT")
            .default_value("0.0.0.0:1445")
            .help("HTTP REST API listen address (0.0.0.0:1445). Set to empty string to disable."))
        .arg(Arg::new("pool-name")
            .long("pool-name").value_name("NAME")
            .default_value("Xenom Pool")
            .help("Pool name shown in the API"))
        .arg(Arg::new("db-path")
            .long("db-path").value_name("PATH")
            .default_value("pool.db")
            .help("SQLite database file path (default: pool.db). Set to empty string to disable."))
        .arg(Arg::new("keygen")
            .long("keygen")
            .action(clap::ArgAction::SetTrue)
            .help("Generate a fresh secp256k1 keypair, print the private key hex and matching xenom: address, then exit"))
}

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();

    // ── Key generator (--keygen) ───────────────────────────────────────────────
    if m.get_flag("keygen") {
        let (sk, pk) = secp256k1::generate_keypair(&mut secp256k1::rand::thread_rng());
        let addr = Address::new(
            Prefix::Mainnet,
            Version::PubKey,
            &pk.x_only_public_key().0.serialize(),
        );
        let addr_str = String::from(&addr);
        println!();
        println!("  Private key  : {}", sk.display_secret());
        println!("  Pool address : {addr_str}");
        println!();
        println!("Use these flags when starting the bridge:");
        println!("  --mining-address {addr_str} \\");
        println!("  --pool-private-key {}", sk.display_secret());
        println!();
        println!("Keep the private key SECRET — it controls spending of all pool coinbase rewards.");
        return Ok(());
    }

    let rpcserver      = m.get_one::<String>("rpcserver").unwrap();
    let listen_str     = m.get_one::<String>("listen").unwrap();
    let mining_address = m.get_one::<String>("mining-address").unwrap();
    let poll_ms        = *m.get_one::<u64>("poll-interval-ms").unwrap();

    let vardiff_cfg = VarDiffConfig {
        initial_diff:          *m.get_one::<f64>("vardiff-initial").unwrap(),
        min_diff:              *m.get_one::<f64>("vardiff-min").unwrap(),
        max_diff:              *m.get_one::<f64>("vardiff-max").unwrap(),
        target_shares_per_min: *m.get_one::<f64>("vardiff-target-spm").unwrap(),
        retarget_secs:         *m.get_one::<u64>("vardiff-retarget-secs").unwrap(),
        ..VarDiffConfig::default()
    };

    let pplns_window    = *m.get_one::<usize>("pplns-window").unwrap();
    let payout_file     = m.get_one::<String>("payout-file").map(PathBuf::from);
    let stats_interval  = *m.get_one::<u64>("stats-interval-secs").unwrap();

    let payment_cfg = PaymentConfig {
        confirm_depth:    *m.get_one::<u64>("confirm-depth").unwrap(),
        min_payout_sompi: *m.get_one::<u64>("min-payout-sompi").unwrap(),
        pool_fee_percent: *m.get_one::<f64>("pool-fee-percent").unwrap(),
        fee_per_output:   *m.get_one::<u64>("fee-per-output").unwrap(),
    };

    // Parse optional pool private key for auto-payouts
    let pool_keypair: Option<secp256k1::Keypair> = m
        .get_one::<String>("pool-private-key")
        .map(|hex| {
            let secp   = secp256k1::Secp256k1::new();
            let secret = secp256k1::SecretKey::from_str(hex)
                .expect("--pool-private-key must be 64 hex chars (32 bytes)");
            secp256k1::Keypair::from_secret_key(&secp, &secret)
        });

    let api_listen_str = m.get_one::<String>("api-listen").unwrap();
    let pool_name      = m.get_one::<String>("pool-name").unwrap().clone();
    let db_path        = m.get_one::<String>("db-path").unwrap().clone();

    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(mining_address.as_str()).context("Invalid --mining-address")?;
    let listen_addr: SocketAddr = listen_str.parse().context("Invalid --listen address")?;

    info!(
        "VarDiff: init={} min={} max={} target={:.1} spm retarget={}s",
        vardiff_cfg.initial_diff, vardiff_cfg.min_diff, vardiff_cfg.max_diff,
        vardiff_cfg.target_shares_per_min, vardiff_cfg.retarget_secs
    );
    info!("PPLNS window: {pplns_window} shares");

    // ── gRPC connection ───────────────────────────────────────────────────────
    let url = format!("grpc://{rpcserver}");
    info!("Connecting to node at {url}");
    let rpc = Arc::new(GrpcClient::connect(url.clone()).await.context("gRPC connect")?);
    info!("Connected to {url}");

    // ── Shared state ──────────────────────────────────────────────────────────
    let job_mgr:    Arc<RwLock<JobManager>> = Arc::new(RwLock::new(JobManager::new()));
    let accounting: Arc<Mutex<Accounting>>  = Arc::new(Mutex::new(
        Accounting::new(pplns_window, payout_file),
    ));
    let (job_tx, job_rx) = watch::channel::<Option<Arc<job::Job>>>(None);

    // ── Node polling task ─────────────────────────────────────────────────────
    {
        let rpc2     = rpc.clone();
        let jmgr2    = job_mgr.clone();
        let jtx2     = job_tx.clone();
        let pay      = pay_address.clone();
        let poll_dur = Duration::from_millis(poll_ms);

        tokio::spawn(async move {
            info!("Block-template poller started (interval={poll_ms}ms)");
            loop {
                match rpc2
                    .get_block_template_call(None, GetBlockTemplateRequest::new(pay.clone(), vec![]))
                    .await
                {
                    Ok(resp) => {
                        if !resp.is_synced {
                            warn!("Node not synced — waiting…");
                        }
                        let mut mgr = jmgr2.write().await;
                        if let Some(job) = mgr.update(resp.block) {
                            info!(
                                "New job {} daa={} bits={:#010x} epoch_seed={}…",
                                job.id,
                                job.template.header.daa_score,
                                job.template.header.bits,
                                &job.epoch_seed_hex[..8]
                            );
                            jtx2.send(Some(job)).ok();
                        }
                    }
                    Err(e) => {
                        warn!("get_block_template: {e} — retrying in 1s");
                        sleep(Duration::from_secs(1)).await;
                    }
                }
                sleep(poll_dur).await;
            }
        });
    }

    // ── Periodic stats logging ────────────────────────────────────────────────
    if stats_interval > 0 {
        let acct2 = accounting.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(stats_interval);
            loop {
                sleep(interval).await;
                acct2.lock().await.log_stats();
            }
        });
    }

    // ── SQLite database ────────────────────────────────────────────────────────────
    let database: Option<Arc<Db>> = if !db_path.is_empty() {
        match Db::open(&db_path).await {
            Ok(d) => {
                info!("Database opened: {db_path}");
                Some(Arc::new(d))
            }
            Err(e) => {
                warn!("Failed to open database {db_path}: {e} — running without DB");
                None
            }
        }
    } else {
        info!("Database disabled");
        None
    };

    // ── Auto-payout confirmation monitor ─────────────────────────────────
    if let Some(keypair) = pool_keypair {
        // ── UTXO consolidation sweep (every 15 s) ────────────────────────
        // Prevents mass-limit failures by keeping the UTXO set small.
        let rpc_sweep  = rpc.clone();
        let addr_sweep = pay_address.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(15);
            loop {
                sleep(interval).await;
                match payments::consolidate_utxos(&rpc_sweep, &addr_sweep, &keypair).await {
                    Ok(Some(tx_id)) => info!("UTXO sweep OK: {tx_id}"),
                    Ok(None)        => {}
                    Err(e)          => warn!("UTXO sweep skipped: {e}"),
                }
            }
        });

        let rpc3           = rpc.clone();
        let acct3          = accounting.clone();
        let pay_addr       = pay_address.clone();
        let pcfg           = payment_cfg.clone();
        let db3            = database.clone();
        let check_interval = Duration::from_secs(30);

        info!(
            "Auto-payout enabled: confirm_depth={} min_payout={} sompi pool_fee={:.1}%",
            pcfg.confirm_depth, pcfg.min_payout_sompi, pcfg.pool_fee_percent
        );

        tokio::spawn(async move {
            loop {
                sleep(check_interval).await;

                let current_daa = match rpc3.get_block_dag_info().await {
                    Ok(info) => info.virtual_daa_score,
                    Err(e)   => { warn!("get_block_dag_info: {e}"); continue; }
                };

                let confirmed = acct3.lock().await
                    .take_confirmed_payouts(current_daa, pcfg.confirm_depth);

                // Process ONE block per cycle — multiple sequential payouts would reuse
                // the same unconfirmed UTXOs and cause double-spend RPC failures.
                // Remaining confirmed blocks are picked up on the next cycle.
                if let Some(payout) = confirmed.into_iter().next() {
                    info!(
                        "Block {} confirmed (daa_score={} current={}), executing payout …",
                        payout.job_id, payout.block_daa_score, current_daa
                    );

                    // Mark as confirmed immediately — block IS valid even if payout later fails
                    if let Some(ref d) = db3 {
                        if let Err(e) = d.update_block_status(&payout.job_id, "confirmed", None).await {
                            warn!("DB update_block_status confirmed: {e}");
                        }
                    }

                    match execute_payout(&rpc3, &pay_addr, &keypair, &payout, &pcfg).await {
                        Ok(tx_id) => {
                            let tx_str = tx_id.to_string();
                            info!("Payout OK: job={} tx={tx_str}", payout.job_id);
                            acct3.lock().await.mark_paid(&payout.job_id, tx_str.clone());
                            if let Some(ref d) = db3 {
                                if let Err(e) = d.update_block_status(&payout.job_id, "paid", Some(&tx_str)).await {
                                    warn!("DB update_block_status paid: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            let reason = e.to_string();
                            if e.downcast_ref::<RetryablePayoutError>().is_some() {
                                // Transient — block stays 'confirmed', will retry next cycle
                                warn!("Payout retry (job {}): {reason}", payout.job_id);
                            } else {
                                // Permanent failure — mark so it won't be retried
                                warn!("Payout FAILED (job {}): {reason}", payout.job_id);
                                acct3.lock().await.mark_failed(&payout.job_id, reason);
                                if let Some(ref d) = db3 {
                                    if let Err(e) = d.update_block_status(&payout.job_id, "payout-failed", None).await {
                                        warn!("DB update_block_status payout-failed: {e}");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    } else {
        info!("Auto-payout disabled (no --pool-private-key provided)");
    }

    // ── REST API server ──────────────────────────────────────────────────────
    let api_state: Option<ApiState> = if !api_listen_str.is_empty() {
        let api_addr: SocketAddr = api_listen_str.parse().context("Invalid --api-listen")?;
        let state = ApiState::new(
            accounting.clone(),
            rpc.clone(),
            pool_name,
            database.clone(),
            listen_str.clone(),
        );
        let state2 = state.clone();
        tokio::spawn(async move {
            if let Err(e) = api::run_api_server(api_addr, state2).await {
                warn!("API server error: {e}");
            }
        });
        Some(state)
    } else {
        info!("REST API disabled");
        None
    };

    // ── Stratum TCP server (blocks forever) ─────────────────────────────────────
    stratum::run_server(listen_addr, job_rx, job_mgr, rpc, vardiff_cfg, accounting, api_state, database.clone()).await?;

    Ok(())
}
