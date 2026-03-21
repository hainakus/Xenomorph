mod capabilities;
mod executor;
mod job_loop;
mod proof;

use anyhow::{Context, Result};
use bioproof_core::BioProofKeypair;
use clap::{Arg, Command};
use std::path::PathBuf;
use std::sync::Arc;

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    kaspa_core::log::init_logger(None, "info");

    let m = cli().get_matches();
    let privkey = m.get_one::<String>("private-key").unwrap().clone();
    let inbox = PathBuf::from(m.get_one::<String>("inbox").unwrap());
    let work_root = PathBuf::from(m.get_one::<String>("work-root").unwrap());
    let node_addr = m.get_one::<String>("node").unwrap().clone();
    let poll_ms: u64 = m.get_one::<String>("poll-ms").and_then(|s| s.parse().ok()).unwrap_or(3_000);
    let concurrency: usize = m.get_one::<String>("concurrency").and_then(|s| s.parse().ok()).unwrap_or(1);
    let submit = m.get_flag("submit");
    let devnet = m.get_flag("devnet");

    // ── Keypair ───────────────────────────────────────────────────────────────
    let keypair = BioProofKeypair::from_hex(&privkey).context("invalid --private-key (expected 64 hex chars)")?;
    let worker_pubkey = keypair.pubkey_hex();
    log::info!("Worker pubkey: {worker_pubkey}");

    // ── Detect hardware + installed software ──────────────────────────────────
    log::info!("Detecting capabilities…");
    let caps = capabilities::detect(worker_pubkey, concurrency).await?;
    log::info!("  CPUs:     {}", caps.cpu_count);
    log::info!("  RAM:      {} MiB", caps.ram_mib);
    log::info!("  GPUs:     {}", caps.gpus.len());
    log::info!("  Backends: {}", caps.backends.iter().map(|b| b.to_string()).collect::<Vec<_>>().join(", "));
    log::info!("  JobTypes: {}", caps.job_types.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", "));

    // Optionally write capabilities JSON to work root for inspection.
    tokio::fs::create_dir_all(&work_root).await?;
    tokio::fs::write(work_root.join("capabilities.json"), serde_json::to_vec_pretty(&caps)?).await?;

    // ── Start daemon loop ─────────────────────────────────────────────────────
    let cfg = Arc::new(job_loop::WorkerConfig {
        job_inbox: inbox,
        work_root,
        privkey_hex: privkey,
        api_url: None,
        node_addr,
        poll_ms,
        submit,
        devnet,
    });

    job_loop::run(Arc::new(caps), cfg).await
}

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("bioproof-worker")
        .about("BioProof Scientific Worker daemon — detects capabilities, polls job inbox, executes pipelines and anchors results on Xenom")
        .arg(Arg::new("private-key")
            .short('k').long("private-key").value_name("HEX").required(true)
            .help("Worker secp256k1 private key (64 hex chars) — defines worker identity"))
        .arg(Arg::new("inbox")
            .short('i').long("inbox").value_name("DIR")
            .default_value("./job-inbox")
            .help("Directory scanned for *.json job files (done/ and failed/ sub-dirs auto-created)"))
        .arg(Arg::new("work-root")
            .short('w').long("work-root").value_name("DIR")
            .default_value("./work")
            .help("Root directory for per-job working directories (input/, output/, manifest.json)"))
        .arg(Arg::new("node")
            .short('n').long("node").value_name("ADDR")
            .default_value("grpc://localhost:36669")
            .help("Xenom node gRPC address for on-chain anchor submission"))
        .arg(Arg::new("poll-ms")
            .long("poll-ms").value_name("MS")
            .default_value("3000")
            .help("Job inbox poll interval in milliseconds"))
        .arg(Arg::new("concurrency")
            .short('c').long("concurrency").value_name("N")
            .default_value("1")
            .help("Maximum concurrent jobs"))
        .arg(Arg::new("submit")
            .long("submit")
            .action(clap::ArgAction::SetTrue)
            .help("Anchor completed job results on-chain (default: dry-run)"))
        .arg(Arg::new("devnet")
            .long("devnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use devnet address prefix and lower coinbase maturity"))
        .arg(Arg::new("testnet")
            .long("testnet")
            .action(clap::ArgAction::SetTrue)
            .help("Use testnet address prefix"))
}
