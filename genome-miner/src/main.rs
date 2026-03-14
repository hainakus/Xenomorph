mod api;
mod gpu;
mod l2_worker;
mod stratum_client;
mod tui;

use std::{
    io::Read,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};

use crate::tui::DashStats;
use clap::{Arg, ArgMatches, Command};
use kaspa_addresses::Address;
use kaspa_consensus_core::header::Header;
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_pow::genome_pow::{build_merkle_root, fragment_index, fragment_leaf_hash, genome_mix_hash, CachedLoader, GenomeDatasetLoader, GenomePowState, SyntheticLoader};
use kaspa_rpc_core::{
    api::rpc::RpcApi,
    model::message::GetBlockTemplateRequest,
    RpcRawBlock, RpcRawHeader,
};
use kaspa_pow::genome_file::FileGenomeLoader;
use kaspa_txscript::pay_to_address_script;
use rayon::prelude::*;
use tokio::{sync::mpsc, time::sleep};
use crate::stratum_client::{StratumClient, StratumJob, StratumSolution};

// ── CLI ───────────────────────────────────────────────────────────────────────

fn cli() -> Command {
    Command::new("genome-miner")
        .about("Xenomorph Genome PoW CPU miner + HF deployment tools")
        .subcommand_required(true)
        .subcommand(
            Command::new("mine")
                .about("Run the CPU miner (genome PoW or legacy KHeavyHash)")
                .arg(Arg::new("rpcserver").long("rpcserver").short('s').value_name("HOST:PORT").help("gRPC node endpoint (default: localhost:16668)"))
                .arg(Arg::new("mining-address").long("mining-address").short('a').value_name("ADDRESS").required(true).help("Reward address"))
                .arg(Arg::new("threads").long("threads").short('t').value_name("N").value_parser(clap::value_parser!(usize)).help("Mining threads (default: logical CPUs)"))
                .arg(Arg::new("nonce-batch").long("nonce-batch").value_name("N").value_parser(clap::value_parser!(u64)).default_value("50000").help("Nonces per rayon task"))
                .arg(Arg::new("genome-activation-daa-score").long("genome-activation-daa-score").value_name("SCORE").value_parser(clap::value_parser!(u64)).help("DAA score where Genome PoW activates (overrides --mainnet/--testnet/--devnet)"))
                .arg(Arg::new("genome-fragment-size").long("genome-fragment-size").value_name("BYTES").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Fragment size in bytes"))
                .arg(Arg::new("genome-file").long("genome-file").value_name("PATH").help("Path to grch38.xenom (required for mainnet Genome PoW; auto-detected from ~/.rusty-xenom/grch38.xenom)"))
                .arg(Arg::new("mainnet").long("mainnet").action(clap::ArgAction::SetTrue).help("Mainnet (genome activation DAA 21_370_801)"))
                .arg(Arg::new("testnet").long("testnet").action(clap::ArgAction::SetTrue).help("Testnet (genome activation DAA 0)"))
                .arg(Arg::new("devnet").long("devnet").action(clap::ArgAction::SetTrue).help("Devnet (genome activation DAA 0)"))
                .arg(Arg::new("no-tui").long("no-tui").action(clap::ArgAction::SetTrue).help("Disable TUI dashboard (plain log output)"))
                .arg(Arg::new("api-port").long("api-port").value_name("PORT").value_parser(clap::value_parser!(u16)).default_value("4000").help("HiveOS stats API port (0 = disabled)"))
                .arg(Arg::new("stratum").long("stratum").value_name("URL").help("Stratum pool URL, e.g. stratum+tcp://127.0.0.1:5555 (mutually exclusive with --rpcserver)"))
                .arg(Arg::new("stratum-worker").long("stratum-worker").value_name("NAME").help("Stratum worker name (default: --mining-address)"))
                .arg(Arg::new("stratum-password").long("stratum-password").value_name("PASS").default_value("x").help("Stratum password (default: x)"))
                .arg(Arg::new("l2-coordinator").long("l2-coordinator").value_name("URL").help("L2 coordinator URL for inline job execution (e.g. http://localhost:8091)"))
                .arg(Arg::new("l2-private-key").long("l2-private-key").value_name("HEX").help("secp256k1 private key (64 hex) for signing L2 results"))
                .arg(Arg::new("l2-gpu").long("l2-gpu").action(clap::ArgAction::SetTrue).help("Use GPU for BirdNET inference (requires CUDA + PyTorch GPU)"))
                .arg(Arg::new("l2-perch-script").long("l2-perch-script").value_name("PATH").help("Path to perch_infer.py (auto-detected if omitted)"))
        )
        .subcommand(
            Command::new("suggest-params")
                .about("Fetch current DAA score and print ready-to-paste params.rs activation values")
                .arg(Arg::new("rpcserver").long("rpcserver").short('s').value_name("HOST:PORT").help("gRPC node endpoint (default: localhost:16668)"))
                .arg(Arg::new("fitness-buffer").long("fitness-buffer").value_name("N").value_parser(clap::value_parser!(u64)).default_value("1000").help("Blocks between tip and fitness_coinbase activation"))
                .arg(Arg::new("pow-buffer").long("pow-buffer").value_name("N").value_parser(clap::value_parser!(u64)).default_value("200").help("Extra buffer blocks after epoch_len for genome_pow activation"))
                .arg(Arg::new("epoch-len").long("epoch-len").value_name("N").value_parser(clap::value_parser!(u64)).default_value("200").help("epoch_len from params (default 200)"))
        )
        .subcommand(
            Command::new("compute-merkle-root")
                .about("Compute genome_merkle_root from a flat GRCh38 binary file")
                .arg(Arg::new("genome-file").long("genome-file").short('f').value_name("PATH").required(true).help("Path to flat GRCh38 genome binary"))
                .arg(Arg::new("fragment-size").long("fragment-size").value_name("BYTES").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Fragment size in bytes"))
        )
        .subcommand(
            Command::new("address-to-script")
                .about("Convert a Xenomorph address to fund_script_public_key hex for params.rs")
                .arg(Arg::new("address").long("address").short('a').value_name("ADDRESS").required(true).help("Fund wallet address"))
        )
        .subcommand(
            Command::new("keygen")
                .about("Generate a fresh secp256k1 keypair for use as --l2-private-key")
        )
        .subcommand(
            Command::new("gpu")
                .about("Run the GPU miner (wgpu — Metal on Mac, Vulkan elsewhere)")
                .arg(Arg::new("rpcserver").long("rpcserver").short('s').value_name("HOST:PORT").help("gRPC node endpoint (default: localhost:36669)"))
                .arg(Arg::new("mining-address").long("mining-address").short('a').value_name("ADDRESS").help("Reward address (required for node mode; optional in stratum mode)"))
                .arg(Arg::new("batch-size").long("batch-size").value_name("N").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Nonces per GPU dispatch (default: 1M)"))
                .arg(Arg::new("genome-activation-daa-score").long("genome-activation-daa-score").value_name("SCORE").value_parser(clap::value_parser!(u64)).help("DAA score where Genome PoW activates (overrides --mainnet/--testnet/--devnet)"))
                .arg(Arg::new("genome-fragment-size").long("genome-fragment-size").value_name("BYTES").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Fragment size in bytes"))
                .arg(Arg::new("mainnet").long("mainnet").action(clap::ArgAction::SetTrue).help("Mainnet (genome activation DAA 21_370_801)"))
                .arg(Arg::new("testnet").long("testnet").action(clap::ArgAction::SetTrue).help("Testnet (genome activation DAA 0)"))
                .arg(Arg::new("devnet").long("devnet").action(clap::ArgAction::SetTrue).help("Devnet (genome activation DAA 0)"))
                .arg(Arg::new("genome-file").long("genome-file").value_name("PATH").help("Path to grch38.xenom (required for mainnet Genome PoW; auto-detected from ~/.rusty-xenom/grch38.xenom if absent)"))
                .arg(Arg::new("gpu").long("gpu").value_name("INDICES|all").default_value("0").help("GPU adapter(s) to mine on: '0', '1', '0,1,2', or 'all'. Run --list-gpus to see indices."))
                .arg(Arg::new("list-gpus").long("list-gpus").action(clap::ArgAction::SetTrue).help("List available GPU adapters with their indices and exit"))
                .arg(Arg::new("nonce-offset").long("nonce-offset").value_name("N").value_parser(clap::value_parser!(u64)).default_value("0").help("Instance index for multi-process setups (0=first, 1=second …). Each instance mines a non-overlapping nonce segment."))
                .arg(Arg::new("no-tui").long("no-tui").action(clap::ArgAction::SetTrue).help("Disable TUI dashboard (plain log output)"))
                .arg(Arg::new("stratum").long("stratum").value_name("URL").help("Stratum pool URL, e.g. stratum+tcp://pool.example.com:1444 (mutually exclusive with --rpcserver)"))
                .arg(Arg::new("stratum-worker").long("stratum-worker").value_name("NAME").help("Stratum worker name (default: --mining-address)"))
                .arg(Arg::new("stratum-password").long("stratum-password").value_name("PASS").default_value("x").help("Stratum password (default: x)"))
                .arg(Arg::new("api-port").long("api-port").value_name("PORT").value_parser(clap::value_parser!(u16)).default_value("4000").help("HiveOS stats API port (0 = disabled)"))
        )
}

// ── Miner state (mine subcommand) ────────────────────────────────────────────

struct MineConfig {
    rpcserver: String,
    mining_address: String,
    threads: usize,
    nonce_batch: u64,
    genome_pow_activation_daa_score: u64,
    genome_fragment_size_bytes: u32,
}

struct MinerState {
    cfg: MineConfig,
    template_generation: AtomicU64,
    template_id: std::sync::Mutex<Option<kaspa_hashes::Hash>>,
    found: AtomicBool,
}

impl MinerState {
    fn new(cfg: MineConfig) -> Self {
        Self { cfg, template_generation: AtomicU64::new(0), template_id: std::sync::Mutex::new(None), found: AtomicBool::new(false) }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    let matches = cli().get_matches();
    // When TUI is active the alternate screen owns the terminal; redirect log
    // output to /tmp/genome-miner.log so it doesn't bleed into the TUI display.
    let tui_active = matches!(matches.subcommand_name(), Some("mine") | Some("gpu"))
        && !matches
            .subcommand()
            .map(|(_, m)| m.get_flag("no-tui"))
            .unwrap_or(true);
    if tui_active {
        kaspa_core::log::init_logger(Some("/tmp"), "info,wgpu_core=warn,wgpu_hal=warn,naga=warn");
        // init_logger always adds a stdout appender; silence it completely so
        // no raw bytes bleed into the TUI alternate screen. Events are shown
        // in the TUI log pane via DashStats::push_log() instead.
        log::set_max_level(log::LevelFilter::Off);
        // Suppress Mesa/Vulkan loader's XDG_RUNTIME_DIR stderr warning which
        // bypasses the log crate and would corrupt the TUI alternate screen.
        if std::env::var_os("XDG_RUNTIME_DIR").is_none() {
            std::env::set_var("XDG_RUNTIME_DIR", "/tmp");
        }
    } else {
        kaspa_core::log::init_logger(None, "info,wgpu_core=warn,wgpu_hal=warn,naga=warn");
    }
    match matches.subcommand() {
        Some(("mine", m)) => {
            let no_tui  = m.get_flag("no-tui");
            let api_port = m.get_one::<u16>("api-port").copied().unwrap_or(4000);
            let rpc     = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:16668".to_owned());
            let dash    = Arc::new(Mutex::new(DashStats::new(
                rpc,
                "CPU · Genome PoW".to_owned(),
                m.get_one::<usize>("threads").copied().unwrap_or_else(rayon::current_num_threads),
            )));
            if !no_tui {
                let d2 = dash.clone();
                std::thread::spawn(move || tui::run_tui(d2));
            }
            if api_port > 0 {
                let d2 = dash.clone();
                let start = std::time::Instant::now();
                tokio::spawn(api::run_api_server(api_port, d2, start));
            }
            cmd_mine(m, dash).await;
        }
        Some(("suggest-params", m))      => cmd_suggest_params(m).await,
        Some(("compute-merkle-root", m)) => cmd_compute_merkle_root(m),
        Some(("address-to-script", m))   => cmd_address_to_script(m),
        Some(("keygen", _))              => cmd_keygen(),
        Some(("gpu", m)) => {
            let no_tui   = m.get_flag("no-tui");
            let api_port = m.get_one::<u16>("api-port").copied().unwrap_or(4000);
            let rpc      = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:36669".to_owned());
            let dash     = Arc::new(Mutex::new(DashStats::new(
                rpc,
                "GPU · initialising".to_owned(),
                0,
            )));
            if !no_tui {
                let d2 = dash.clone();
                std::thread::spawn(move || tui::run_tui(d2));
            }
            if api_port > 0 {
                let start = std::time::Instant::now();
                tokio::spawn(api::run_api_server(api_port, dash.clone(), start));
                tokio::spawn(api::hw_poll_task(dash.clone()));
            }
            gpu::cmd_gpu(m, dash).await;
        }
        _ => unreachable!(),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Resolves the genome PoW activation DAA score from CLI flags.
/// Priority: explicit `--genome-activation-daa-score` > `--testnet`/`--devnet` > mainnet default
fn resolve_activation(m: &ArgMatches) -> u64 {
    if let Some(&score) = m.get_one::<u64>("genome-activation-daa-score") {
        return score;
    }
    if m.get_flag("testnet") || m.get_flag("devnet") {
        return 0;
    }
    kaspa_consensus_core::hashing::header::EPOCH_SEED_HASH_ACTIVATION_MAINNET
}

// ── mine ─────────────────────────────────────────────────────────────────────

async fn cmd_mine(m: &ArgMatches, dash: Arc<Mutex<DashStats>>) {
    let threads  = m.get_one::<usize>("threads").copied().unwrap_or_else(rayon::current_num_threads);
    let frag_size = m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576);
    let genome_activation = resolve_activation(m);
    let genome_path: Option<String> = m.get_one::<String>("genome-file").cloned().or_else(|| {
        let default = dirs::home_dir()?.join(".rusty-xenom").join("grch38.xenom");
        if default.exists() { Some(default.to_string_lossy().into_owned()) } else { None }
    });

    // Load the real genome dataset if available; otherwise fall back to SyntheticLoader.
    // The Arc<dyn GenomeDatasetLoader> is shared across all rayon worker threads.
    let file_loader: Option<Arc<FileGenomeLoader>> = genome_path.as_deref().map(|path| {
        info!("Loading genome file {path} ...");
        let loader = FileGenomeLoader::open(std::path::Path::new(path), frag_size, false)
            .unwrap_or_else(|e| panic!("Failed to open genome file '{path}': {e}"));
        info!("Genome file loaded — {} MB", loader.packed_dataset().map(|b| b.len() / 1_048_576).unwrap_or(0));
        Arc::new(loader)
    });
    if genome_path.is_some() && file_loader.is_none() {
        warn!("Genome file not found — mainnet Genome PoW will fail. Use --genome-file <PATH>.");
    }

    let mining_addr = m.get_one::<String>("mining-address").cloned().expect("--mining-address required");

    // ── Stratum mode ──────────────────────────────────────────────────────────
    if let Some(stratum_url) = m.get_one::<String>("stratum").cloned() {
        let worker = m.get_one::<String>("stratum-worker")
            .cloned()
            .unwrap_or_else(|| mining_addr.clone());
        let password = m.get_one::<String>("stratum-password")
            .cloned()
            .unwrap_or_else(|| "x".to_owned());

        let l2_cfg = match (
            m.get_one::<String>("l2-coordinator").cloned(),
            m.get_one::<String>("l2-private-key").cloned(),
        ) {
            (Some(url), Some(key)) => {
                let use_gpu = m.get_flag("l2-gpu");
                let perch_script = m.get_one::<String>("l2-perch-script").map(std::path::PathBuf::from);
                match l2_worker::L2Config::new(url, key, use_gpu, perch_script) {
                    Ok(c)  => { info!("L2 inline worker enabled — coordinator={}", c.coordinator_url); Some(c) }
                    Err(e) => { warn!("L2 config error: {e} — L2 disabled"); None }
                }
            }
            _ => None,
        };

        let (job_tx, job_rx) = mpsc::channel::<StratumJob>(8);
        let (sol_tx, sol_rx) = mpsc::channel::<StratumSolution>(32);
        let client = StratumClient::new(&stratum_url, &worker, &password);
        let dash2  = dash.clone();
        tokio::spawn(async move { client.run(job_tx, sol_rx, dash2).await; });

        mine_stratum(threads, frag_size, genome_activation, file_loader, l2_cfg, job_rx, sol_tx, dash).await;
        return;
    }

    let cfg = MineConfig {
        rpcserver: m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:16668".to_owned()),
        mining_address: mining_addr,
        threads,
        nonce_batch: m.get_one::<u64>("nonce-batch").copied().unwrap_or(50_000),
        genome_pow_activation_daa_score: genome_activation,
        genome_fragment_size_bytes: frag_size,
    };

    let url = format!("grpc://{}", cfg.rpcserver);
    info!("Connecting to {url}");
    let rpc = Arc::new(GrpcClient::connect(url).await.expect("Failed to connect"));
    info!("Connected — threads={} genome_activation={} genome_file={}",
        cfg.threads, cfg.genome_pow_activation_daa_score, genome_path.as_deref().unwrap_or("(synthetic)"));
    dash.lock().unwrap().connected = true;

    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(cfg.mining_address.as_str()).expect("Invalid --mining-address");
    let state = Arc::new(MinerState::new(cfg));
    let pool  = rayon::ThreadPoolBuilder::new().num_threads(state.cfg.threads).build().expect("rayon pool");

    let mut total_hashes: u64 = 0;
    let mut report_timer = Instant::now();

    loop {
        let resp = match rpc.get_block_template_call(None, GetBlockTemplateRequest::new(pay_address.clone(), vec![])).await {
            Ok(r) => r,
            Err(e) => { warn!("get_block_template: {e}"); sleep(Duration::from_secs(1)).await; continue; }
        };
        let rpc_block: RpcRawBlock = resp.block;
        if !resp.is_synced { warn!("Node not synced"); }

        let current_id = rpc_block.header.accepted_id_merkle_root;
        {
            let mut guard = state.template_id.lock().unwrap();
            if *guard == Some(current_id) {
                drop(guard); // must drop before .await
                sleep(Duration::from_millis(200)).await;
                continue;
            }
            *guard = Some(current_id);
        }

        let header: Header = (&rpc_block.header).into();
        let genome_active = header.daa_score >= state.cfg.genome_pow_activation_daa_score;
        let gen = state.template_generation.fetch_add(1, Ordering::Relaxed) + 1;
        state.found.store(false, Ordering::Relaxed);
        {
            let mut s = dash.lock().unwrap();
            s.daa_score     = header.daa_score;
            s.bits          = header.bits;
            s.genome_active = genome_active;
            let mode_str    = if genome_active { "Genome PoW" } else { "KHeavyHash" };
            s.mode          = format!("CPU×{} · {mode_str}", state.cfg.threads);
            s.push_log(format!("New template daa={} bits={:#010x} genome={}", header.daa_score, header.bits, genome_active));
        }
        info!("New template daa={} bits={:#010x} genome={}", header.daa_score, header.bits, genome_active);

        let batch     = state.cfg.nonce_batch;
        let frag_size = state.cfg.genome_fragment_size_bytes;

        // Pre-compute per-template constants once (not per-nonce).
        let pre_pow_hash = kaspa_consensus_core::hashing::header::hash_override_nonce_time(&header, 0, 0);
        let target       = kaspa_math::Uint256::from_compact_target_bits(header.bits);
        let epoch_seed   = header.epoch_seed;

        // When a real packed dataset is available, use genome_mix_hash directly
        // (same algorithm as the GPU shader + node validator — no per-nonce unpack).
        // Fall back to SyntheticLoader + check_pow_with_fragment for devnet.
        let packed_opt: Option<&[u8]> = if genome_active {
            file_loader.as_ref().and_then(|fl| fl.packed_dataset())
        } else {
            None
        };
        let synth_loader: Option<Arc<dyn GenomeDatasetLoader>> = if genome_active && packed_opt.is_none() {
            Some(Arc::new(CachedLoader::new(SyntheticLoader::new(frag_size, epoch_seed), 256)))
        } else {
            None
        };

        let mut nonce_base: u64 = 0;
        let solution: Option<u64> = 'search: loop {
            if state.template_generation.load(Ordering::Relaxed) != gen { break 'search None; }
            let range_start = nonce_base;
            nonce_base = nonce_base.saturating_add(batch * state.cfg.threads as u64);
            let winning = pool.install(|| {
                (0..state.cfg.threads as u64).into_par_iter().find_map_first(|tid| {
                    let start = range_start + tid * batch;
                    if let Some(packed) = packed_opt {
                        try_nonce_range_genome_packed(packed, &epoch_seed, &pre_pow_hash, &target, start, start + batch)
                    } else if let Some(loader) = synth_loader.as_ref() {
                        try_nonce_range_genome(&header, start, start + batch, frag_size, loader.as_ref())
                    } else {
                        try_nonce_range_legacy(&header, start, start + batch)
                    }
                })
            });
            total_hashes += batch * state.cfg.threads as u64;
            if let Some(n) = winning { break 'search Some(n); }
            if report_timer.elapsed() >= Duration::from_secs(10) {
                let elapsed = report_timer.elapsed().as_secs_f64();
                let mhs = total_hashes as f64 / elapsed / 1_000_000.0;
                info!("Hashrate: {:.3} MH/s", mhs);
                dash.lock().unwrap().total_mhs = mhs;
                total_hashes = 0;
                report_timer = Instant::now();
            }
            tokio::task::yield_now().await;
        };

        if let Some(nonce) = solution {
            match rpc.submit_block(build_raw_block(&rpc_block, nonce), false).await {
                Ok(r) => {
                    info!("Block submitted: {:?}", r.report);
                    let accepted = matches!(r.report, kaspa_rpc_core::SubmitBlockReport::Success);
                    let mut s = dash.lock().unwrap();
                    if accepted { s.accepted += 1; s.push_log(format!("Block accepted  daa={}", header.daa_score)); }
                    else        { s.rejected += 1; s.push_log(format!("Block rejected  {:?}", r.report)); }
                }
                Err(e) => { warn!("submit_block: {e}"); dash.lock().unwrap().push_log(format!("submit error: {e}")); }
            }
        }
    }
}

// ── mine_stratum — CPU mining loop driven by Stratum pool ─────────────────────
//
// Nonce assignment: the bridge owns the upper 32 bits (extranonce1).
// The miner searches lower 32 bits (extranonce2) in [0, 2^32).
// Full nonce = (extranonce1 << 32) | extranonce2.

async fn mine_stratum(
    threads:           usize,
    frag_size:         u32,
    genome_activation: u64,
    file_loader:       Option<Arc<FileGenomeLoader>>,
    l2_cfg:            Option<l2_worker::L2Config>,
    mut job_rx:        mpsc::Receiver<StratumJob>,
    sol_tx:            mpsc::Sender<StratumSolution>,
    dash:              Arc<Mutex<DashStats>>,
) {
    info!("Stratum CPU miner started — waiting for first job …");
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");

    let batch: u64 = 50_000;
    let mut total_hashes: u64 = 0;
    let mut report_timer = Instant::now();

    // Wait for first job
    let mut current_job: StratumJob = match job_rx.recv().await {
        Some(j) => j,
        None    => { warn!("Stratum job channel closed before first job"); return; }
    };

    // nonce_base persists across share submissions for the same job
    // — only resets when a genuinely new job arrives.
    let mut nonce_base: u64 = (current_job.extranonce1 as u64) << 32;
    let mut active_job_id = current_job.job_id.clone();

    loop {
        // Apply any pending new job; dispatch L2 task if present
        loop {
            match job_rx.try_recv() {
                Ok(new_job) => {
                    if new_job.job_id != active_job_id || new_job.clean_jobs {
                        nonce_base    = (new_job.extranonce1 as u64) << 32;
                        active_job_id = new_job.job_id.clone();
                    }
                    // Spawn inline L2 worker if coordinator is configured
                    if let (Some(ref cfg), Some(ref l2_val)) = (&l2_cfg, &new_job.l2_job) {
                        let l2_job_id = l2_val["job_id"].as_str().unwrap_or("").to_owned();
                        if !l2_job_id.is_empty() {
                            let cfg2  = cfg.clone();
                            let val2  = l2_val.clone();
                            tokio::spawn(async move {
                                l2_worker::run_l2_job(cfg2, val2).await;
                            });
                        }
                    }
                    current_job = new_job;
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => return,
            }
        }

        let genome_active = current_job.daa_score >= genome_activation;
        let pre_pow_hash  = current_job.pre_pow_hash;
        let epoch_seed    = current_job.epoch_seed;
        let timestamp     = current_job.timestamp;
        let bits          = current_job.bits;
        let extranonce1   = current_job.extranonce1;
        let job_id        = current_job.job_id.clone();

        let nonce_start: u64 = (extranonce1 as u64) << 32;
        let nonce_max:   u64 = nonce_start + u32::MAX as u64;
        let target = kaspa_math::Uint256::from_compact_target_bits(bits);

        let packed_opt: Option<&[u8]> = if genome_active {
            file_loader.as_ref().and_then(|fl| fl.packed_dataset())
        } else {
            None
        };
        let synth_loader: Option<Arc<dyn GenomeDatasetLoader>> = if genome_active && packed_opt.is_none() {
            Some(Arc::new(CachedLoader::new(SyntheticLoader::new(frag_size, epoch_seed), 256)))
        } else {
            None
        };

        {
            let mut s = dash.lock().unwrap();
            s.bits          = bits;
            s.genome_active = genome_active;
            s.connected     = true;
            let mode = if genome_active { "Genome PoW" } else { "KHeavyHash" };
            s.mode = format!("CPU×{threads} · {mode} · Pool");
        }

        // Advance nonce_base, wrapping within [nonce_start, nonce_max]
        if nonce_base > nonce_max { nonce_base = nonce_start; }
        let start = nonce_base;
        let end   = (start + batch * threads as u64).min(nonce_max);
        nonce_base = end;

        let winning = pool.install(|| {
            (0..threads as u64).into_par_iter().find_map_first(|tid| {
                let s = start + tid * batch;
                let e = (s + batch).min(nonce_max);
                if genome_active {
                    if let Some(packed) = packed_opt {
                        try_nonce_range_genome_packed(packed, &epoch_seed, &pre_pow_hash, &target, s, e)
                    } else if let Some(ref loader) = synth_loader {
                        try_nonce_range_genome_stratum(&pre_pow_hash, &target, &epoch_seed, frag_size, loader.as_ref(), s, e)
                    } else {
                        None
                    }
                } else {
                    try_nonce_range_legacy_stratum(&pre_pow_hash, timestamp, bits, s, e)
                }
            })
        });

        total_hashes += batch * threads as u64;

        if let Some(nonce) = winning {
            let extranonce2 = (nonce & 0xFFFF_FFFF) as u32;
            info!("Share found! nonce={:#018x} en2={:08x} job={job_id}", nonce, extranonce2);
            dash.lock().unwrap().push_log(format!("Share  nonce={nonce:#018x}  job={job_id}"));
            // Advance nonce_base past winning nonce to avoid re-finding it
            nonce_base = nonce + 1;
            if sol_tx.send(StratumSolution { job_id, extranonce2 }).await.is_err() {
                warn!("Stratum solution channel closed — exiting");
                return;
            }
        }

        if report_timer.elapsed() >= Duration::from_secs(10) {
            let elapsed = report_timer.elapsed().as_secs_f64();
            let mhs = total_hashes as f64 / elapsed / 1_000_000.0;
            info!("Hashrate: {:.3} MH/s", mhs);
            dash.lock().unwrap().total_mhs = mhs;
            total_hashes = 0;
            report_timer = Instant::now();
        }

        tokio::task::yield_now().await;
    }
}

/// Construct a KHeavyHash PoW state for stratum mining.
/// Uses public primitives from kaspa_pow — no consensus logic modified.
#[inline]
fn stratum_state(pre_pow_hash: kaspa_hashes::Hash, timestamp: u64, bits: u32) -> kaspa_pow::State {
    kaspa_pow::State {
        matrix: kaspa_pow::matrix::Matrix::generate(pre_pow_hash),
        target: kaspa_math::Uint256::from_compact_target_bits(bits),
        hasher: kaspa_hashes::PowHash::new(pre_pow_hash, timestamp),
    }
}

fn try_nonce_range_legacy_stratum(
    pre_pow_hash: &kaspa_hashes::Hash,
    timestamp:    u64,
    bits:         u32,
    start:        u64,
    end:          u64,
) -> Option<u64> {
    let state = stratum_state(*pre_pow_hash, timestamp, bits);
    for nonce in start..end {
        let (valid, _) = state.check_pow(nonce);
        if valid { return Some(nonce); }
    }
    None
}

fn try_nonce_range_genome_stratum(
    pre_pow_hash: &kaspa_hashes::Hash,
    target:       &kaspa_math::Uint256,
    epoch_seed:   &kaspa_hashes::Hash,
    frag_size:    u32,
    loader:       &dyn GenomeDatasetLoader,
    start:        u64,
    end:          u64,
) -> Option<u64> {
    let state = GenomePowState::new(*pre_pow_hash, *target, *epoch_seed, frag_size);
    for nonce in start..end {
        let idx      = fragment_index(epoch_seed, nonce, frag_size);
        let fragment = loader.load_fragment(idx)?;
        let (valid, _, _) = state.check_pow_with_fragment(nonce, &fragment);
        if valid { return Some(nonce); }
    }
    None
}

// ── suggest-params ────────────────────────────────────────────────────────────

async fn cmd_suggest_params(m: &ArgMatches) {
    let rpcserver = m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:16668".to_owned());
    let fitness_buffer = m.get_one::<u64>("fitness-buffer").copied().unwrap_or(1_000);
    let pow_buffer     = m.get_one::<u64>("pow-buffer").copied().unwrap_or(200);
    let epoch_len      = m.get_one::<u64>("epoch-len").copied().unwrap_or(200);

    let url = format!("grpc://{rpcserver}");
    let rpc = GrpcClient::connect(url).await.expect("Failed to connect");
    let dag_info = rpc
        .get_block_dag_info_call(None, kaspa_rpc_core::model::message::GetBlockDagInfoRequest {})
        .await
        .expect("get_block_dag_info failed");

    let tip_daa = dag_info.virtual_daa_score;
    let fitness_activation = tip_daa + fitness_buffer;
    let genome_activation  = fitness_activation + epoch_len + pow_buffer;

    println!();
    println!("// ── Current chain tip DAA score: {tip_daa} ──");
    println!();
    println!("// Paste these two lines into the relevant Params block in params.rs:");
    println!("fitness_coinbase_activation_daa_score: {fitness_activation},");
    println!("genome_pow_activation_daa_score:       {genome_activation},");
    println!();
    println!("// Timeline:");
    println!("//   tip                 = {tip_daa}");
    println!("//   fitness_coinbase_activation = {fitness_activation}  (+{fitness_buffer} blocks)");
    println!("//   genome_pow_activation       = {genome_activation}  (+{} blocks from fitness)", epoch_len + pow_buffer);
    println!();
    println!("// Remaining checklist items (set manually):");
    println!("//   genome_merkle_root:      run `genome-miner compute-merkle-root --genome-file <PATH>`");
    println!("//   fund_script_public_key:  run `genome-miner address-to-script --address <ADDR>`");
}

// ── compute-merkle-root ───────────────────────────────────────────────────────

fn cmd_compute_merkle_root(m: &ArgMatches) {
    let path          = m.get_one::<String>("genome-file").unwrap();
    let fragment_size = m.get_one::<u32>("fragment-size").copied().unwrap_or(1_048_576) as usize;

    eprintln!("Reading {path} with fragment_size={fragment_size} bytes ...");

    let mut file = std::fs::File::open(path).unwrap_or_else(|e| panic!("Cannot open {path}: {e}"));
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Read failed");

    let total_fragments = data.len().div_ceil(fragment_size);
    eprintln!("File size: {} bytes → {total_fragments} fragments", data.len());

    // Compute leaf hashes in parallel
    let leaves: Vec<kaspa_hashes::Hash> = (0..total_fragments)
        .into_par_iter()
        .map(|idx| {
            let start = idx * fragment_size;
            let end   = (start + fragment_size).min(data.len());
            fragment_leaf_hash(idx as u64, &data[start..end])
        })
        .collect();

    let root = build_merkle_root(&leaves);

    // Format as 64-char lowercase hex
    let root_bytes: &[u8] = &root.as_bytes()[..];
    let mut hex_buf = vec![0u8; root_bytes.len() * 2];
    faster_hex::hex_encode(root_bytes, &mut hex_buf).expect("hex encode");
    let root_hex = std::str::from_utf8(&hex_buf).unwrap();

    println!();
    println!("// Paste this line into the relevant Params block in params.rs:");
    println!("genome_merkle_root: \"{root_hex}\",");
}

// ── address-to-script ─────────────────────────────────────────────────────────

fn cmd_address_to_script(m: &ArgMatches) {
    let addr_str = m.get_one::<String>("address").unwrap();
    let address  = Address::try_from(addr_str.as_str()).unwrap_or_else(|e| panic!("Invalid address: {e}"));
    let spk      = pay_to_address_script(&address);

    // Format: version(2B big-endian hex) || script bytes hex
    let version_bytes = spk.version.to_be_bytes();
    let script_bytes  = spk.script();
    let total_len     = (version_bytes.len() + script_bytes.len()) * 2;
    let mut hex_buf   = vec![0u8; total_len];
    faster_hex::hex_encode(&version_bytes, &mut hex_buf[..4]).expect("hex version");
    faster_hex::hex_encode(script_bytes, &mut hex_buf[4..]).expect("hex script");
    let hex_str = std::str::from_utf8(&hex_buf).unwrap();

    println!();
    println!("// Paste this line into the relevant Params block in params.rs:");
    println!("fund_script_public_key: \"{hex_str}\",");
}

// ── keygen ────────────────────────────────────────────────────────────────────

fn cmd_keygen() {
    let kp = bioproof_core::BioProofKeypair::generate();
    println!();
    println!("=== L2 Worker Keypair ===");
    println!("Private key (--l2-private-key): {}", kp.privkey_hex());
    println!("Public  key (worker identity):  {}", kp.pubkey_hex());
    println!();
    println!("KEEP THE PRIVATE KEY SECRET — never share it.");
    println!("This keypair is INDEPENDENT from the pool's key.");
    println!();
}

// ── PoW search helpers ────────────────────────────────────────────────────────

/// Fast Genome PoW path when the full packed dataset is in RAM.
/// Uses `genome_mix_hash` (same algorithm as GPU shader + node validator).
/// No per-nonce fragment unpacking — 9 blake3 hashes + 8 random 32-byte reads.
fn try_nonce_range_genome_packed(
    packed: &[u8],
    epoch_seed: &kaspa_hashes::Hash,
    pre_pow_hash: &kaspa_hashes::Hash,
    target: &kaspa_math::Uint256,
    start: u64,
    end: u64,
) -> Option<u64> {
    for nonce in start..end {
        let pow = genome_mix_hash(packed, epoch_seed, nonce, pre_pow_hash);
        if pow <= *target {
            return Some(nonce);
        }
    }
    None
}

/// Tries all nonces in `[start, end)` using either Genome PoW or legacy KHeavyHash.
/// Returns the first winning nonce or `None` if none found in range.
#[allow(dead_code)]
fn try_nonce_range(
    header: &Header,
    start: u64,
    end: u64,
    genome_active: bool,
    fragment_size_bytes: u32,
    loader: &dyn GenomeDatasetLoader,
) -> Option<u64> {
    if genome_active {
        try_nonce_range_genome(header, start, end, fragment_size_bytes, loader)
    } else {
        try_nonce_range_legacy(header, start, end)
    }
}

fn try_nonce_range_legacy(header: &Header, start: u64, end: u64) -> Option<u64> {
    let state = kaspa_pow::State::new(header);
    for nonce in start..end {
        let (valid, _pow) = state.check_pow(nonce);
        if valid {
            return Some(nonce);
        }
    }
    None
}

fn try_nonce_range_genome(
    header: &Header,
    start: u64,
    end: u64,
    fragment_size_bytes: u32,
    loader: &dyn GenomeDatasetLoader,
) -> Option<u64> {
    let state = GenomePowState::new(
        kaspa_consensus_core::hashing::header::hash_override_nonce_time(header, 0, 0),
        kaspa_math::Uint256::from_compact_target_bits(header.bits),
        header.epoch_seed,
        fragment_size_bytes,
    );
    for nonce in start..end {
        let idx = fragment_index(&header.epoch_seed, nonce, fragment_size_bytes);
        let fragment = match loader.load_fragment(idx) {
            Some(f) => f,
            None => continue,
        };
        let (valid, _pow, _fitness) = state.check_pow_with_fragment(nonce, &fragment);
        if valid {
            return Some(nonce);
        }
    }
    None
}

/// Builds an `RpcRawBlock` from the template with the winning nonce injected.
fn build_raw_block(template: &RpcRawBlock, nonce: u64) -> RpcRawBlock {
    let raw_header = RpcRawHeader {
        version: template.header.version,
        parents_by_level: template.header.parents_by_level.clone(),
        hash_merkle_root: template.header.hash_merkle_root,
        accepted_id_merkle_root: template.header.accepted_id_merkle_root,
        utxo_commitment: template.header.utxo_commitment,
        timestamp: template.header.timestamp,
        bits: template.header.bits,
        nonce,
        daa_score: template.header.daa_score,
        blue_work: template.header.blue_work,
        blue_score: template.header.blue_score,
        epoch_seed: template.header.epoch_seed,
        pruning_point: template.header.pruning_point,
    };
    RpcRawBlock { header: raw_header, transactions: template.transactions.clone() }
}
