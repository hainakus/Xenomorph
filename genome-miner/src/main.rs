use std::{
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use clap::{Arg, Command};
use kaspa_consensus_core::header::Header;
use kaspa_core::{info, warn};
use kaspa_addresses::Address;
use kaspa_grpc_client::GrpcClient;
use kaspa_pow::genome_pow::{fragment_index, CachedLoader, GenomeDatasetLoader, GenomePowState, SyntheticLoader};
use kaspa_rpc_core::{
    api::rpc::RpcApi,
    model::message::GetBlockTemplateRequest,
    RpcRawBlock, RpcRawHeader,
};
use rayon::prelude::*;
use tokio::time::sleep;

// ── CLI ───────────────────────────────────────────────────────────────────────

struct Config {
    rpcserver: String,
    mining_address: String,
    threads: usize,
    nonce_batch: u64,
    genome_pow_activation_daa_score: u64,
    genome_fragment_size_bytes: u32,
}

impl Config {
    fn parse() -> Self {
        let m = cli().get_matches();
        let threads = m.get_one::<usize>("threads").copied().unwrap_or_else(|| rayon::current_num_threads());
        Self {
            rpcserver: m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:16668".to_owned()),
            mining_address: m.get_one::<String>("mining-address").cloned().expect("--mining-address is required"),
            threads,
            nonce_batch: m.get_one::<u64>("nonce-batch").copied().unwrap_or(50_000),
            genome_pow_activation_daa_score: m
                .get_one::<u64>("genome-activation-daa-score")
                .copied()
                .unwrap_or(u64::MAX),
            genome_fragment_size_bytes: m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576),
        }
    }
}

fn cli() -> Command {
    Command::new("genome-miner")
        .about("Xenomorph Genome PoW CPU miner")
        .arg(
            Arg::new("rpcserver")
                .long("rpcserver")
                .short('s')
                .value_name("HOST:PORT")
                .help("gRPC endpoint of the node (default: localhost:16668)"),
        )
        .arg(
            Arg::new("mining-address")
                .long("mining-address")
                .short('a')
                .value_name("ADDRESS")
                .required(true)
                .help("Address to receive the coinbase reward"),
        )
        .arg(
            Arg::new("threads")
                .long("threads")
                .short('t')
                .value_name("N")
                .value_parser(clap::value_parser!(usize))
                .help("Mining threads (default: logical CPUs)"),
        )
        .arg(
            Arg::new("nonce-batch")
                .long("nonce-batch")
                .value_name("N")
                .value_parser(clap::value_parser!(u64))
                .default_value("50000")
                .help("Nonces tried per rayon task before re-checking for a new template"),
        )
        .arg(
            Arg::new("genome-activation-daa-score")
                .long("genome-activation-daa-score")
                .value_name("SCORE")
                .value_parser(clap::value_parser!(u64))
                .help("DAA score at which Genome PoW activates (default: u64::MAX — never)"),
        )
        .arg(
            Arg::new("genome-fragment-size")
                .long("genome-fragment-size")
                .value_name("BYTES")
                .value_parser(clap::value_parser!(u32))
                .default_value("1048576")
                .help("Genome fragment size in bytes (must match network params)"),
        )
}

// ── Miner state ───────────────────────────────────────────────────────────────

struct MinerState {
    config: Config,
    loader: Arc<dyn GenomeDatasetLoader>,
    /// Monotonically increasing counter used to detect when the template has changed.
    template_generation: AtomicU64,
    /// Hash of the current template's `accepted_id_merkle_root` (used as a cheap
    /// change-detection key without storing the full template outside the lock).
    template_id: std::sync::Mutex<Option<kaspa_hashes::Hash>>,
    found: AtomicBool,
}

impl MinerState {
    fn new(config: Config) -> Self {
        let epoch_seed = kaspa_hashes::Hash::from_bytes([0u8; 32]);
        let inner_loader = SyntheticLoader::new(config.genome_fragment_size_bytes, epoch_seed);
        let loader: Arc<dyn GenomeDatasetLoader> = Arc::new(CachedLoader::new(inner_loader, 256));
        Self {
            config,
            loader,
            template_generation: AtomicU64::new(0),
            template_id: std::sync::Mutex::new(None),
            found: AtomicBool::new(false),
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    kaspa_core::log::init_logger(None, "info");
    let config = Config::parse();

    let url = format!("grpc://{}", config.rpcserver);
    info!("Connecting to node at {url}");

    let rpc = Arc::new(
        GrpcClient::connect(url)
            .await
            .expect("Failed to connect to node"),
    );
    info!("Connected");

    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(config.mining_address.as_str()).expect("Invalid --mining-address");
    let state = Arc::new(MinerState::new(config));

    // Build rayon pool with requested thread count
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(state.config.threads)
        .build()
        .expect("Failed to build thread pool");

    let mut total_hashes: u64 = 0;
    let mut report_timer = Instant::now();

    loop {
        // ── Fetch template ────────────────────────────────────────────────────
        let resp = match rpc
            .get_block_template_call(
                None,
                GetBlockTemplateRequest::new(pay_address.clone(), vec![]),
            )
            .await
        {
            Ok(r) => r,
            Err(e) => {
                warn!("get_block_template: {e}");
                sleep(Duration::from_secs(1)).await;
                continue;
            }
        };

        let rpc_block: RpcRawBlock = resp.block;

        // Cheap change detection: skip if same template as last iteration
        let current_id = rpc_block.header.accepted_id_merkle_root;
        if !resp.is_synced {
            warn!("Node not synced — mined block unlikely to be accepted");
        }
        {
            let mut guard = state.template_id.lock().unwrap();
            if *guard == Some(current_id) {
                sleep(Duration::from_millis(200)).await;
                continue;
            }
            *guard = Some(current_id);
        }

        // Convert RpcBlock header to consensus Header for PoW state construction
        let header: Header = (&rpc_block.header).into();
        let genome_active = header.daa_score >= state.config.genome_pow_activation_daa_score;
        let gen = state.template_generation.fetch_add(1, Ordering::Relaxed) + 1;
        state.found.store(false, Ordering::Relaxed);

        info!(
            "New template daa={} bits={:#010x} genome={}",
            header.daa_score,
            header.bits,
            genome_active
        );

        // ── Solve PoW ─────────────────────────────────────────────────────────
        let batch = state.config.nonce_batch;
        let mut nonce_base: u64 = 0;

        let solution: Option<u64> = 'search: loop {
            if state.template_generation.load(Ordering::Relaxed) != gen {
                break 'search None; // Stale template
            }

            let range_start = nonce_base;
            let range_end = nonce_base.saturating_add(batch * state.config.threads as u64);
            nonce_base = range_end;

            // Partition into per-thread chunks and run in rayon pool
            let header_ref = &header;
            let loader_ref = state.loader.as_ref();
            let frag_size = state.config.genome_fragment_size_bytes;

            let winning = pool.install(|| {
                (0..state.config.threads as u64)
                    .into_par_iter()
                    .find_map_first(|thread_id| {
                        let start = range_start + thread_id * batch;
                        let end = start + batch;
                        try_nonce_range(header_ref, start, end, genome_active, frag_size, loader_ref)
                    })
            });

            total_hashes += batch * state.config.threads as u64;

            if let Some(nonce) = winning {
                break 'search Some(nonce);
            }

            // Periodically report hashrate
            if report_timer.elapsed() >= Duration::from_secs(10) {
                let hps = total_hashes as f64 / report_timer.elapsed().as_secs_f64();
                info!("Hashrate: {:.2} kH/s", hps / 1000.0);
                total_hashes = 0;
                report_timer = Instant::now();
            }

            // Yield to allow tokio to poll for new templates
            tokio::task::yield_now().await;
        };

        // ── Submit solution ───────────────────────────────────────────────────
        if let Some(winning_nonce) = solution {
            let solved_raw = build_raw_block(&rpc_block, winning_nonce);
            match rpc.submit_block(solved_raw, false).await {
                Ok(resp) => {
                    info!("Block submitted: {:?}", resp.report);
                }
                Err(e) => {
                    warn!("submit_block failed: {e}");
                }
            }
        }
    }
}

// ── PoW search helpers ────────────────────────────────────────────────────────

/// Tries all nonces in `[start, end)` using either Genome PoW or legacy KHeavyHash.
/// Returns the first winning nonce or `None` if none found in range.
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
