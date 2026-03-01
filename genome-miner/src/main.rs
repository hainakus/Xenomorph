mod gpu;

use std::{
    io::Read,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

use clap::{Arg, ArgMatches, Command};
use kaspa_addresses::Address;
use kaspa_consensus_core::header::Header;
use kaspa_core::{info, warn};
use kaspa_grpc_client::GrpcClient;
use kaspa_pow::genome_pow::{build_merkle_root, fragment_index, fragment_leaf_hash, CachedLoader, GenomeDatasetLoader, GenomePowState, SyntheticLoader};
use kaspa_rpc_core::{
    api::rpc::RpcApi,
    model::message::{GetBlockDagInfoRequest, GetBlockTemplateRequest},
    RpcRawBlock, RpcRawHeader,
};
use kaspa_txscript::pay_to_address_script;
use rayon::prelude::*;
use tokio::time::sleep;

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
                .arg(Arg::new("mainnet").long("mainnet").action(clap::ArgAction::SetTrue).help("Mainnet (genome activation DAA 21_370_801)"))
                .arg(Arg::new("testnet").long("testnet").action(clap::ArgAction::SetTrue).help("Testnet (genome activation DAA 0)"))
                .arg(Arg::new("devnet").long("devnet").action(clap::ArgAction::SetTrue).help("Devnet (genome activation DAA 0)"))
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
            Command::new("gpu")
                .about("Run the GPU miner (wgpu — Metal on Mac, Vulkan elsewhere)")
                .arg(Arg::new("rpcserver").long("rpcserver").short('s').value_name("HOST:PORT").help("gRPC node endpoint (default: localhost:36669)"))
                .arg(Arg::new("mining-address").long("mining-address").short('a').value_name("ADDRESS").required(true).help("Reward address"))
                .arg(Arg::new("batch-size").long("batch-size").value_name("N").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Nonces per GPU dispatch (default: 1M)"))
                .arg(Arg::new("genome-activation-daa-score").long("genome-activation-daa-score").value_name("SCORE").value_parser(clap::value_parser!(u64)).help("DAA score where Genome PoW activates (overrides --mainnet/--testnet/--devnet)"))
                .arg(Arg::new("genome-fragment-size").long("genome-fragment-size").value_name("BYTES").value_parser(clap::value_parser!(u32)).default_value("1048576").help("Fragment size in bytes"))
                .arg(Arg::new("mainnet").long("mainnet").action(clap::ArgAction::SetTrue).help("Mainnet (genome activation DAA 21_370_801)"))
                .arg(Arg::new("testnet").long("testnet").action(clap::ArgAction::SetTrue).help("Testnet (genome activation DAA 0)"))
                .arg(Arg::new("devnet").long("devnet").action(clap::ArgAction::SetTrue).help("Devnet (genome activation DAA 0)"))
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
    loader: Arc<dyn GenomeDatasetLoader>,
    template_generation: AtomicU64,
    template_id: std::sync::Mutex<Option<kaspa_hashes::Hash>>,
    found: AtomicBool,
}

impl MinerState {
    fn new(cfg: MineConfig) -> Self {
        let epoch_seed = kaspa_hashes::Hash::from_bytes([0u8; 32]);
        let inner = SyntheticLoader::new(cfg.genome_fragment_size_bytes, epoch_seed);
        let loader: Arc<dyn GenomeDatasetLoader> = Arc::new(CachedLoader::new(inner, 256));
        Self { cfg, loader, template_generation: AtomicU64::new(0), template_id: std::sync::Mutex::new(None), found: AtomicBool::new(false) }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() {
    kaspa_core::log::init_logger(None, "info,wgpu_core=warn,wgpu_hal=warn,naga=warn");
    let matches = cli().get_matches();
    match matches.subcommand() {
        Some(("mine", m))                => cmd_mine(m).await,
        Some(("suggest-params", m))      => cmd_suggest_params(m).await,
        Some(("compute-merkle-root", m)) => cmd_compute_merkle_root(m),
        Some(("address-to-script", m))   => cmd_address_to_script(m),
        Some(("gpu", m))                 => gpu::cmd_gpu(m).await,
        _ => unreachable!(),
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Resolves the genome PoW activation DAA score from CLI flags.
/// Priority: explicit `--genome-activation-daa-score` > `--mainnet` > `--testnet`/`--devnet` > u64::MAX
fn resolve_activation(m: &ArgMatches) -> u64 {
    if let Some(&score) = m.get_one::<u64>("genome-activation-daa-score") {
        return score;
    }
    if m.get_flag("mainnet") {
        return kaspa_consensus_core::hashing::header::EPOCH_SEED_HASH_ACTIVATION_MAINNET;
    }
    if m.get_flag("testnet") || m.get_flag("devnet") {
        return 0;
    }
    u64::MAX
}

// ── mine ─────────────────────────────────────────────────────────────────────

async fn cmd_mine(m: &ArgMatches) {
    let threads = m.get_one::<usize>("threads").copied().unwrap_or_else(|| rayon::current_num_threads());
    let cfg = MineConfig {
        rpcserver: m.get_one::<String>("rpcserver").cloned().unwrap_or_else(|| "localhost:16668".to_owned()),
        mining_address: m.get_one::<String>("mining-address").cloned().expect("--mining-address required"),
        threads,
        nonce_batch: m.get_one::<u64>("nonce-batch").copied().unwrap_or(50_000),
        genome_pow_activation_daa_score: resolve_activation(m),
        genome_fragment_size_bytes: m.get_one::<u32>("genome-fragment-size").copied().unwrap_or(1_048_576),
    };

    let url = format!("grpc://{}", cfg.rpcserver);
    info!("Connecting to {url}");
    let rpc = Arc::new(GrpcClient::connect(url).await.expect("Failed to connect"));
    info!("Connected — threads={} genome_activation={}", cfg.threads, cfg.genome_pow_activation_daa_score);

    let pay_address: kaspa_rpc_core::RpcAddress =
        Address::try_from(cfg.mining_address.as_str()).expect("Invalid --mining-address");
    let state = Arc::new(MinerState::new(cfg));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(state.cfg.threads).build().expect("rayon pool");

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
            if *guard == Some(current_id) { sleep(Duration::from_millis(200)).await; continue; }
            *guard = Some(current_id);
        }

        let header: Header = (&rpc_block.header).into();
        let genome_active = header.daa_score >= state.cfg.genome_pow_activation_daa_score;
        let gen = state.template_generation.fetch_add(1, Ordering::Relaxed) + 1;
        state.found.store(false, Ordering::Relaxed);
        info!("New template daa={} bits={:#010x} genome={}", header.daa_score, header.bits, genome_active);

        let batch = state.cfg.nonce_batch;
        let mut nonce_base: u64 = 0;
        let solution: Option<u64> = 'search: loop {
            if state.template_generation.load(Ordering::Relaxed) != gen { break 'search None; }
            let range_start = nonce_base;
            nonce_base = nonce_base.saturating_add(batch * state.cfg.threads as u64);
            let frag_size = state.cfg.genome_fragment_size_bytes;
            let loader_ref = state.loader.as_ref();
            let winning = pool.install(|| {
                (0..state.cfg.threads as u64).into_par_iter().find_map_first(|tid| {
                    let start = range_start + tid * batch;
                    try_nonce_range(&header, start, start + batch, genome_active, frag_size, loader_ref)
                })
            });
            total_hashes += batch * state.cfg.threads as u64;
            if let Some(n) = winning { break 'search Some(n); }
            if report_timer.elapsed() >= Duration::from_secs(10) {
                info!("Hashrate: {:.2} kH/s", total_hashes as f64 / report_timer.elapsed().as_secs_f64() / 1000.0);
                total_hashes = 0;
                report_timer = Instant::now();
            }
            tokio::task::yield_now().await;
        };

        if let Some(nonce) = solution {
            match rpc.submit_block(build_raw_block(&rpc_block, nonce), false).await {
                Ok(r)  => info!("Block submitted: {:?}", r.report),
                Err(e) => warn!("submit_block: {e}"),
            }
        }
    }
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

    let total_fragments = (data.len() + fragment_size - 1) / fragment_size;
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
