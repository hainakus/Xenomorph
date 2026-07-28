use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use kaspa_consensus_core::network::NetworkType;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

use xenom_miner::block::BlockBuilder;
use xenom_miner::cli::gpu_args::GpuArgs;
use xenom_miner::config::MinerConfig;
use xenom_miner::lora::LoraConfig;
use xenom_miner::model::DnaBert2Config;
use xenom_miner::model_client::{fetch_model_checkpoint, ModelBundle, ModelCache};
use xenom_miner::prover::{PublicInputs, ZkProver};
use xenom_miner::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch, TrainingBlock};
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::tokenizer::DnaTokenizer;
use xenom_miner::trainer::{
    CpuTrainer, GpuBackend, GpuTrainer, GradientUpdate, Mgm1MultiGpuTrainer, Mgm1Trainer, MockTrainer, MultiGpuConfig,
    MultiGpuTrainer, Trainer, TrainingResult,
};
use xenom_miner::wallet::{validate_address, WalletManager};

const DEFAULT_RPC_URL: &str = "ws://xeno-node:17110";
const DEFAULT_MODEL_ID: &str = "xeno/mgm-1";
const DEFAULT_THREADS: usize = 4;
const DEFAULT_DATA_DIR: &str = "~/.xenom-miner";
const BLOCK_REWARD: u64 = 100;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const RETRY_DELAY: Duration = Duration::from_secs(2);
/// Canonical GRCh38 `.xenom` archive merkle root used for real human-genome
/// DNABERT-2 training on all networks (mainnet, testnet, devnet, simnet).
const HUMAN_GENOME_MERKLE_ROOT: &str = "577126c448d24d132ba77436517a7db2203d6fce0cd81e2b84db39875d43ee80";

/// Shared, async-mutex protected RPC handle. `None` means disconnected and a
/// reconnect should be attempted before the next call.
type SharedRpc = Arc<tokio::sync::Mutex<Option<XenomRpcClient>>>;

#[derive(Parser, Debug)]
#[command(name = "xenom-miner", version, about = "UsefulPoW Miner for Xenomorph")]
struct Args {
    /// Miner wallet address (optional; derived from stored wallet if omitted).
    #[arg(long, default_value = "")]
    wallet: String,

    /// WebSocket RPC URL of the Xenomorph seed node.
    #[arg(long, default_value = DEFAULT_RPC_URL)]
    rpc_url: String,

    /// Model ID to train.
    #[arg(long, default_value = DEFAULT_MODEL_ID)]
    model_id: String,

    /// Number of CPU threads to use for real training.
    #[arg(long, default_value_t = DEFAULT_THREADS)]
    threads: usize,

    /// Trainer backend to use: mock, cpu, dnabert2, mgm1, gpu, cuda, rocm, or metal.
    #[arg(long, value_parser = ["mock", "cpu", "dnabert2", "mgm1", "gpu", "cuda", "rocm", "metal"], default_value = "mock")]
    trainer: String,

    /// Deprecated alias for --trainer=mock.
    #[arg(long = "mock-mode", visible_alias = "mock", hide = true)]
    mock: bool,

    /// Legacy GPU device ordinal. Used only when --gpus is not provided.
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,

    /// Multi-GPU training options (gpus, micro-batch-size, gradient accumulation, fp16, etc.).
    #[clap(flatten)]
    gpu: GpuArgs,

    /// Network to mine on. Used to derive the canonical genome merkle root
    /// from consensus parameters. Ignored when --genome-merkle is provided.
    #[arg(long, value_parser = ["mainnet", "testnet", "devnet", "simnet"])]
    network: Option<String>,

    /// Optional genome archive merkle root (hex). When set, the miner requests
    /// genome-backed DNABERT-2 training batches instead of synthetic ones.
    /// Overrides the merkle root derived from --network.
    #[arg(long)]
    genome_merkle: Option<String>,

    /// Number of DNA sequences to request per genome batch.
    /// With 5 GPUs × micro-batch 4 × gradient-accumulation 8 this gives an
    /// effective batch of 160 sequences (≈1.5k masked labels for 512 bp).
    #[arg(long, default_value_t = 160)]
    genome_batch_size: usize,

    /// Do not submit mined blocks; useful for local testing.
    #[arg(long)]
    dry_run: bool,

    /// Directory for wallet and configuration files.
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: String,

    /// Directory to cache downloaded model checkpoints.
    /// Defaults to `<data_dir>/models`.
    #[arg(long)]
    models_dir: Option<String>,

    /// Password used to encrypt the wallet file.
    #[arg(long, default_value = "", env = "XENOM_WALLET_PASSWORD")]
    password: String,
}

fn expand_tilde(path: &str) -> PathBuf {
    if let Some(stripped) = path.strip_prefix("~/") {
        dirs::home_dir().map(|home| home.join(stripped)).unwrap_or_else(|| PathBuf::from(path))
    } else {
        PathBuf::from(path)
    }
}

async fn ensure_rpc_connection(rpc_client: &SharedRpc, rpc_url: &str, dry_run: bool) -> Result<()> {
    {
        let guard = rpc_client.lock().await;
        if guard.is_some() {
            return Ok(());
        }
    }

    let max_attempts = 10;
    let mut delay = Duration::from_secs(1);

    for attempt in 1..=max_attempts {
        let mut client = XenomRpcClient::new(rpc_url.to_string());
        match timeout(CONNECT_TIMEOUT, client.connect()).await {
            Ok(Ok(())) => {
                info!("Connected to {}", rpc_url);
                *rpc_client.lock().await = Some(client);
                return Ok(());
            }
            Ok(Err(e)) => {
                if dry_run {
                    warn!("RPC not available ({}), running dry-run with local batches", e);
                    return Ok(());
                }
                warn!("Failed to connect to {} (attempt {}/{}): {}", rpc_url, attempt, max_attempts, e);
            }
            Err(_) => {
                if dry_run {
                    warn!("RPC connection timed out, running dry-run with local batches");
                    return Ok(());
                }
                warn!("Timed out connecting to {} (attempt {}/{})", rpc_url, attempt, max_attempts);
            }
        }

        if attempt < max_attempts {
            tokio::time::sleep(delay).await;
            delay = (delay * 2).min(Duration::from_secs(30));
        }
    }

    bail!("Failed to connect to {} after {} attempts", rpc_url, max_attempts)
}

async fn get_batch(rpc_client: &SharedRpc, model_id: &str) -> Option<TrainingBatch> {
    let mut guard = rpc_client.lock().await;
    if let Some(client) = guard.as_mut() {
        match client.get_training_batch(model_id).await {
            Ok(Some(batch)) => return Some(batch),
            Ok(None) => {
                warn!("No training batch available from the node");
            }
            Err(e) => {
                warn!("Failed to fetch training batch: {}", e);
            }
        }
    }
    None
}

fn parse_genome_merkle(hex_str: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(hex_str.trim()).context("Invalid --genome-merkle hex string")?;
    if bytes.len() != 32 {
        bail!("--genome-merkle must be exactly 32 bytes (64 hex chars), got {}", bytes.len());
    }
    let mut root = [0u8; 32];
    root.copy_from_slice(&bytes);
    Ok(root)
}

async fn get_genome_batch(
    rpc_client: &SharedRpc,
    genome_merkle: [u8; 32],
    model_id: &str,
    batch_size: usize,
) -> Option<GenomeTrainingBatchMsg> {
    let mut guard = rpc_client.lock().await;
    if let Some(client) = guard.as_mut() {
        match client.get_genome_training_batch(genome_merkle, model_id, batch_size).await {
            Ok(msg) => return Some(msg),
            Err(e) => {
                warn!("Failed to fetch genome training batch: {}", e);
            }
        }
    }
    None
}

/// A training batch variant used by the main mining loop.
enum MinerBatch {
    Standard(TrainingBatch),
    Genome(GenomeTrainingBatchMsg),
}

impl MinerBatch {
    fn batch_id(&self) -> u64 {
        match self {
            MinerBatch::Standard(b) => b.batch_id,
            MinerBatch::Genome(msg) => msg.batch.batch_id,
        }
    }

    fn base_checkpoint(&self) -> [u8; 32] {
        match self {
            MinerBatch::Standard(b) => b.base_checkpoint,
            MinerBatch::Genome(msg) => msg.base_checkpoint,
        }
    }

    fn train(self, trainer: Arc<dyn Trainer>) -> Result<(TrainingResult, Option<GradientUpdate>)> {
        match self {
            MinerBatch::Standard(batch) => trainer.train_with_gradients(&batch),
            MinerBatch::Genome(msg) => trainer.train_genome_with_gradients(&msg),
        }
    }
}

async fn fetch_miner_batch(
    rpc_client: &SharedRpc,
    genome_merkle: Option<[u8; 32]>,
    model_id: &str,
    genome_batch_size: usize,
    block_number: u64,
    dry_run: bool,
) -> Option<MinerBatch> {
    if let Some(merkle) = genome_merkle {
        get_genome_batch(rpc_client, merkle, model_id, genome_batch_size).await.map(MinerBatch::Genome)
    } else {
        match get_batch(rpc_client, model_id).await {
            Some(b) => Some(MinerBatch::Standard(b)),
            None if dry_run => Some(MinerBatch::Standard(make_local_batch(model_id, block_number))),
            None => None,
        }
    }
}

fn make_local_batch(model_id: &str, block_number: u64) -> TrainingBatch {
    let mut checkpoint = [0u8; 32];
    checkpoint[..8].copy_from_slice(&block_number.to_le_bytes());
    TrainingBatch {
        batch_id: block_number,
        model_id: model_id.to_string(),
        base_checkpoint: checkpoint,
        data_indices: (0..4).collect(),
        target_improvement: 0.01,
        learning_rate: 0.01,
    }
}

async fn maybe_reload_base(
    trainer: &Arc<dyn Trainer>,
    rpc_client: &SharedRpc,
    model_cache: &ModelCache,
    model_id: &str,
    base_checkpoint: [u8; 32],
) -> Result<()> {
    if trainer.current_base_checkpoint() == Some(base_checkpoint) {
        return Ok(());
    }

    info!(
        "Base checkpoint changed from {:?} to {}; reloading model",
        trainer.current_base_checkpoint().map(hex::encode),
        hex::encode(base_checkpoint)
    );

    let mut guard = rpc_client.lock().await;
    let client = guard.as_mut().context("No RPC connection to reload base checkpoint")?;
    let bundle = fetch_model_checkpoint_with_retry(client, model_id, model_cache)
        .await
        .context("Failed to fetch new base checkpoint from seed-node")?;
    drop(guard);

    trainer.load_base_checkpoint(bundle.base_checkpoint, &bundle.weights)?;
    Ok(())
}

async fn submit_block(rpc_client: &SharedRpc, block: TrainingBlock) -> Result<[u8; 32]> {
    let mut guard = rpc_client.lock().await;
    let client = guard.as_mut().context("No RPC connection to submit block")?;
    match client.submit_block(block).await {
        Ok(hash) => Ok(hash),
        Err(e) => {
            let msg = e.to_string().to_lowercase();
            if !(msg.contains("base checkpoint") && msg.contains("active model weights hash")) {
                // Connection-level or unknown error; force a reconnect on the next attempt.
                *guard = None;
            }
            Err(anyhow::anyhow!("Failed to submit block: {}", e))
        }
    }
}

async fn submit_gradients(gradient_client: &SharedRpc, update: GradientUpdate) -> Result<Option<[u8; 32]>> {
    let mut guard = gradient_client.lock().await;
    let client = guard.as_mut().context("No RPC connection to submit gradients")?;
    match client.submit_gradients(update).await {
        Ok(res) => Ok(res),
        Err(e) => {
            *guard = None;
            Err(anyhow::anyhow!("Failed to submit gradients: {}", e))
        }
    }
}

async fn load_trainer(
    rpc_client: &SharedRpc,
    model_id: &str,
    cache: &ModelCache,
    backend: GpuBackend,
    gpu_config: MultiGpuConfig,
    threads: usize,
    dry_run: bool,
) -> Result<Arc<dyn Trainer>> {
    if dry_run {
        let guard = rpc_client.lock().await;
        if guard.is_none() {
            warn!("Dry-run without RPC connection; falling back to mock trainer");
            return Ok(Arc::new(MockTrainer::new()));
        }
    }

    let mut guard = rpc_client.lock().await;
    let client = guard.as_mut().context("No RPC connection to fetch model checkpoint")?;

    // Retry until the seed-node has the model loaded. In Docker Compose the
    // service dependency should already guarantee this, but the retry makes
    // manual/standalone runs robust against slow model downloads.
    let ModelBundle { model_id, base_checkpoint, config, tokenizer, weights, .. } =
        fetch_model_checkpoint_with_retry(client, model_id, cache).await.context("Failed to fetch model checkpoint from seed-node")?;

    if model_id.contains("mgm-1") {
        if gpu_config.gpus.len() > 1 {
            let trainer =
                Mgm1MultiGpuTrainer::new(&model_id, &config, &tokenizer, weights, base_checkpoint, gpu_config, backend, 1e-3, threads)
                    .context("Failed to initialize multi-GPU MGM-1 trainer")?;
            info!("Loaded multi-GPU MGM-1 model checkpoint for {}", model_id);
            info!("Trainer device: {:?}", trainer.device_info());
            return Ok(Arc::new(trainer));
        }

        let device_index = gpu_config.gpus.first().copied().unwrap_or(0);
        let (device, _, _) = GpuTrainer::select_device(backend, device_index)?;
        let trainer =
            Mgm1Trainer::new(&model_id, &config, &tokenizer, weights, base_checkpoint, device, 1e-4, &gpu_config)
                .context("Failed to initialize MGM-1 trainer")?;
        info!("Loaded MGM-1 model checkpoint for {}", model_id);
        info!("Trainer device: {:?}", trainer.device_info());
        return Ok(Arc::new(trainer));
    }

    let config = DnaBert2Config::from_bytes(&config).context("Failed to parse DNABERT-2 config")?;
    let tokenizer = DnaTokenizer::from_bytes(&tokenizer).context("Failed to parse tokenizer")?;
    let model_id_for_log = model_id.clone();
    let trainer = MultiGpuTrainer::new(model_id, config, weights, tokenizer, gpu_config, backend, threads)
        .context("Failed to initialize multi-GPU DNABERT-2 trainer")?;
    info!("Loaded DNABERT-2 model checkpoint for {}", model_id_for_log);
    info!("Trainer device: {:?}", trainer.device_info());
    Ok(Arc::new(trainer))
}

async fn fetch_model_checkpoint_with_retry(client: &mut XenomRpcClient, model_id: &str, cache: &ModelCache) -> Result<ModelBundle> {
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    let max_attempts = 60;

    for attempt in 1..=max_attempts {
        match fetch_model_checkpoint(client, model_id, cache).await {
            Ok(bundle) => return Ok(bundle),
            Err(e) if attempt < max_attempts => {
                warn!("Model checkpoint not ready (attempt {}/{}): {}", attempt, max_attempts, e);
                interval.tick().await;
            }
            Err(e) => return Err(e),
        }
    }

    bail!("Seed-node did not provide model checkpoint after {} retries", max_attempts)
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let data_dir = expand_tilde(&args.data_dir);
    std::fs::create_dir_all(&data_dir).with_context(|| format!("Failed to create data directory {:?}", data_dir))?;

    let network_type = NetworkType::from_str(args.network.as_deref().unwrap_or("mainnet"))
        .with_context(|| format!("Invalid network: {}", args.network.as_deref().unwrap_or("mainnet")))?;

    let mut config = MinerConfig::load_or_create(&data_dir)?;
    if !args.wallet.is_empty() {
        config.wallet_address = args.wallet.clone();
    }
    if !args.rpc_url.is_empty() {
        config.rpc_url = args.rpc_url.clone();
    }
    if !args.model_id.is_empty() {
        config.model_id = args.model_id.clone();
    }
    let trainer_kind = if args.mock { "mock".to_string() } else { args.trainer };
    config.threads = args.threads;
    config.mock_mode = trainer_kind == "mock";
    config.dry_run = args.dry_run;
    config.data_dir = data_dir.clone();
    config.save(&data_dir)?;

    let models_dir = args.models_dir.map(|p| expand_tilde(&p)).unwrap_or_else(|| data_dir.join("models"));
    let model_cache = ModelCache::new(&models_dir);

    let wallet = Arc::new(
        WalletManager::load_or_create(&data_dir, &args.password, network_type).with_context(|| "Failed to load or create wallet")?,
    );
    let miner_address = if config.wallet_address.is_empty() {
        wallet.address().to_string()
    } else {
        validate_address(&config.wallet_address, network_type).with_context(|| "Invalid --wallet address for the selected network")?;
        config.wallet_address.clone()
    };
    info!("Miner address: {}", miner_address);

    let rpc_client: SharedRpc = Arc::new(tokio::sync::Mutex::new(None));
    let gradient_client: SharedRpc = Arc::new(tokio::sync::Mutex::new(None));
    ensure_rpc_connection(&rpc_client, &config.rpc_url, config.dry_run).await?;
    if !config.dry_run {
        ensure_rpc_connection(&gradient_client, &config.rpc_url, config.dry_run).await?;
    }

    let trainer: Arc<dyn Trainer> = match trainer_kind.as_str() {
        "mock" => {
            info!("Using mock trainer");
            Arc::new(MockTrainer::new())
        }
        "cpu" => {
            info!("Using legacy Candle CPU trainer");
            Arc::new(CpuTrainer::new(config.threads)?)
        }
        "dnabert2" | "mgm1" | "gpu" | "cuda" | "rocm" | "metal" => {
            let backend = match trainer_kind.as_str() {
                "dnabert2" | "mgm1" | "gpu" => GpuBackend::Auto,
                "cuda" => GpuBackend::Cuda,
                "rocm" => GpuBackend::Rocm,
                "metal" => GpuBackend::Metal,
                _ => unreachable!(),
            };

            let gpus = if args.gpu.gpus.is_empty() { vec![args.gpu_device] } else { args.gpu.gpus.clone() };
            let lora_config = if args.gpu.lora {
                let target_modules = if args.gpu.lora_target_modules.is_empty() {
                    LoraConfig::default_target_modules()
                } else {
                    args.gpu.lora_target_modules.iter().cloned().collect()
                };
                Some(LoraConfig {
                    rank: args.gpu.lora_rank,
                    alpha: args.gpu.lora_alpha,
                    dropout: args.gpu.lora_dropout,
                    target_modules,
                })
            } else {
                None
            };
            let gpu_config = MultiGpuConfig {
                gpus,
                micro_batch_size: args.gpu.micro_batch_size,
                gradient_accumulation_steps: args.gpu.gradient_accumulation,
                use_mixed_precision: args.gpu.fp16,
                use_gradient_checkpointing: args.gpu.gradient_checkpointing,
                zero_optimization: args.gpu.zero,
                gradient_top_k_ratio: args.gpu.gradient_top_k_ratio,
                lora_config,
                max_seq_len: args.gpu.max_seq_len,
            };
            gpu_config.validate()?;

            info!("Using model trainer with {:?} backend and config {:?}", backend, gpu_config);
            load_trainer(&rpc_client, &config.model_id, &model_cache, backend, gpu_config, config.threads, config.dry_run).await?
        }
        other => bail!("Unknown trainer: {}. Use mock, cpu, dnabert2, mgm1, gpu, cuda, rocm, or metal.", other),
    };

    let dna_model_backends = ["dnabert2", "mgm1", "gpu", "cuda", "rocm", "metal"];
    let is_dna_model = dna_model_backends.contains(&trainer_kind.as_str());

    let genome_merkle: Option<[u8; 32]> = if let Some(hex_str) = args.genome_merkle.as_deref() {
        Some(parse_genome_merkle(hex_str)?)
    } else if is_dna_model {
        // DNA trainers always train on the real human GRCh38 genome, regardless
        // of the network selected for the wallet/address prefix.
        info!("Using canonical human genome merkle root for DNABERT-2 training");
        Some(parse_genome_merkle(HUMAN_GENOME_MERKLE_ROOT)?)
    } else {
        None
    };
    if genome_merkle.is_some() && !is_dna_model {
        warn!("--genome-merkle is only supported with --trainer=dnabert2/gpu/cuda/rocm/metal; genome training will likely fail");
    }

    let prover = ZkProver::new();
    let mut block_builder = BlockBuilder::new(miner_address.clone());

    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner().tick_chars("⠁⠂⠄⡀⢀⠠⠐⠈ ").template("{spinner} {msg}").expect("valid spinner template"),
    );

    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);
    let start = Instant::now();
    let mut blocks_submitted: u64 = 0;
    let mut total_reward: u64 = 0;
    let mut block_number: u64 = 0;

    let mut current_batch: Option<MinerBatch> = None;
    let mut next_batch_handle: Option<tokio::task::JoinHandle<Option<MinerBatch>>> = None;
    let mut gradient_handle: Option<tokio::task::JoinHandle<Result<Option<[u8; 32]>>>> = None;

    loop {
        progress.set_message(format!("block {} | preparing | reward {}", block_number, total_reward));

        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("Shutdown signal received; finishing current work");
                break;
            }

            result = async {
                // Ensure RPC connections are alive.
                if !config.dry_run {
                    if let Err(e) = ensure_rpc_connection(&rpc_client, &config.rpc_url, config.dry_run).await {
                        warn!("RPC reconnect failed: {}", e);
                        tokio::time::sleep(RETRY_DELAY).await;
                        return Ok::<_, anyhow::Error>(());
                    }
                    if let Err(e) = ensure_rpc_connection(&gradient_client, &config.rpc_url, config.dry_run).await {
                        warn!("Gradient RPC reconnect failed: {}", e);
                        tokio::time::sleep(RETRY_DELAY).await;
                        return Ok::<_, anyhow::Error>(());
                    }
                }

                // Make sure we have a batch to train on.
                let batch = match current_batch.take() {
                    Some(b) => b,
                    None => match fetch_miner_batch(&rpc_client, genome_merkle, &config.model_id, args.genome_batch_size, block_number, config.dry_run).await {
                        Some(b) => b,
                        None => {
                            warn!("No batch available; retrying");
                            tokio::time::sleep(RETRY_DELAY).await;
                            return Ok::<_, anyhow::Error>(());
                        }
                    },
                };

                // Hot-reload the model if the base checkpoint changed.
                if let Err(e) = maybe_reload_base(
                    &trainer,
                    &rpc_client,
                    &model_cache,
                    &config.model_id,
                    batch.base_checkpoint(),
                )
                .await {
                    warn!("Failed to reload base checkpoint: {:#}", e);
                    current_batch = Some(batch);
                    tokio::time::sleep(RETRY_DELAY).await;
                    return Ok::<_, anyhow::Error>(());
                }

                // If the seed-node moved to a new active checkpoint, the prefetched
                // batch is stale. Discard it and refetch from the new base.
                if trainer.current_base_checkpoint() != Some(batch.base_checkpoint()) {
                    info!("Seed-node active checkpoint changed; discarding prefetched batch and refetching");
                    if let Some(h) = next_batch_handle.take() {
                        h.abort();
                    }
                    current_batch = None;
                    return Ok::<_, anyhow::Error>(());
                }

                // Start fetching the next batch in the background while this one trains.
                let pending_fetch_rpc = Arc::clone(&rpc_client);
                let pending_model_id = config.model_id.clone();
                let pending_block_number = block_number + 1;
                let pending_dry_run = config.dry_run;
                let pending_genome_merkle = genome_merkle;
                let pending_genome_batch_size = args.genome_batch_size;
                let pending_next_batch = next_batch_handle.take().unwrap_or_else(|| {
                    tokio::spawn(async move {
                        fetch_miner_batch(&pending_fetch_rpc, pending_genome_merkle, &pending_model_id, pending_genome_batch_size, pending_block_number, pending_dry_run).await
                    })
                });

                // Train the current batch off the async runtime.
                let batch_id = batch.batch_id();
                let trainer = Arc::clone(&trainer);
                let train_handle = tokio::task::spawn_blocking(move || batch.train(trainer));

                let train_result = train_handle.await;
                let next_batch = pending_next_batch.await;

                let (result, gradient_update) = match train_result {
                    Ok(Ok(v)) => v,
                    Ok(Err(e)) => {
                        warn!("Training failed: {}", e);
                        current_batch = next_batch.ok().flatten();
                        return Ok::<_, anyhow::Error>(());
                    }
                    Err(e) => {
                        warn!("Training task panicked: {}", e);
                        current_batch = next_batch.ok().flatten();
                        return Ok::<_, anyhow::Error>(());
                    }
                };

                let next_batch = next_batch.ok().flatten();

                // Build and submit the training block.
                let public_inputs = PublicInputs {
                    model_id: config.model_id.clone(),
                    batch_id,
                    loss_before: result.loss_before,
                    loss_after: result.loss_after,
                    gradients_commitment: result.gradients_commitment,
                    base_checkpoint: result.base_checkpoint,
                };
                let zk_proof = prover.generate_proof(&result, &public_inputs)?;
                let mut block = block_builder.build_block(&config.model_id, &result, zk_proof, [0u8; 32])?;
                wallet.sign_block(&mut block)?;

                let built_block_number = block.header.block_number;
                let mut stale_base = false;
                if config.dry_run {
                    info!(
                        "Dry-run block {} built (merkle {})",
                        built_block_number,
                        hex::encode(&block.header.merkle_root[..8])
                    );
                } else {
                    match submit_block(&rpc_client, block).await {
                        Ok(block_hash) => {
                            block_builder.set_prev_block(block_hash, built_block_number);
                            blocks_submitted += 1;
                            total_reward += BLOCK_REWARD;
                            info!(
                                "Submitted block {}: {} | loss {:.6} -> {:.6} | improvement {:.6}",
                                built_block_number,
                                hex::encode(block_hash),
                                result.loss_before,
                                result.loss_after,
                                result.loss_before - result.loss_after
                            );
                        }
                        Err(e) => {
                            warn!("Failed to submit block: {}", e);
                            if e.to_string().to_lowercase().contains("base checkpoint does not match active model weights hash") {
                                warn!("Block rejected because the active model checkpoint moved; discarding prefetched batches and refetching");
                                stale_base = true;
                            }
                        }
                    }
                }

                // Await previous gradient submission so we don't queue
                // unbounded gradient payloads in memory.
                if let Some(handle) = gradient_handle.take() {
                    match handle.await {
                        Ok(Ok(Some(new_checkpoint))) => info!("FedAvg produced new checkpoint: {}", hex::encode(new_checkpoint)),
                        Ok(Ok(None)) => info!("Gradient update accepted; waiting for more participants"),
                        Ok(Err(e)) => {
                            if e.to_string().to_lowercase().contains("stale") {
                                warn!("Previous gradient submission rejected (stale base): {}", e);
                                stale_base = true;
                            } else {
                                warn!("Previous gradient submission failed: {}", e);
                            }
                        }
                        Err(e) => warn!("Gradient submission task panicked: {}", e),
                    }
                }

                // Submit the gradient update in the background; it will overlap
                // with the next batch's training and batch fetch.
                if let Some(update) = gradient_update {
                    let gradient_client = Arc::clone(&gradient_client);
                    gradient_handle = Some(tokio::spawn(async move {
                        submit_gradients(&gradient_client, update).await
                    }));
                }

                // If a gradient was rejected as stale, the prefetched batches are
                // likely stale too. Discard them and refetch from the new base.
                if stale_base {
                    warn!("Base checkpoint moved while gradient was in flight; discarding prefetched batches and refetching");
                    if let Some(h) = next_batch_handle.take() {
                        h.abort();
                    }
                    current_batch = None;
                    block_number += 1;
                    let elapsed = start.elapsed().as_secs_f64().max(1.0);
                    let blocks_per_min = block_number as f64 / elapsed * 60.0;
                    progress.set_message(format!(
                        "blocks {} | stale base, refetching | {:.1} blocks/min",
                        block_number, blocks_per_min
                    ));
                    progress.tick();
                    return Ok::<_, anyhow::Error>(());
                }

                // Queue the fetch for the batch after next, then advance the loop.
                let future_fetch_rpc = Arc::clone(&rpc_client);
                let future_model_id = config.model_id.clone();
                let future_block_number = block_number + 2;
                let future_dry_run = config.dry_run;
                let future_genome_merkle = genome_merkle;
                let future_genome_batch_size = args.genome_batch_size;
                current_batch = next_batch;
                next_batch_handle = Some(tokio::spawn(async move {
                    fetch_miner_batch(&future_fetch_rpc, future_genome_merkle, &future_model_id, future_genome_batch_size, future_block_number, future_dry_run).await
                }));

                block_number += 1;
                let elapsed = start.elapsed().as_secs_f64().max(1.0);
                let blocks_per_min = block_number as f64 / elapsed * 60.0;
                progress.set_message(format!(
                    "blocks {} | loss {:.4}->{:.4} | reward {} | {:.1} blocks/min",
                    block_number,
                    result.loss_before,
                    result.loss_after,
                    total_reward,
                    blocks_per_min
                ));
                progress.tick();

                Ok::<_, anyhow::Error>(())
            } => {
                if let Err(e) = result {
                    error!("Mining error: {:?}", e);
                    tokio::time::sleep(RETRY_DELAY).await;
                }
            }
        }
    }

    progress.finish_with_message(format!("Finished {} blocks ({} reward)", block_number, total_reward));
    info!("Miner shut down gracefully");
    Ok(())
}
