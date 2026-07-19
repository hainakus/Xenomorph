use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use kaspa_consensus_core::config::params::Params;
use kaspa_consensus_core::network::NetworkType;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

use xenom_miner::block::BlockBuilder;
use xenom_miner::config::MinerConfig;
use xenom_miner::model::DnaBert2Config;
use xenom_miner::model_client::{fetch_model_checkpoint, ModelBundle};
use xenom_miner::prover::{PublicInputs, ZkProver};
use xenom_miner::rpc::messages::{GenomeTrainingBatchMsg, TrainingBatch};
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::tokenizer::DnaTokenizer;
use xenom_miner::trainer::{CpuTrainer, GpuBackend, GpuTrainer, MockTrainer, Trainer};
use xenom_miner::wallet::WalletManager;

const DEFAULT_RPC_URL: &str = "ws://xeno-seed:17110";
const DEFAULT_MODEL_ID: &str = "multimolecule/dnabert2";
const DEFAULT_THREADS: usize = 4;
const DEFAULT_DATA_DIR: &str = "~/.xenom-miner";
const BLOCK_REWARD: u64 = 100;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const RETRY_DELAY: Duration = Duration::from_secs(2);

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

    /// Trainer backend to use: mock, cpu, dnabert2, gpu, cuda, rocm, or metal.
    #[arg(long, value_parser = ["mock", "cpu", "dnabert2", "gpu", "cuda", "rocm", "metal"], default_value = "mock")]
    trainer: String,

    /// Deprecated alias for --trainer=mock.
    #[arg(long = "mock-mode", visible_alias = "mock", hide = true)]
    mock: bool,

    /// GPU device ordinal to use when --trainer is gpu/cuda/metal.
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,

    /// Use FP16 mixed precision on supported GPU backends (CUDA/Metal).
    #[arg(long)]
    fp16: bool,

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
    #[arg(long, default_value_t = 4)]
    genome_batch_size: usize,

    /// Do not submit mined blocks; useful for local testing.
    #[arg(long)]
    dry_run: bool,

    /// Directory for wallet and configuration files.
    #[arg(long, default_value = DEFAULT_DATA_DIR)]
    data_dir: String,

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

async fn ensure_rpc_connection(rpc_client: &mut Option<XenomRpcClient>, rpc_url: &str, dry_run: bool) -> Result<()> {
    if rpc_client.is_some() {
        return Ok(());
    }

    let mut client = XenomRpcClient::new(rpc_url.to_string());
    match timeout(CONNECT_TIMEOUT, client.connect()).await {
        Ok(Ok(())) => {
            info!("Connected to {}", rpc_url);
            *rpc_client = Some(client);
            Ok(())
        }
        Ok(Err(e)) => {
            if dry_run {
                warn!("RPC not available ({}), running dry-run with local batches", e);
                Ok(())
            } else {
                bail!("Failed to connect to {}: {}", rpc_url, e)
            }
        }
        Err(_) => {
            if dry_run {
                warn!("RPC connection timed out, running dry-run with local batches");
                Ok(())
            } else {
                bail!("Timed out connecting to {}", rpc_url)
            }
        }
    }
}

async fn get_batch(rpc_client: &mut Option<XenomRpcClient>, model_id: &str) -> Option<TrainingBatch> {
    if let Some(client) = rpc_client.as_mut() {
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
    rpc_client: &mut Option<XenomRpcClient>,
    genome_merkle: [u8; 32],
    model_id: &str,
    batch_size: usize,
) -> Option<GenomeTrainingBatchMsg> {
    if let Some(client) = rpc_client.as_mut() {
        match client.get_genome_training_batch(genome_merkle, model_id, batch_size).await {
            Ok(msg) => return Some(msg),
            Err(e) => {
                warn!("Failed to fetch genome training batch: {}", e);
            }
        }
    }
    None
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

async fn load_trainer(
    rpc_client: &mut Option<XenomRpcClient>,
    model_id: &str,
    backend: GpuBackend,
    gpu_device: usize,
    fp16: bool,
    threads: usize,
    dry_run: bool,
) -> Result<Arc<dyn Trainer>> {
    if dry_run && rpc_client.is_none() {
        warn!("Dry-run without RPC connection; falling back to mock trainer");
        return Ok(Arc::new(MockTrainer::new()));
    }

    let client = rpc_client.as_mut().context("No RPC connection to fetch model checkpoint")?;

    // Retry until the seed-node has the model loaded. In Docker Compose the
    // service dependency should already guarantee this, but the retry makes
    // manual/standalone runs robust against slow model downloads.
    let ModelBundle { model_id, config, tokenizer, weights, .. } =
        fetch_model_checkpoint_with_retry(client, model_id).await
            .context("Failed to fetch model checkpoint from seed-node")?;

    let config = DnaBert2Config::from_bytes(&config).context("Failed to parse DNABERT-2 config")?;
    let tokenizer = DnaTokenizer::from_bytes(&tokenizer).context("Failed to parse tokenizer")?;
    let model_id_for_log = model_id.clone();
    let trainer =
        GpuTrainer::new(config, weights, tokenizer, backend, gpu_device, fp16, threads)
            .context("Failed to initialize DNABERT-2 trainer")?;
    info!("Loaded DNABERT-2 model checkpoint for {}", model_id_for_log);
    info!("Trainer device: {:?}", trainer.device_info());
    Ok(Arc::new(trainer))
}

async fn fetch_model_checkpoint_with_retry(
    client: &mut XenomRpcClient,
    model_id: &str,
) -> Result<ModelBundle> {
    let mut interval = tokio::time::interval(Duration::from_secs(2));
    let max_attempts = 60;

    for attempt in 1..=max_attempts {
        match fetch_model_checkpoint(client, model_id).await {
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

    let wallet =
        Arc::new(WalletManager::load_or_create(&data_dir, &args.password).with_context(|| "Failed to load or create wallet")?);
    let miner_address = if config.wallet_address.is_empty() { wallet.address().to_string() } else { config.wallet_address.clone() };
    info!("Miner address: {}", miner_address);

    let mut rpc_client: Option<XenomRpcClient> = None;
    ensure_rpc_connection(&mut rpc_client, &config.rpc_url, config.dry_run).await?;

    let trainer: Arc<dyn Trainer> = match trainer_kind.as_str() {
        "mock" => {
            info!("Using mock trainer");
            Arc::new(MockTrainer::new())
        }
        "cpu" => {
            info!("Using legacy Candle CPU trainer");
            Arc::new(CpuTrainer::new(config.threads)?)
        }
        "dnabert2" | "gpu" | "cuda" | "rocm" | "metal" => {
            let backend = match trainer_kind.as_str() {
                "dnabert2" | "gpu" => GpuBackend::Auto,
                "cuda" => GpuBackend::Cuda,
                "rocm" => GpuBackend::Rocm,
                "metal" => GpuBackend::Metal,
                _ => unreachable!(),
            };
            info!("Using DNABERT-2 trainer with {:?} GPU backend", backend);
            load_trainer(
                &mut rpc_client,
                &config.model_id,
                backend,
                args.gpu_device,
                args.fp16,
                config.threads,
                config.dry_run,
            )
            .await?
        }
        other => bail!("Unknown trainer: {}. Use mock, cpu, dnabert2, gpu, cuda, rocm, or metal.", other),
    };

    let dna_model_backends = ["dnabert2", "gpu", "cuda", "rocm", "metal"];
    let is_dna_model = dna_model_backends.contains(&trainer_kind.as_str());

    let genome_merkle: Option<[u8; 32]> = if let Some(hex_str) = args.genome_merkle.as_deref() {
        Some(parse_genome_merkle(hex_str)?)
    } else if is_dna_model {
        let network = args.network.as_deref().unwrap_or("mainnet");
        let network_type = NetworkType::from_str(network).with_context(|| format!("Invalid network: {}", network))?;
        let params = Params::from(network_type);
        info!("Using {} genome merkle root: {}", network, params.genome_merkle_root);
        Some(parse_genome_merkle(params.genome_merkle_root)?)
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

    loop {
        progress.set_message(format!("block {} | fetching batch | reward {}", block_number, total_reward));

        tokio::select! {
            biased;
            _ = &mut shutdown => {
                info!("Shutdown signal received; finishing current work");
                break;
            }

            result = async {
                if rpc_client.is_none() && !config.dry_run {
                    tokio::time::sleep(RETRY_DELAY).await;
                    if let Err(e) = ensure_rpc_connection(&mut rpc_client, &config.rpc_url, config.dry_run).await {
                        warn!("RPC reconnect failed: {}", e);
                    }
                }

                let (result, batch_id) = if let Some(merkle) = genome_merkle {
                    let msg = match get_genome_batch(&mut rpc_client, merkle, &config.model_id, args.genome_batch_size).await {
                        Some(msg) => msg,
                        None if config.dry_run => {
                            warn!("Dry-run with --genome-merkle but no RPC; cannot generate genome batch");
                            tokio::time::sleep(RETRY_DELAY).await;
                            return Ok::<_, anyhow::Error>(());
                        }
                        None => {
                            warn!("No genome batch available; retrying");
                            tokio::time::sleep(RETRY_DELAY).await;
                            return Ok::<_, anyhow::Error>(());
                        }
                    };

                    let batch_id = msg.batch.batch_id;
                    let trainer = Arc::clone(&trainer);
                    let result = tokio::task::spawn_blocking(move || trainer.train_genome(&msg))
                        .await
                        .context("Genome training task panicked")??;
                    (result, batch_id)
                } else {
                    let batch = match get_batch(&mut rpc_client, &config.model_id).await {
                        Some(batch) => batch,
                        None if config.dry_run => make_local_batch(&config.model_id, block_number),
                        None => {
                            warn!("No batch available; retrying");
                            tokio::time::sleep(RETRY_DELAY).await;
                            return Ok::<_, anyhow::Error>(());
                        }
                    };

                    let batch_id = batch.batch_id;
                    let batch_for_training = batch.clone();
                    let trainer = Arc::clone(&trainer);
                    let result = tokio::task::spawn_blocking(move || trainer.train(&batch_for_training))
                        .await
                        .context("Training task panicked")??;
                    (result, batch_id)
                };

                let public_inputs = PublicInputs {
                    model_id: config.model_id.clone(),
                    batch_id,
                    loss_before: result.loss_before,
                    loss_after: result.loss_after,
                    gradients_commitment: result.gradients_commitment,
                    base_checkpoint: result.base_checkpoint,
                };

                let zk_proof = prover.generate_proof(&result, &public_inputs)?;
                let mut block = block_builder.build_block(&result, zk_proof, [0u8; 32])?;
                wallet.sign_block(&mut block)?;

                if config.dry_run {
                    info!(
                        "Dry-run block {} built (merkle {})",
                        block.header.block_number,
                        hex::encode(&block.header.merkle_root[..8])
                    );
                } else if let Some(client) = rpc_client.as_mut() {
                    match client.submit_block(block.clone()).await {
                        Ok(block_hash) => {
                            block_builder.set_prev_block(block_hash, block.header.block_number);
                            blocks_submitted += 1;
                            total_reward += BLOCK_REWARD;
                            info!(
                                "Submitted block {}: {}",
                                block.header.block_number,
                                hex::encode(block_hash)
                            );
                        }
                        Err(e) => {
                            warn!("Failed to submit block: {}", e);
                            rpc_client = None;
                        }
                    }
                }

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
