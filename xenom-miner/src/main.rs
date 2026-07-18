use anyhow::{bail, Context, Result};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::timeout;
use tracing::{error, info, warn};

use xenom_miner::block::BlockBuilder;
use xenom_miner::config::MinerConfig;
use xenom_miner::prover::{PublicInputs, ZkProver};
use xenom_miner::rpc::messages::TrainingBatch;
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::trainer::{CpuTrainer, MockTrainer, Trainer};
use xenom_miner::wallet::WalletManager;

const DEFAULT_RPC_URL: &str = "ws://localhost:16110";
const DEFAULT_MODEL_ID: &str = "dnabert2";
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

    /// Use the fast mock trainer instead of real Candle training.
    #[arg(long = "mock-mode", visible_alias = "mock")]
    mock: bool,

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
    config.threads = args.threads;
    config.mock_mode = args.mock;
    config.dry_run = args.dry_run;
    config.data_dir = data_dir.clone();
    config.save(&data_dir)?;

    let wallet =
        Arc::new(WalletManager::load_or_create(&data_dir, &args.password).with_context(|| "Failed to load or create wallet")?);
    let miner_address = if config.wallet_address.is_empty() { wallet.address().to_string() } else { config.wallet_address.clone() };
    info!("Miner address: {}", miner_address);

    let mut rpc_client: Option<XenomRpcClient> = None;
    ensure_rpc_connection(&mut rpc_client, &config.rpc_url, config.dry_run).await?;

    let trainer: Arc<dyn Trainer> = if config.mock_mode {
        info!("Using mock trainer");
        Arc::new(MockTrainer::new())
    } else {
        info!("Using Candle CPU trainer with {} threads", config.threads);
        Arc::new(CpuTrainer::new(config.threads)?)
    };

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

                let batch = match get_batch(&mut rpc_client, &config.model_id).await {
                    Some(batch) => batch,
                    None if config.dry_run => make_local_batch(&config.model_id, block_number),
                    None => {
                        warn!("No batch available; retrying");
                        tokio::time::sleep(RETRY_DELAY).await;
                        return Ok::<_, anyhow::Error>(());
                    }
                };

                let batch_for_training = batch.clone();
                let trainer = Arc::clone(&trainer);
                let result = tokio::task::spawn_blocking(move || trainer.train(&batch_for_training))
                    .await
                    .context("Training task panicked")??;

                let public_inputs = PublicInputs {
                    model_id: config.model_id.clone(),
                    batch_id: batch.batch_id,
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
