//! Build test miners backed by the real `xenom-miner` crates.

use anyhow::{Context, Result};
use kaspa_consensus_core::network::NetworkType;
use tempfile::TempDir;
use xenom_miner::block::BlockBuilder;
use xenom_miner::prover::{PublicInputs, ZkProver};
use xenom_miner::rpc::messages::TrainingBatch;
use xenom_miner::rpc::XenomRpcClient;
use xenom_miner::trainer::{MockTrainer, Trainer, TrainingResult};
use xenom_miner::wallet::WalletManager;

/// A test miner that uses real `xenom-miner` components and talks to a mock
/// WebSocket seed node.
pub struct TestMiner {
    pub wallet: WalletManager,
    pub rpc: XenomRpcClient,
    pub address: String,
    data_dir: TempDir,
    builder: BlockBuilder,
    trainer: MockTrainer,
    prover: ZkProver,
}

impl TestMiner {
    /// Create a miner and load/create a wallet in a temporary directory.
    pub async fn new(url: &str) -> Result<Self> {
        let data_dir = tempfile::tempdir()?;
        let wallet = WalletManager::load_or_create(data_dir.path(), "test-password", NetworkType::Devnet)
            .context("failed to create test wallet")?;
        let address = wallet.address().to_string();
        let rpc = XenomRpcClient::new(url.to_string());

        Ok(Self {
            wallet,
            rpc,
            address: address.clone(),
            data_dir,
            builder: BlockBuilder::new(address),
            trainer: MockTrainer::new(),
            prover: ZkProver::new(),
        })
    }

    /// Connect to the seed node (or reconnect).
    pub async fn connect(&mut self) -> Result<()> {
        self.rpc.connect().await
    }

    /// Request a training batch for `model_id`.
    pub async fn request_batch(&mut self, model_id: &str) -> Result<TrainingBatch> {
        self.rpc.get_training_batch(model_id).await?.context("seed node returned no batch")
    }

    /// Train a batch using the mock CPU trainer.
    pub fn train(&self, batch: TrainingBatch) -> Result<TrainingResult> {
        self.trainer.train(&batch).context("training failed")
    }

    /// Build and sign a block from a training result, then submit it.
    pub async fn submit_block(&mut self, result: TrainingResult) -> Result<[u8; 32]> {
        let public_inputs = PublicInputs {
            model_id: result.model_id.clone(),
            batch_id: 1,
            loss_before: result.loss_before,
            loss_after: result.loss_after,
            gradients_commitment: result.gradients_commitment,
            base_checkpoint: result.base_checkpoint,
        };

        let zk_proof = self.prover.generate_proof(&result, &public_inputs).context("failed to generate zk proof")?;

        let mut block = self.builder.build_block(&result.model_id, &result, zk_proof, [0u8; 32]).context("failed to build block")?;

        self.wallet.sign_block(&mut block).context("failed to sign block")?;

        self.rpc.submit_block(block).await
    }

    /// Submit a block with an intentionally invalid (zeroed) proof.
    pub async fn submit_invalid_block(&mut self, result: TrainingResult) -> Result<[u8; 32]> {
        let mut block = self.builder.build_block(&result.model_id, &result, vec![0u8; 32], [0u8; 32])
            .context("failed to build invalid block")?;
        self.wallet.sign_block(&mut block)?;
        self.rpc.submit_block(block).await
    }

    pub async fn get_balance(&mut self) -> Result<u64> {
        self.rpc.get_balance(&self.address).await
    }

    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn data_dir(&self) -> &std::path::Path {
        self.data_dir.path()
    }
}
