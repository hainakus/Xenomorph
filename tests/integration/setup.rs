//! Setup/teardown for the integration-test devnet.

use anyhow::{Context, Result};
use ethers::prelude::*;
use seed_node::consensus::dynamic_registry::{ActiveModel, DynamicModelRegistry};
use seed_node::governance::proposer::{GovernanceProposer, ProposalArgs};
use seed_node::governance::voter::{GovernanceVoter, ProposalSummary};
use std::sync::Arc;
use tempfile::TempDir;

use crate::support::contract_deployer::{deploy_contracts, DeployedContracts};
use crate::support::miner_builder::TestMiner;
use crate::support::node_runner::{spawn_anvil, spawn_mock_seed_node, MockSeedNode};

const PROPOSAL_STAKE: u64 = 10_000_000_000_000_000; // 10M Xenom
const VOTER_STAKE: u64 = 300_000_000_000_000; // 300k Xenom each
const GOV_TIME_SKIP: u64 = 8 * 24 * 60 * 60 + 10; // 7 days voting + 1 day delay + buffer

/// Full devnet environment used by integration tests.
pub struct DevnetEnvironment {
    pub data_dir: TempDir,
    pub contracts: DeployedContracts,
    pub registry: DynamicModelRegistry,
    pub mock_node: MockSeedNode,
    pub anvil: ethers::utils::AnvilInstance,
    pub faucet: LocalWallet,
}

impl DevnetEnvironment {
    /// Create an isolated devnet with a local Anvil node, deployed contracts,
    /// a mock seed node, and a seeded model registry.
    pub async fn new() -> Result<Self> {
        let data_dir = tempfile::tempdir()?;
        let anvil = spawn_anvil()?;
        let contracts = deploy_contracts(&anvil).await?;
        let faucet = contracts.faucet.clone();

        let registry = DynamicModelRegistry::new(contracts.governance, contracts.provider.clone(), 1).await?;

        let mock_node = spawn_mock_seed_node().await?;

        // Seed the faucet with enough governance stake to create proposals.
        stake_governance(&contracts, &faucet, PROPOSAL_STAKE).await?;

        Ok(Self { data_dir, contracts, registry, mock_node, anvil, faucet })
    }

    /// Create a test miner pointed at the mock seed node.
    pub async fn create_miner(&self) -> Result<TestMiner> {
        TestMiner::new(&self.mock_node.url).await
    }

    /// Build a `GovernanceVoter` from one of the Anvil dev accounts.
    pub fn voter_wallet(&self, index: usize) -> Result<LocalWallet> {
        let secret = self.anvil.keys().get(index).context("not enough anvil keys")?.clone();
        let signing_key = ethers::core::k256::ecdsa::SigningKey::from(&secret);
        Ok(LocalWallet::from(signing_key).with_chain_id(self.anvil.chain_id()))
    }

    /// Create `n` voters and pre-fund their governance stake.
    pub async fn create_voters(&self, n: usize) -> Result<Vec<GovernanceVoter>> {
        let mut voters = Vec::with_capacity(n);
        for i in 1..=n {
            let wallet = self.voter_wallet(i)?;
            stake_governance(&self.contracts, &wallet, VOTER_STAKE).await?;

            let voter = GovernanceVoter::new(self.contracts.governance, self.contracts.provider.clone(), wallet)?;
            voters.push(voter);
        }
        Ok(voters)
    }

    /// Create a `GovernanceProposer` for the faucet account.
    pub fn proposer(&self) -> Result<GovernanceProposer> {
        GovernanceProposer::new(self.contracts.governance, self.contracts.provider.clone(), self.contracts.faucet.clone())
    }

    /// Submit a model proposal and return the emitted proposal ID.
    pub async fn create_proposal(&self, args: ProposalArgs) -> Result<u64> {
        let proposer = self.proposer()?;
        let (_, id) = proposer.propose_model(args).await?;
        Ok(id)
    }

    /// Cast a vote using the provided voter.
    pub async fn vote(&self, voter: &GovernanceVoter, proposal_id: u64, support: bool) -> Result<()> {
        voter.vote_on_proposal(proposal_id, support).await?;
        Ok(())
    }

    /// Execute a proposal after voting and time delay.
    pub async fn execute_proposal(&self, proposal_id: u64) -> Result<()> {
        // Advance chain time past voting period + execution delay.
        self.increase_time(GOV_TIME_SKIP).await?;
        let voter = GovernanceVoter::new(self.contracts.governance, self.contracts.provider.clone(), self.contracts.faucet.clone())?;
        voter.execute_proposal(proposal_id).await?;
        Ok(())
    }

    /// Mine `n` empty blocks.
    pub async fn mine_blocks(&self, n: u64) -> Result<()> {
        for _ in 0..n {
            let _response: serde_json::Value = self.contracts.provider.request("evm_mine", ()).await?;
            tracing::debug!("mined block, response {:?}", _response);
        }
        Ok(())
    }

    /// Increase chain time by `seconds`.
    pub async fn increase_time(&self, seconds: u64) -> Result<()> {
        let params = (U256::from(seconds),);
        let _response: serde_json::Value = self.contracts.provider.request("evm_increaseTime", params).await?;
        // Mine a block so the new timestamp is applied.
        let _mined: serde_json::Value = self.contracts.provider.request("evm_mine", ()).await?;
        Ok(())
    }

    /// Refresh the registry cache and return active models.
    pub async fn active_models(&self) -> Result<Vec<ActiveModel>> {
        let block = self.contracts.provider.get_block_number().await?;
        self.registry.refresh(block.as_u64()).await?;
        Ok(self.registry.list_active_models().await)
    }

    /// Query a proposal on-chain.
    pub async fn get_proposal(&self, proposal_id: u64) -> Result<ProposalSummary> {
        let proposer = self.proposer()?;
        Ok(proposer.get_proposal(proposal_id).await?)
    }

    /// Check whether `miner_stake` is enough to train `model_id`.
    pub async fn can_mine_model(&self, _address: &str, model_id: &str, miner_stake: u64) -> bool {
        self.registry.can_mine_model(model_id, miner_stake).await
    }

    /// Return the per-block reward for an active model (ignoring the block hash).
    pub async fn get_block_reward(&self, _block_hash: &[u8; 32]) -> Result<u64> {
        // The mock seed node does not track block hashes; the reward is read
        // from the on-chain registry for the currently active model.
        self.registry.get_reward("evo2-7b").await.context("no active evo2-7b model in registry")
    }
}

/// Stake `amount` on `ModelGovernance` from `wallet`.
async fn stake_governance(contracts: &DeployedContracts, wallet: &LocalWallet, amount: u64) -> Result<()> {
    let client = Arc::new(SignerMiddleware::new_with_provider_chain(contracts.provider.clone(), wallet.clone()).await?);
    let abi_bytes = std::fs::read(format!("{}/fixtures/contracts/ModelGovernance.abi", env!("CARGO_MANIFEST_DIR")))?;
    let abi = ethers::abi::Abi::load(abi_bytes.as_slice())?;
    let contract = Contract::new(contracts.governance, abi, client);
    let call = contract.method::<U256, ()>("stake", U256::from(amount))?;
    let pending = call.send().await.context("failed to send stake transaction")?;
    let _receipt = pending.await.context("stake transaction failed")?.context("stake transaction dropped / no receipt")?;
    Ok(())
}
