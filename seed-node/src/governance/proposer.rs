//! Create model proposals on-chain through ModelGovernance.sol.

use anyhow::{Context, Result};
use ethers::prelude::*;
use std::sync::Arc;
use tracing::info;

use super::voter::ProposalSummary;
use super::ProposalRaw;

pub struct ProposalArgs {
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: [u8; 32],
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
}

pub struct GovernanceProposer {
    contract: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
}

impl GovernanceProposer {
    pub fn new(contract_address: Address, provider: Provider<Http>, wallet: LocalWallet) -> Result<Self> {
        let abi = ethers::abi::Abi::load(&include_bytes!("../../abi/ModelGovernance.json")[..])
            .context("Failed to load ModelGovernance ABI")?;
        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let contract = Contract::new(contract_address, abi, client);
        Ok(Self { contract })
    }

    /// Submit a `proposeModel` transaction and return the emitted proposal ID.
    pub async fn propose_model(&self, args: ProposalArgs) -> Result<(H256, u64)> {
        let proposal_count_before: U256 = self.contract.method::<(), U256>("proposalCount", ())?.call().await?;

        let genesis = H256::from(args.genesis_checkpoint);

        let call = self.contract.method::<(String, String, String, H256, U256, U256, U256), ()>(
            "proposeModel",
            (
                args.model_id.clone(),
                args.hf_repo,
                args.hf_revision,
                genesis,
                U256::from(args.vram_required),
                U256::from(args.reward_per_block),
                U256::from(args.min_stake_to_train),
            ),
        )?;

        let pending = call.send().await?;
        let receipt =
            pending.await.context("proposeModel transaction failed")?.context("proposeModel transaction dropped / no receipt")?;
        let tx_hash = receipt.transaction_hash;

        let proposal_id = proposal_count_before.as_u64();

        info!("Created proposal {} for model {}: tx {}", proposal_id, args.model_id, tx_hash);

        Ok((tx_hash, proposal_id))
    }

    /// Fetch a proposal's on-chain summary.
    pub async fn get_proposal(&self, proposal_id: u64) -> Result<ProposalSummary> {
        let raw: ProposalRaw = self.contract.method::<U256, ProposalRaw>("proposals", U256::from(proposal_id))?.call().await?;

        Ok(ProposalSummary {
            proposal_id,
            model_id: raw.0,
            vote_end: raw.8.as_u64(),
            yes_votes: raw.9.as_u64(),
            no_votes: raw.10.as_u64(),
            executed: raw.11,
        })
    }
}
