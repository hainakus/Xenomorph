//! Vote on and execute model proposals on ModelGovernance.sol.

use anyhow::{Context, Result};
use ethers::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tracing::info;

use super::ProposalRaw;

pub struct ProposalSummary {
    pub proposal_id: u64,
    pub model_id: String,
    pub vote_end: u64,
    pub yes_votes: u64,
    pub no_votes: u64,
    pub executed: bool,
}

pub struct GovernanceVoter {
    contract: Contract<SignerMiddleware<Provider<Http>, LocalWallet>>,
}

impl GovernanceVoter {
    pub fn new(contract_address: Address, provider: Provider<Http>, wallet: LocalWallet) -> Result<Self> {
        let abi = ethers::abi::Abi::load(&include_bytes!("../../abi/ModelGovernance.json")[..])
            .context("Failed to load ModelGovernance ABI")?;
        let client = Arc::new(SignerMiddleware::new(provider, wallet));
        let contract = Contract::new(contract_address, abi, client);
        Ok(Self { contract })
    }

    /// Cast a stake-weighted vote on a proposal.
    pub async fn vote_on_proposal(&self, proposal_id: u64, support: bool) -> Result<H256> {
        let call = self.contract.method::<(U256, bool), ()>("vote", (U256::from(proposal_id), support))?;

        let pending = call.send().await?;
        let receipt = pending.await.context("vote transaction failed")?.context("vote transaction dropped / no receipt")?;
        let tx_hash = receipt.transaction_hash;

        info!("Voted on proposal {} (support={}): tx {}", proposal_id, support, tx_hash);
        Ok(tx_hash)
    }

    /// Execute an approved proposal, activating the model on-chain.
    pub async fn execute_proposal(&self, proposal_id: u64) -> Result<H256> {
        let call = self.contract.method::<U256, ()>("executeProposal", U256::from(proposal_id))?;

        let pending = call.send().await?;
        let receipt = pending
            .await
            .context("executeProposal transaction failed")?
            .context("executeProposal transaction dropped / no receipt")?;
        let tx_hash = receipt.transaction_hash;

        info!("Executed proposal {}: tx {}", proposal_id, tx_hash);
        Ok(tx_hash)
    }

    /// Return proposals that are still open for voting (not executed).
    pub async fn get_active_proposals(&self) -> Result<Vec<ProposalSummary>> {
        let count: U256 = self.contract.method::<(), U256>("proposalCount", ())?.call().await?;

        let mut active = Vec::new();
        for id in 0..count.as_u64() {
            if let Ok(raw) = self.contract.method::<U256, ProposalRaw>("proposals", U256::from(id))?.call().await {
                if !raw.11 {
                    active.push(ProposalSummary {
                        proposal_id: id,
                        model_id: raw.0,
                        vote_end: raw.8.as_u64(),
                        yes_votes: raw.9.as_u64(),
                        no_votes: raw.10.as_u64(),
                        executed: raw.11,
                    });
                }
            }
        }

        Ok(active)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CastVoteRequest {
    pub proposal_id: u64,
    pub support: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecuteRequest {
    pub proposal_id: u64,
}
