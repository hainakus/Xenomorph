//! Governance REST endpoints backed by ModelGovernance.sol.
//!
//! The gateway exposes read endpoints for active models and proposals, and
//! optionally signs on-chain write transactions when `GOVERNANCE_OPERATOR_KEY`
//! is configured.

use anyhow::{Context, Result};
use ethers::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub mod proposals;
pub mod voting;

/// On-chain representation of an active model (mirrors ModelGovernance.sol).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveModel {
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: [u8; 32],
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
    pub activation_block: u64,
    pub deprecation_block: u64,
    pub active: bool,
}

/// On-chain proposal summary (mirrors ModelGovernance.sol).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalSummary {
    pub proposal_id: u64,
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: [u8; 32],
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
    pub vote_start: u64,
    pub vote_end: u64,
    pub yes_votes: u64,
    pub no_votes: u64,
    pub executed: bool,
}

type ProposalRaw = (String, String, String, H256, U256, U256, U256, U256, U256, U256, U256, bool);

type ModelRaw = (String, String, String, H256, U256, U256, U256, U256, U256, bool);

pub struct GovernanceClient {
    read_contract: Contract<Provider<Http>>,
    write_contract: Option<Contract<SignerMiddleware<Provider<Http>, LocalWallet>>>,
}

impl GovernanceClient {
    pub fn new(rpc_url: &str, contract_address: Address, operator_key_hex: Option<&str>) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url).context("Invalid Ethereum RPC URL")?;

        let abi = ethers::abi::Abi::load(&include_bytes!("../../abi/ModelGovernance.json")[..])
            .context("Failed to load ModelGovernance ABI")?;
        let base = BaseContract::from(abi);

        let read_contract = Contract::new(contract_address, base.clone(), Arc::new(provider.clone()));

        let write_contract = operator_key_hex
            .and_then(decode_key)
            .map(|key| {
                let wallet = LocalWallet::from_bytes(&key).context("Invalid operator private key")?;
                let client = Arc::new(SignerMiddleware::new(provider.clone(), wallet));
                Ok::<_, anyhow::Error>(Contract::new(contract_address, base.clone(), client))
            })
            .transpose()?;

        Ok(Self { read_contract, write_contract })
    }

    pub async fn list_active_models(&self) -> Result<Vec<ActiveModel>> {
        let raw: Vec<ModelRaw> = self
            .read_contract
            .method::<(), Vec<ModelRaw>>("getActiveModels", ())?
            .call()
            .await
            .context("Failed to call getActiveModels")?;

        Ok(raw.into_iter().map(decode_active_model).collect())
    }

    pub async fn get_model_data(&self, model_id: &str) -> Result<Option<ActiveModel>> {
        let raw: ModelRaw = self.read_contract.method::<String, ModelRaw>("getModelData", model_id.to_string())?.call().await?;

        if raw.8 == U256::zero() {
            // deprecationBlock == 0 but activationBlock can be 0 too before activation;
            // we treat an activationBlock of 0 as "not found".
            return Ok(None);
        }

        Ok(Some(decode_active_model(raw)))
    }

    pub async fn list_proposals(&self, active_only: bool) -> Result<Vec<ProposalSummary>> {
        let count: U256 = self.read_contract.method::<(), U256>("proposalCount", ())?.call().await?;

        let mut proposals = Vec::new();
        for id in 0..count.as_u64() {
            if let Ok(raw) = self.read_contract.method::<U256, ProposalRaw>("proposals", U256::from(id))?.call().await {
                if !active_only || !raw.11 {
                    proposals.push(decode_proposal(id, raw));
                }
            }
        }

        Ok(proposals)
    }

    pub async fn get_proposal(&self, proposal_id: u64) -> Result<ProposalSummary> {
        let raw: ProposalRaw = self.read_contract.method::<U256, ProposalRaw>("proposals", U256::from(proposal_id))?.call().await?;

        Ok(decode_proposal(proposal_id, raw))
    }

    pub async fn propose_model(&self, req: ProposeModelRequest) -> Result<(H256, u64)> {
        let contract = self.write_contract.as_ref().context("Gateway not configured with GOVERNANCE_OPERATOR_KEY")?;

        let count_before: U256 = contract.method::<(), U256>("proposalCount", ())?.call().await?;

        let genesis = H256::from(req.genesis_checkpoint);

        let call = contract.method::<(String, String, String, H256, U256, U256, U256), ()>(
            "proposeModel",
            (
                req.model_id,
                req.hf_repo,
                req.hf_revision,
                genesis,
                U256::from(req.vram_required),
                U256::from(req.reward_per_block),
                U256::from(req.min_stake_to_train),
            ),
        )?;

        let pending = call.send().await?;
        let receipt =
            pending.await.context("proposeModel transaction failed")?.context("proposeModel transaction dropped / no receipt")?;
        let tx_hash = receipt.transaction_hash;
        let proposal_id = count_before.as_u64();

        Ok((tx_hash, proposal_id))
    }

    pub async fn vote(&self, proposal_id: u64, support: bool) -> Result<H256> {
        let contract = self.write_contract.as_ref().context("Gateway not configured with GOVERNANCE_OPERATOR_KEY")?;

        let call = contract.method::<(U256, bool), ()>("vote", (U256::from(proposal_id), support))?;

        let pending = call.send().await?;
        let receipt = pending.await.context("vote transaction failed")?.context("vote transaction dropped / no receipt")?;

        Ok(receipt.transaction_hash)
    }

    pub async fn execute(&self, proposal_id: u64) -> Result<H256> {
        let contract = self.write_contract.as_ref().context("Gateway not configured with GOVERNANCE_OPERATOR_KEY")?;

        let call = contract.method::<U256, ()>("executeProposal", U256::from(proposal_id))?;

        let pending = call.send().await?;
        let receipt = pending
            .await
            .context("executeProposal transaction failed")?
            .context("executeProposal transaction dropped / no receipt")?;

        Ok(receipt.transaction_hash)
    }
}

#[derive(Debug, Deserialize)]
pub struct ProposeModelRequest {
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: [u8; 32],
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
}

fn decode_active_model(raw: ModelRaw) -> ActiveModel {
    ActiveModel {
        model_id: raw.0,
        hf_repo: raw.1,
        hf_revision: raw.2,
        genesis_checkpoint: raw.3.to_fixed_bytes(),
        vram_required: raw.4.as_u64(),
        reward_per_block: raw.5.as_u64(),
        min_stake_to_train: raw.6.as_u64(),
        activation_block: raw.7.as_u64(),
        deprecation_block: raw.8.as_u64(),
        active: raw.9,
    }
}

fn decode_proposal(id: u64, raw: ProposalRaw) -> ProposalSummary {
    ProposalSummary {
        proposal_id: id,
        model_id: raw.0,
        hf_repo: raw.1,
        hf_revision: raw.2,
        genesis_checkpoint: raw.3.to_fixed_bytes(),
        vram_required: raw.4.as_u64(),
        reward_per_block: raw.5.as_u64(),
        min_stake_to_train: raw.6.as_u64(),
        vote_start: raw.7.as_u64(),
        vote_end: raw.8.as_u64(),
        yes_votes: raw.9.as_u64(),
        no_votes: raw.10.as_u64(),
        executed: raw.11,
    }
}

fn decode_key(key_hex: &str) -> Option<[u8; 32]> {
    let bytes = hex::decode(key_hex.trim_start_matches("0x")).ok()?;
    if bytes.len() != 32 {
        return None;
    }
    let mut key = [0u8; 32];
    key.copy_from_slice(&bytes);
    Some(key)
}
