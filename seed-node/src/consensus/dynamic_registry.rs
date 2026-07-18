//! Dynamic Model Registry - reads active models from ModelGovernance.sol.
//!
//! Instead of hard-coding supported models, the seed node queries the
//! governance contract and caches the list, refreshing every N blocks.

use anyhow::{Context, Result};
use ethers::abi::Abi;
use ethers::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

/// On-chain metadata for an active model.
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

/// Client for the ModelGovernance contract that caches active models locally.
pub struct DynamicModelRegistry {
    contract: ContractInstance<Arc<Provider<Http>>, Provider<Http>>,
    cache: Arc<RwLock<Vec<ActiveModel>>>,
    last_update: Arc<RwLock<u64>>,
    update_interval: u64,
}

impl DynamicModelRegistry {
    pub async fn new(
        contract_address: Address,
        provider: Provider<Http>,
        update_interval: u64,
    ) -> Result<Self> {
        let abi = Abi::load(&include_bytes!("../../abi/ModelGovernance.json")[..])
            .context("Failed to load ModelGovernance ABI")?;

        let client = Arc::new(provider);
        let contract = Contract::new(contract_address, abi, client);

        Ok(Self {
            contract,
            cache: Arc::new(RwLock::new(Vec::new())),
            last_update: Arc::new(RwLock::new(0)),
            update_interval,
        })
    }

    /// Update the cached model list if the update interval has passed.
    pub async fn update_if_needed(&self, current_block: u64) -> Result<()> {
        let last = *self.last_update.read().await;

        if current_block.saturating_sub(last) >= self.update_interval {
            info!("Updating model registry from contract...");

            let models = self.fetch_active_models().await?;

            let mut cache = self.cache.write().await;
            *cache = models;
            *self.last_update.write().await = current_block;

            info!("Registry updated: {} active models", cache.len());
            for m in cache.iter() {
                info!(
                    "  - {}: {} Xenom/block, {}GB VRAM",
                    m.model_id, m.reward_per_block, m.vram_required
                );
            }
        }

        Ok(())
    }

    /// Force a refresh from the contract regardless of the cache interval.
    pub async fn refresh(&self, current_block: u64) -> Result<()> {
        info!("Refreshing model registry from contract...");

        let models = self.fetch_active_models().await?;

        let mut cache = self.cache.write().await;
        *cache = models;
        *self.last_update.write().await = current_block;

        info!("Registry refreshed: {} active models", cache.len());
        Ok(())
    }

    /// Check whether a model is valid for mining at the given block.
    pub async fn is_valid_for_mining(&self, model_id: &str, block: u64) -> bool {
        match self.query_model_active(model_id, block).await {
            Ok(active) => active,
            Err(e) => {
                warn!("Failed to query model activity: {}", e);
                false
            }
        }
    }

    /// Return the per-block reward for a model, if active.
    pub async fn get_reward(&self, model_id: &str) -> Option<u64> {
        let cache = self.cache.read().await;
        cache
            .iter()
            .find(|m| m.model_id == model_id && m.active)
            .map(|m| m.reward_per_block)
    }

    /// List all currently cached active models.
    pub async fn list_active_models(&self) -> Vec<ActiveModel> {
        self.cache
            .read()
            .await
            .iter()
            .filter(|m| m.active)
            .cloned()
            .collect()
    }

    /// Verify that a miner has enough stake to train a given model.
    pub async fn can_mine_model(&self, model_id: &str, miner_stake: u64) -> bool {
        let cache = self.cache.read().await;
        cache
            .iter()
            .find(|m| m.model_id == model_id && m.active)
            .map(|m| miner_stake >= m.min_stake_to_train)
            .unwrap_or(false)
    }

    /// Return true if the contract considers the model active at `block`.
    async fn query_model_active(&self, model_id: &str, block: u64) -> Result<bool> {
        let result: bool = self
            .contract
            .method::<(String, U256), bool>(
                "isModelActive",
                (model_id.to_string(), U256::from(block)),
            )?
            .call()
            .await?;
        Ok(result)
    }

    /// Fetch active models from the contract and decode them into `ActiveModel`.
    async fn fetch_active_models(&self) -> Result<Vec<ActiveModel>> {
        type RawModel = (
            String,
            String,
            String,
            H256,
            U256,
            U256,
            U256,
            U256,
            U256,
            bool,
        );

        let raw_models: Vec<RawModel> = self
            .contract
            .method::<(), Vec<RawModel>>("getActiveModels", ())?
            .call()
            .await
            .context("Failed to call getActiveModels")?;

        Ok(raw_models.into_iter().map(decode_active_model).collect())
    }
}

fn decode_active_model(
    raw: (
        String,
        String,
        String,
        H256,
        U256,
        U256,
        U256,
        U256,
        U256,
        bool,
    ),
) -> ActiveModel {
    let (
        model_id,
        hf_repo,
        hf_revision,
        genesis_checkpoint_h256,
        vram_required,
        reward_per_block,
        min_stake_to_train,
        activation_block,
        deprecation_block,
        active,
    ) = raw;

    ActiveModel {
        model_id,
        hf_repo,
        hf_revision,
        genesis_checkpoint: genesis_checkpoint_h256.to_fixed_bytes(),
        vram_required: vram_required.as_u64(),
        reward_per_block: reward_per_block.as_u64(),
        min_stake_to_train: min_stake_to_train.as_u64(),
        activation_block: activation_block.as_u64(),
        deprecation_block: deprecation_block.as_u64(),
        active,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_active_model() -> ActiveModel {
        ActiveModel {
            model_id: "dnabert2".to_string(),
            hf_repo: "xenom/dnabert2".to_string(),
            hf_revision: "main".to_string(),
            genesis_checkpoint: [0u8; 32],
            vram_required: 8,
            reward_per_block: 1000,
            min_stake_to_train: 100,
            activation_block: 1,
            deprecation_block: 0,
            active: true,
        }
    }

    #[test]
    fn test_can_mine_model() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let provider = Provider::<Http>::try_from("http://localhost:8545")
                .expect("valid localhost URL");
            let registry = DynamicModelRegistry {
                contract: Contract::new(
                    Address::zero(),
                    Abi::load(&b"[]"[..]).unwrap(),
                    Arc::new(provider),
                ),
                cache: Arc::new(RwLock::new(vec![sample_active_model()])),
                last_update: Arc::new(RwLock::new(0)),
                update_interval: 100,
            };

            assert!(registry.can_mine_model("dnabert2", 100).await);
            assert!(!registry.can_mine_model("dnabert2", 50).await);
            assert!(!registry.can_mine_model("unknown", 100).await);
        });
    }
}
