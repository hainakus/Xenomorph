use anyhow::Result;

use crate::rpc::messages::ModelCheckpoint;
use crate::rpc::XenomRpcClient;

/// In-memory bundle returned by the seed-node. The miner does not persist these bytes.
pub struct ModelBundle {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}

impl From<ModelCheckpoint> for ModelBundle {
    fn from(cp: ModelCheckpoint) -> Self {
        Self {
            model_id: cp.model_id,
            base_checkpoint: cp.base_checkpoint,
            config: cp.config,
            tokenizer: cp.tokenizer,
            weights: cp.weights,
        }
    }
}

/// Convenience helper that fetches a model checkpoint from the seed-node and returns it in memory.
pub async fn fetch_model_checkpoint(rpc: &mut XenomRpcClient, model_id: &str) -> Result<ModelBundle> {
    let cp = rpc.get_model_checkpoint(model_id).await?;
    Ok(cp.into())
}
