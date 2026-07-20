use borsh::{BorshDeserialize, BorshSerialize};

use crate::genome::GenomeTrainingBatch;

pub type BlockHash = [u8; 32];
pub type DifficultyTarget = [u8; 32];

/// A batch of training data requested from the Xenomorph node.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct TrainingBatch {
    pub batch_id: u64,
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub data_indices: Vec<u64>,
    pub target_improvement: f64,
    pub learning_rate: f32,
}

/// Proof that training work has been performed.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct TrainingProof {
    pub base_checkpoint: [u8; 32],
    pub loss_before: f64,
    pub loss_after: f64,
    pub gradients_commitment: [u8; 32],
    pub zk_proof: Vec<u8>,
    pub batch_indices: Vec<u64>,
    pub compute_time_ms: u64,
}

/// Header for a training block.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct BlockHeader {
    pub prev_block_hash: BlockHash,
    pub block_number: u64,
    pub timestamp: u64,
    pub merkle_root: [u8; 32],
    pub difficulty: DifficultyTarget,
    pub nonce: u64,
}

/// A full training block ready to be submitted.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct TrainingBlock {
    pub header: BlockHeader,
    pub model_id: String,
    pub training_proof: TrainingProof,
    pub miner_address: String,
    pub timestamp: u64,
    pub signature: [u8; 64],
}

/// Raw model checkpoint bytes returned by the seed-node. The seed-node is the only entity that
/// downloads and stores model weights; the miner receives them in memory and does not persist.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpoint {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
}

/// Request for a genome-backed DNABERT-2 training batch.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GetGenomeTrainingBatch {
    pub genome_merkle_root: [u8; 32],
    pub model_id: String,
    pub preferred_batch_size: usize,
}

/// Response containing a genome-backed training batch and the extracted DNA sequences.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GenomeTrainingBatchMsg {
    pub batch: GenomeTrainingBatch,
    pub sequences: Vec<String>,
}

/// Request messages sent from the miner to the Xenomorph node.
/// New variants are appended at the end to preserve Borsh enum indices for
/// existing miners.
#[allow(clippy::large_enum_variant)]
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum RpcRequest {
    GetTrainingBatch { model_id: String },
    GetModelCheckpoint { model_id: String },
    SubmitBlock(TrainingBlock),
    GetBalance { address: String },
    GetDifficulty,
    Heartbeat,
    GetGenomeTrainingBatch(GetGenomeTrainingBatch),
}

/// Response messages sent from the Xenomorph node to the miner.
/// New variants are appended at the end to preserve Borsh enum indices for
/// existing miners.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum RpcResponse {
    TrainingBatch(Option<TrainingBatch>),
    ModelCheckpoint(ModelCheckpoint),
    BlockHash(BlockHash),
    Balance(u64),
    Difficulty(DifficultyTarget),
    Pong,
    Error(String),
    GenomeTrainingBatch(GenomeTrainingBatchMsg),
}

/// Wire envelope used by the RPC client to tag requests.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct RpcEnvelope {
    pub request_id: u64,
    pub payload: RpcRequest,
}

#[cfg(test)]
mod tests {
    use super::*;
    use borsh::to_vec;

    #[test]
    fn test_roundtrip_model_checkpoint() {
        let checkpoint = ModelCheckpoint {
            model_id: "multimolecule/dnabert2".to_string(),
            base_checkpoint: [1u8; 32],
            config: b"{}".to_vec(),
            tokenizer: b"[]".to_vec(),
            weights: vec![0u8; 64],
        };
        let bytes = to_vec(&checkpoint).unwrap();
        let decoded: ModelCheckpoint = ModelCheckpoint::try_from_slice(&bytes).unwrap();
        assert_eq!(checkpoint, decoded);
    }

    #[test]
    fn test_roundtrip_request() {
        let req = RpcRequest::GetModelCheckpoint { model_id: "dnabert2".to_string() };
        let env = RpcEnvelope { request_id: 1, payload: req };
        let bytes = to_vec(&env).unwrap();
        let decoded: RpcEnvelope = RpcEnvelope::try_from_slice(&bytes).unwrap();
        assert_eq!(env, decoded);
    }
}
