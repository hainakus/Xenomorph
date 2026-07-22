use borsh::{BorshDeserialize, BorshSerialize};
use std::collections::HashMap;

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
///
/// When `encrypted` is true, `config`, `tokenizer` and `weights` are AES-256-GCM ciphertexts
/// with the nonce prepended, and the miner must decrypt them with the same `XENO_MODEL_KEY`.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpoint {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
    pub encrypted: bool,
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
    /// Hash of the model weights this genome batch is based on; must be used as the
    /// training block's `base_checkpoint`.
    pub base_checkpoint: [u8; 32],
}

/// One layer's gradient vector plus its original shape, used for FedAvg
/// aggregation across miners.
///
/// When `indices` is empty the layer is dense (`values` has the full flattened
/// tensor in row-major order). When `indices` is non-empty only those flattened
/// positions are non-zero and `values[i]` corresponds to `indices[i]`.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GradientLayer {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub indices: Vec<usize>,
}

/// Plaintext gradient payload that is serialized and then encrypted before
/// being sent over the wire.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GradientPayload {
    pub layer_gradients: HashMap<String, GradientLayer>,
}

/// Gradient update submitted by a miner to the aggregator.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GradientUpdate {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub encrypted_payload: Vec<u8>,
    /// Relative weight of this participant in the average (e.g. dataset size).
    pub participant_weight: f32,
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
    GetModelCheckpointInfo(GetModelCheckpointInfo),
    SubmitGradients(GradientUpdate),
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
    ModelCheckpointInfo(ModelCheckpointInfo),
    GradientAck { new_checkpoint: Option<[u8; 32]> },
}

/// Request the lightweight metadata for the active model checkpoint.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GetModelCheckpointInfo {
    pub model_id: String,
}

/// Lightweight model checkpoint metadata (no weights).
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpointInfo {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
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
            encrypted: true,
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
