use borsh::{BorshDeserialize, BorshSerialize};
use std::collections::HashMap;

pub type BlockHash = [u8; 32];
pub type DifficultyTarget = [u8; 32];

/// A slice of the genome selected for MLM training.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GenomeSlice {
    pub chunk_idx: u64,
    pub start_base: u32,
    pub length: u32,
}

/// A training batch composed of genome slices returned by the seed-node.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GenomeTrainingBatch {
    pub batch_id: u64,
    pub model_id: String,
    pub genome_merkle_root: [u8; 32],
    pub data_indices: Vec<GenomeSlice>,
    pub mask_ratio: f32,
    pub seq_length: usize,
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

/// Raw model checkpoint bytes returned by the seed-node.
///
/// When `encrypted` is true, `config`, `tokenizer` and `weights` are AES-256-GCM
/// ciphertexts (nonce || ciphertext) and must be decrypted with `XENO_MODEL_KEY`.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpoint {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
    pub encrypted: bool,
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

/// V2 model checkpoint for LoRA adapter-aware sync.
///
/// `base_checkpoint` is the combined hash used as the training block base.
/// `base_hash` is the hash of the frozen base weights the node used as the
/// foundation for the adapter. `is_adapter` tells the receiver whether
/// `weights` contains the full merged checkpoint or only the adapter tensors.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpointV2 {
    pub model_id: String,
    pub base_checkpoint: [u8; 32],
    pub base_hash: [u8; 32],
    pub config: Vec<u8>,
    pub tokenizer: Vec<u8>,
    pub weights: Vec<u8>,
    pub encrypted: bool,
    pub is_adapter: bool,
}

/// V2 request for `ModelCheckpointV2`.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GetModelCheckpointV2 {
    pub model_id: String,
    /// If the miner already has a base checkpoint with this hash, the node may
    /// return only the LoRA adapter. `None` always requests the full bundle.
    pub cached_base_hash: Option<[u8; 32]>,
}

/// V2 lightweight checkpoint metadata exposing both the combined and base hashes.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct GetModelCheckpointInfoV2 {
    pub model_id: String,
}

/// V2 lightweight checkpoint metadata.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct ModelCheckpointInfoV2 {
    pub model_id: String,
    /// Combined hash (the active checkpoint id used as `base_checkpoint` in training).
    pub base_checkpoint: [u8; 32],
    /// Hash of the frozen base weights the current adapter is built on.
    pub base_hash: [u8; 32],
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
/// New variants are appended at the end to preserve Borsh enum indices for the
/// seed-node.
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
    GetModelCheckpointInfoV2(GetModelCheckpointInfoV2),
    GetModelCheckpointV2(GetModelCheckpointV2),
}

/// Response messages sent from the Xenomorph node to the miner.
/// New variants are appended at the end to preserve Borsh enum indices for the
/// seed-node.
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
    ModelCheckpointInfoV2(ModelCheckpointInfoV2),
    ModelCheckpointV2(ModelCheckpointV2),
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
    fn test_roundtrip_training_batch() {
        let batch = TrainingBatch {
            batch_id: 12345,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0xab; 32],
            data_indices: vec![0, 1, 2, 3],
            target_improvement: 0.01,
            learning_rate: 0.001,
        };

        let bytes = to_vec(&batch).unwrap();
        let decoded: TrainingBatch = TrainingBatch::try_from_slice(&bytes).unwrap();
        assert_eq!(batch, decoded);
    }

    #[test]
    fn test_roundtrip_training_block() {
        let block = TrainingBlock {
            header: BlockHeader {
                prev_block_hash: [0u8; 32],
                block_number: 1,
                timestamp: 0,
                merkle_root: [1u8; 32],
                difficulty: [2u8; 32],
                nonce: 0,
            },
            model_id: "dnabert2".to_string(),
            training_proof: TrainingProof {
                base_checkpoint: [3u8; 32],
                loss_before: 2.45,
                loss_after: 2.41,
                gradients_commitment: [4u8; 32],
                zk_proof: vec![0u8; 64],
                batch_indices: vec![0, 1, 2],
                compute_time_ms: 100,
            },
            miner_address: "xnom:test".to_string(),
            timestamp: 0,
            signature: [0u8; 64],
        };

        let bytes = to_vec(&block).unwrap();
        let decoded: TrainingBlock = TrainingBlock::try_from_slice(&bytes).unwrap();
        assert_eq!(block, decoded);
    }

    #[test]
    fn test_roundtrip_rpc_messages() {
        let req = RpcRequest::GetTrainingBatch { model_id: "dnabert2".to_string() };
        let env = RpcEnvelope { request_id: 1, payload: req };
        let bytes = to_vec(&env).unwrap();
        let decoded: RpcEnvelope = RpcEnvelope::try_from_slice(&bytes).unwrap();
        assert_eq!(env, decoded);
    }

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
}
