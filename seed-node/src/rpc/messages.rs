use borsh_miner::{BorshDeserialize, BorshSerialize};

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
    pub training_proof: TrainingProof,
    pub miner_address: String,
    pub timestamp: u64,
    pub signature: [u8; 64],
}

/// Request messages sent from the miner to the Xenomorph node.
#[allow(clippy::large_enum_variant)]
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum RpcRequest {
    GetTrainingBatch { model_id: String },
    SubmitBlock(TrainingBlock),
    GetBalance { address: String },
    GetDifficulty,
    Heartbeat,
}

/// Response messages sent from the Xenomorph node to the miner.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub enum RpcResponse {
    TrainingBatch(Option<TrainingBatch>),
    BlockHash(BlockHash),
    Balance(u64),
    Difficulty(DifficultyTarget),
    Pong,
    Error(String),
}

/// Wire envelope used by the RPC client to tag requests.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq)]
pub struct RpcEnvelope {
    pub request_id: u64,
    pub payload: RpcRequest,
}
