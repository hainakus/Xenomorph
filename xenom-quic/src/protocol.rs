use borsh::{BorshDeserialize, BorshSerialize};

/// The kind of checkpoint file being requested over QUIC.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CheckpointFileType {
    Config,
    Tokenizer,
    Weights,
    Adapter,
}

/// Client request for a single checkpoint file.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct CheckpointFileRequest {
    pub model_id: String,
    pub weights_hash: [u8; 32],
    pub file_type: CheckpointFileType,
}

/// Status returned in the QUIC response header.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResponseStatus {
    Ok,
    NotFound,
    NotAuthorized,
}

/// Header sent before the raw file bytes.
#[derive(BorshSerialize, BorshDeserialize, Debug, Clone, PartialEq, Eq)]
pub struct CheckpointFileResponseHeader {
    pub status: ResponseStatus,
    pub length: u64,
    /// blake3 hash of the response payload, used by the client to verify integrity.
    pub content_hash: [u8; 32],
}
