//! Shared test data and helpers.

use seed_node::governance::proposer::ProposalArgs;
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct FixtureModel {
    pub model_id: String,
    pub hf_repo: String,
    pub hf_revision: String,
    pub genesis_checkpoint: String,
    pub vram_required: u64,
    pub reward_per_block: u64,
    pub min_stake_to_train: u64,
}

pub fn load_models() -> Vec<FixtureModel> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/models.json");
    let json = std::fs::read_to_string(path).expect("models.json fixture missing");
    serde_json::from_str(&json).expect("failed to parse models.json")
}

pub fn evo2_proposal() -> ProposalArgs {
    ProposalArgs {
        model_id: "evo2-7b".to_string(),
        hf_repo: "arcinstitute/evo2_7b".to_string(),
        hf_revision: "main".to_string(),
        genesis_checkpoint: [0u8; 32],
        vram_required: 48,
        reward_per_block: 50_000_000_000,
        min_stake_to_train: 50_000,
    }
}

pub fn dnabert2_proposal() -> ProposalArgs {
    ProposalArgs {
        model_id: "dnabert2".to_string(),
        hf_repo: "xenom/dnabert2".to_string(),
        hf_revision: "main".to_string(),
        genesis_checkpoint: [0u8; 32],
        vram_required: 8,
        reward_per_block: 10_000_000_000,
        min_stake_to_train: 10_000,
    }
}

/// Convert a string to a 32-byte identifier used as `bytes32` on-chain.
///
/// * Valid hex strings (optionally prefixed with `0x`, at most 64 hex digits)
///   are decoded and right-padded with zeros (standard `H256`/`bytes32` hex).
/// * ASCII strings of 32 bytes or less are left-padded with zeros so the
///   characters sit at the start of the array.
/// * Longer inputs are deterministically hashed with Keccak-256.
pub fn hex_to_bytes32(input: &str) -> [u8; 32] {
    let mut arr = [0u8; 32];

    let trimmed = input.trim_start_matches("0x");
    let is_valid_hex =
        !trimmed.is_empty() && trimmed.len() <= 64 && trimmed.len() % 2 == 0 && trimmed.chars().all(|c| c.is_ascii_hexdigit());

    if is_valid_hex {
        let decoded = hex::decode(trimmed).expect("valid hex");
        arr[32 - decoded.len()..].copy_from_slice(&decoded);
        return arr;
    }

    let bytes = input.as_bytes();
    if bytes.len() <= 32 {
        arr[..bytes.len()].copy_from_slice(bytes);
        return arr;
    }

    // Long input: hash to 32 bytes.
    use sha3::{Digest, Keccak256};
    let hash = Keccak256::digest(bytes);
    arr.copy_from_slice(&hash);
    arr
}
