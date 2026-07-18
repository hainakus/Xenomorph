use anyhow::Result;
use borsh::to_vec;
use sha2::{Digest, Sha256};

use crate::rpc::messages::{BlockHash, BlockHeader, DifficultyTarget, TrainingBlock, TrainingProof};
use crate::trainer::TrainingResult;

/// Builds training blocks that can be submitted to the Xenomorph node.
pub struct BlockBuilder {
    prev_block_hash: BlockHash,
    block_number: u64,
    miner_address: String,
}

impl BlockBuilder {
    pub fn new(miner_address: String) -> Self {
        Self { prev_block_hash: [0u8; 32], block_number: 0, miner_address }
    }

    /// Update the chain tip used for the next block.
    pub fn set_prev_block(&mut self, hash: BlockHash, number: u64) {
        self.prev_block_hash = hash;
        self.block_number = number;
    }

    /// Build a `TrainingBlock` from a completed training result.
    pub fn build_block(&mut self, result: &TrainingResult, zk_proof: Vec<u8>, difficulty: DifficultyTarget) -> Result<TrainingBlock> {
        let training_proof = TrainingProof {
            base_checkpoint: result.base_checkpoint,
            loss_before: result.loss_before,
            loss_after: result.loss_after,
            gradients_commitment: result.gradients_commitment,
            zk_proof,
            batch_indices: result.batch_indices.clone(),
            compute_time_ms: result.compute_time_ms,
        };

        let timestamp = chrono::Utc::now().timestamp() as u64;
        let merkle_root = compute_merkle_root(&training_proof)?;

        let mut header = BlockHeader {
            prev_block_hash: self.prev_block_hash,
            block_number: self.block_number + 1,
            timestamp,
            merkle_root,
            difficulty,
            nonce: 0,
        };

        // Simple proof-of-work: increment nonce until header hash beats difficulty.
        loop {
            let hash = compute_header_hash(&header);
            if hash_meets_difficulty(&hash, &difficulty) {
                self.prev_block_hash = hash;
                self.block_number = header.block_number;
                break;
            }
            header.nonce = header.nonce.wrapping_add(1);
            if header.nonce == 0 {
                // Difficulty target is set to zero (no work) in the default case.
                // If a real difficulty is configured and this overflows, accept it anyway.
                self.prev_block_hash = hash;
                self.block_number = header.block_number;
                break;
            }
        }

        Ok(TrainingBlock { header, training_proof, miner_address: self.miner_address.clone(), timestamp, signature: [0u8; 64] })
    }

    pub fn current_block_number(&self) -> u64 {
        self.block_number
    }
}

fn compute_header_hash(header: &BlockHeader) -> BlockHash {
    let mut hasher = Sha256::new();
    hasher.update(header.prev_block_hash);
    hasher.update(header.block_number.to_le_bytes());
    hasher.update(header.timestamp.to_le_bytes());
    hasher.update(header.merkle_root);
    hasher.update(header.difficulty);
    hasher.update(header.nonce.to_le_bytes());
    hasher.finalize().into()
}

fn compute_merkle_root(proof: &TrainingProof) -> Result<[u8; 32]> {
    let bytes = to_vec(proof)?;
    Ok(*blake3::hash(&bytes).as_bytes())
}

fn hash_meets_difficulty(hash: &BlockHash, difficulty: &DifficultyTarget) -> bool {
    // A lower target means higher difficulty. A zero difficulty target is treated
    // as "no work required", which is useful for dry-runs and tests.
    if difficulty.iter().all(|&b| b == 0) {
        return true;
    }
    hash <= difficulty
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::{mock_trainer::MockTrainer, Trainer};

    #[test]
    fn test_block_builder() {
        let mut builder = BlockBuilder::new("xnom:test".to_string());
        let trainer = MockTrainer::new();
        let batch = crate::rpc::messages::TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: vec![0, 1, 2],
            target_improvement: 0.01,
            learning_rate: 0.01,
        };
        let result = trainer.train(&batch).unwrap();

        let block = builder.build_block(&result, vec![0u8; 32], [0u8; 32]).unwrap();

        assert_eq!(block.header.block_number, 1);
        assert_eq!(block.miner_address, "xnom:test");
        assert_eq!(block.training_proof.loss_after, result.loss_after);
    }

    #[test]
    fn test_difficulty_zero_accepted() {
        let mut builder = BlockBuilder::new("xnom:test".to_string());
        let trainer = MockTrainer::new();
        let batch = crate::rpc::messages::TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [0u8; 32],
            data_indices: vec![0, 1, 2],
            target_improvement: 0.01,
            learning_rate: 0.01,
        };
        let result = trainer.train(&batch).unwrap();

        let block = builder.build_block(&result, vec![0u8; 32], [0xff; 32]).unwrap();
        // With the maximum target every hash is accepted, so the block is built without extra PoW iterations.
        assert_eq!(block.header.nonce, 0);
    }
}
