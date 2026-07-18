//! Miner <-> node integration tests.

use crate::support::assertions::{assert_loss_improved, assert_ok};
use crate::support::miner_builder::TestMiner;
use crate::support::node_runner::spawn_mock_seed_node;
use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_miner_connects_to_node() {
    let node = spawn_mock_seed_node().await.unwrap();
    let mut miner = TestMiner::new(&node.url).await.unwrap();
    assert_ok(miner.connect().await, "miner should connect to seed node");
}

#[tokio::test]
#[serial]
async fn test_miner_receives_training_batch() {
    let node = spawn_mock_seed_node().await.unwrap();
    let mut miner = TestMiner::new(&node.url).await.unwrap();
    miner.connect().await.unwrap();

    let batch = miner.request_batch("dnabert2").await.unwrap();
    assert_eq!(batch.model_id, "dnabert2");
    assert!(!batch.data_indices.is_empty());
    assert!(batch.target_improvement > 0.0);
}

#[tokio::test]
#[serial]
async fn test_miner_submits_valid_block() {
    let node = spawn_mock_seed_node().await.unwrap();
    let mut miner = TestMiner::new(&node.url).await.unwrap();
    miner.connect().await.unwrap();

    let batch = miner.request_batch("dnabert2").await.unwrap();
    let result = miner.train(batch).unwrap();
    assert_loss_improved(result.loss_before, result.loss_after, 0.001);

    let block_hash = miner.submit_block(result).await.unwrap();
    assert_ne!(block_hash, [0u8; 32]);
    assert_eq!(node.accepted_blocks(), 1);
}

#[tokio::test]
#[serial]
async fn test_miner_gets_reward() {
    let node = spawn_mock_seed_node().await.unwrap();
    let mut miner = TestMiner::new(&node.url).await.unwrap();
    miner.connect().await.unwrap();

    let initial_balance = miner.get_balance().await.unwrap();
    let batch = miner.request_batch("dnabert2").await.unwrap();
    let result = miner.train(batch).unwrap();
    miner.submit_block(result).await.unwrap();
    let final_balance = miner.get_balance().await.unwrap();

    assert!(final_balance > initial_balance, "reward not credited");
    assert!(final_balance - initial_balance >= 10_000_000_000, "reward too small");
}
