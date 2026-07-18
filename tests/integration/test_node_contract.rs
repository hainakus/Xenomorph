//! Node <-> smart contract integration tests.

use crate::fixtures::dnabert2_proposal;
use crate::setup::DevnetEnvironment;
use seed_node::model::checkpoint::{ModelCheckpoint, ModelMetrics};
use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_node_reads_model_from_contract() {
    let env = DevnetEnvironment::new().await.unwrap();

    // Create and pass a governance proposal for dnabert2.
    let proposal = dnabert2_proposal();
    let proposal_id = env.create_proposal(proposal).await.unwrap();

    // Stake voters and vote yes to reach quorum.
    let voters = env.create_voters(5).await.unwrap();
    for voter in &voters {
        env.vote(voter, proposal_id, true).await.unwrap();
    }

    // Execute the proposal and refresh the seed-node registry cache.
    env.execute_proposal(proposal_id).await.unwrap();
    env.mine_blocks(1).await.unwrap();

    let models = env.active_models().await.unwrap();
    assert!(models.iter().any(|m| m.model_id == "dnabert2"), "dnabert2 should be active on-chain");

    // A miner with enough stake should be allowed to train dnabert2.
    let can_mine = env.can_mine_model("", "dnabert2", 20_000).await;
    assert!(can_mine, "stake 20k should satisfy min_stake_to_train 10k");
}

#[tokio::test]
#[serial]
async fn test_checkpoint_serialization_roundtrip() {
    // The Xenomorph protocol serializes checkpoints with Borsh before storing
    // them on the node. This test verifies the seed-node codec.
    let checkpoint = ModelCheckpoint {
        block_height: 1000,
        model_id: "dnabert2".to_string(),
        version: 1,
        weights_hash: [1u8; 32],
        metrics: ModelMetrics { loss: 0.5, accuracy: Some(0.9), f1_score: None, precision: None, recall: None },
        encryption: seed_node::model::checkpoint::EncryptionData::default(),
    };

    let bytes = checkpoint.serialize().unwrap();
    let recovered = ModelCheckpoint::deserialize(&bytes).unwrap();
    assert_eq!(checkpoint.model_id, recovered.model_id);
    assert_eq!(checkpoint.weights_hash, recovered.weights_hash);
    assert_eq!(checkpoint.metrics.loss, recovered.metrics.loss);
}
