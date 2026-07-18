//! Full governance lifecycle: propose -> vote -> execute -> mine.

use crate::fixtures::evo2_proposal;
use crate::setup::DevnetEnvironment;
use serial_test::serial;

#[tokio::test]
#[serial]
async fn test_full_governance_lifecycle() {
    let env = DevnetEnvironment::new().await.unwrap();

    // PASSO 1: Create a governance proposal for evo2-7b.
    let proposal = evo2_proposal();
    let proposal_id = env.create_proposal(proposal).await.unwrap();
    println!("Proposal created: {}", proposal_id);

    // PASSO 2: Stake voters and vote yes (5 voters x 300k = 1.5M, > 1M quorum).
    let voters = env.create_voters(5).await.unwrap();
    for voter in &voters {
        env.vote(voter, proposal_id, true).await.unwrap();
    }

    let proposal_data = env.get_proposal(proposal_id).await.unwrap();
    assert!(proposal_data.yes_votes > proposal_data.no_votes);

    // PASSO 3: Advance time and execute.
    env.execute_proposal(proposal_id).await.unwrap();
    env.mine_blocks(1).await.unwrap();

    // PASSO 4: Seed-node registry sees the activated model.
    let models = env.active_models().await.unwrap();
    let evo2 = models.iter().find(|m| m.model_id == "evo2-7b").expect("evo2-7b should be active");
    assert_eq!(evo2.vram_required, 48);
    assert_eq!(evo2.reward_per_block, 50_000_000_000);

    // PASSO 5: A miner with sufficient stake can train evo2-7b.
    let can_mine = env.can_mine_model("xnom:test", "evo2-7b", 60_000).await;
    assert!(can_mine, "60k stake should satisfy min_stake_to_train 50k");

    // Train and submit a block through the mock seed node.
    let mut miner = env.create_miner().await.unwrap();
    miner.connect().await.unwrap();

    let batch = miner.request_batch("evo2-7b").await.unwrap();
    let result = miner.train(batch).unwrap();
    let block_hash = miner.submit_block(result).await.unwrap();

    let reward = env.get_block_reward(&block_hash).await.unwrap();
    assert_eq!(reward, 50_000_000_000, "reward should match proposal");
}
