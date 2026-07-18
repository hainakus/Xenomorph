//! Fault tolerance and recovery integration tests.

use crate::setup::DevnetEnvironment;
use crate::support::miner_builder::TestMiner;
use crate::support::node_runner::spawn_mock_seed_node;
use anyhow::Result;
use ethers::prelude::*;
use serial_test::serial;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::timeout;

#[tokio::test]
#[serial]
async fn test_miner_reconnects_after_server_close() -> Result<()> {
    let node = spawn_mock_seed_node().await?;
    let mut miner = TestMiner::new(&node.url).await?;

    // Configure the seed node to close the connection after one response.
    miner.connect().await?;
    node.close_after_n(1);

    // First request: server responds and closes the socket.
    let batch = miner.request_batch("dnabert2").await?;
    let _ = miner.train(batch);

    // Second request: the miner's `ensure_connected` logic reconnects to the
    // same listener and the request succeeds.
    let result = timeout(Duration::from_secs(5), miner.request_batch("dnabert2"))
        .await
        .expect("miner did not reconnect in time")
        .expect("batch request failed after reconnect");
    assert_eq!(result.model_id, "dnabert2");
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_invalid_proof_rejected() -> Result<()> {
    let node = spawn_mock_seed_node().await?;
    let mut miner = TestMiner::new(&node.url).await?;
    miner.connect().await?;

    let batch = miner.request_batch("dnabert2").await?;
    let result = miner.train(batch)?;

    let submit = miner.submit_invalid_block(result).await;
    assert!(submit.is_err(), "invalid proof should be rejected");
    assert_eq!(node.accepted_blocks(), 0);
    Ok(())
}

#[tokio::test]
#[serial]
async fn test_double_spend_protection() -> Result<()> {
    let env = DevnetEnvironment::new().await?;
    let user = env.voter_wallet(1)?;

    let user_client = Arc::new(SignerMiddleware::new_with_provider_chain(env.contracts.provider.clone(), user.clone()).await?);
    let faucet_client =
        Arc::new(SignerMiddleware::new_with_provider_chain(env.contracts.provider.clone(), env.contracts.faucet.clone()).await?);

    let usdt_abi_bytes = std::fs::read(format!("{}/fixtures/contracts/MockUSDT.abi", env!("CARGO_MANIFEST_DIR")))?;
    let usdt_abi = ethers::abi::Abi::load(usdt_abi_bytes.as_slice())?;
    let usdt = Contract::new(env.contracts.usdt, usdt_abi.clone(), faucet_client.clone());
    let usdt_user = Contract::new(env.contracts.usdt, usdt_abi, user_client.clone());

    let payments_abi_bytes = std::fs::read(format!("{}/fixtures/contracts/InferencePayments.abi", env!("CARGO_MANIFEST_DIR")))?;
    let payments_abi = ethers::abi::Abi::load(payments_abi_bytes.as_slice())?;
    let user_payments = Contract::new(env.contracts.payments, payments_abi.clone(), user_client);
    let admin_payments = Contract::new(env.contracts.payments, payments_abi, faucet_client);

    let amount = U256::from(100u64) * U256::exp10(6);

    // Fund user and approve.
    let mint_call = usdt.method::<(Address, U256), ()>("mint", (user.address(), amount))?;
    let _ = mint_call.send().await?.await?;

    let approve_call = usdt_user.method::<(Address, U256), ()>("approve", (env.contracts.payments, amount))?;
    let _ = approve_call.send().await?.await?;

    // Initiate a query payment.
    let query_id = "double-spend-001";
    let query_hash = crate::fixtures::hex_to_bytes32(query_id);
    let init_call = user_payments
        .method::<(H256, String, U256), ()>("initiatePayment", (H256::from(query_hash), "dnabert2".to_string(), amount))?;
    let _ = init_call.send().await?.await?;

    // First completion succeeds.
    let first_call = admin_payments.method::<H256, ()>("completePayment", H256::from(query_hash))?;
    let first = first_call.send().await?.await?;
    assert!(first.is_some(), "first completePayment should succeed");

    // Second completion must fail because the query is already completed.
    let second_call = admin_payments.method::<H256, ()>("completePayment", H256::from(query_hash))?;
    let second = second_call.send().await;
    assert!(second.is_err(), "double spend should be rejected");

    Ok(())
}
