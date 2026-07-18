//! USDT payment flow for inference queries.

use crate::setup::DevnetEnvironment;
use anyhow::Result;
use api_gateway::payments::verifier::PaymentVerifier;
use ethers::prelude::*;
use serial_test::serial;
use std::sync::Arc;

#[tokio::test]
#[serial]
async fn test_user_pays_for_inference() -> Result<()> {
    let env = DevnetEnvironment::new().await?;

    // User wallet is one of the anvil dev accounts.
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

    // 100 USDT (6 decimals).
    let amount = U256::from(100u64) * U256::exp10(6);

    // Faucet mints USDT to the user.
    let mint_call = usdt.method::<(Address, U256), ()>("mint", (user.address(), amount))?;
    let mint = mint_call.send().await?.await?;
    assert!(mint.is_some(), "mint should be mined");

    // User approves the InferencePayments contract on the USDT token.
    let approve_call = usdt_user.method::<(Address, U256), ()>("approve", (env.contracts.payments, amount))?;
    let approve = approve_call.send().await?.await?;
    assert!(approve.is_some(), "approve should be mined");

    // User initiates payment for a query.
    let query_id = "query-001";
    let query_hash = crate::fixtures::hex_to_bytes32(query_id);
    let initiate_call = user_payments
        .method::<(H256, String, U256), ()>("initiatePayment", (H256::from(query_hash), "dnabert2".to_string(), amount))?;
    let initiate = initiate_call.send().await?.await?;
    assert!(initiate.is_some(), "initiatePayment should be mined");

    // Build the payment verifier used by the API gateway.
    let verifier = PaymentVerifier::new(format!("{:?}", env.contracts.payments), env.contracts.endpoint.clone()).await?;

    // Payment is recorded but not completed yet.
    let verified_before = verifier.verify_payment(query_id).await?;
    assert!(!verified_before, "payment should not be verified before completion");

    // Admin completes the payment.
    let complete_call = admin_payments.method::<H256, ()>("completePayment", H256::from(query_hash))?;
    let complete = complete_call.send().await?.await?;
    assert!(complete.is_some(), "completePayment should be mined");

    // Payment is now verified.
    let verified_after = verifier.verify_payment(query_id).await?;
    assert!(verified_after, "payment should be verified after completion");

    let payment_amount = verifier.get_payment_amount(query_id).await?;
    assert_eq!(payment_amount, amount, "payment amount mismatch");

    Ok(())
}
