//! Deploy Xenomorph smart contracts on a local Anvil instance.

use anyhow::{Context, Result};
use ethers::contract::ContractFactory;
use ethers::prelude::*;
use std::sync::Arc;
use tracing::info;

/// Contracts deployed for a test run.
pub struct DeployedContracts {
    pub endpoint: String,
    pub provider: Provider<Http>,
    pub faucet: LocalWallet,
    pub governance: Address,
    pub payments: Address,
    pub usdt: Address,
}

fn load_artifact(name: &str) -> Result<(ethers::abi::Abi, Bytes)> {
    let abi_path = format!("{}/fixtures/contracts/{}.abi", env!("CARGO_MANIFEST_DIR"), name);
    let bin_path = format!("{}/fixtures/contracts/{}.bin", env!("CARGO_MANIFEST_DIR"), name);

    let abi_json = std::fs::read_to_string(&abi_path).with_context(|| format!("failed to read ABI for {}", name))?;
    let abi: ethers::abi::Abi = serde_json::from_str(&abi_json).with_context(|| format!("failed to parse ABI for {}", name))?;

    let bin_hex =
        std::fs::read_to_string(&bin_path).with_context(|| format!("failed to read bytecode for {}", name))?.trim().to_string();
    let bin = hex::decode(&bin_hex).with_context(|| format!("failed to decode bytecode for {}", name))?;

    Ok((abi, Bytes::from(bin)))
}

fn faucet_wallet(anvil: &ethers::utils::AnvilInstance) -> Result<LocalWallet> {
    let secret_key = anvil.keys()[0].clone();
    let signing_key = ethers::core::k256::ecdsa::SigningKey::from(&secret_key);
    let wallet = LocalWallet::from(signing_key).with_chain_id(anvil.chain_id());
    Ok(wallet)
}

/// Deploy `ModelGovernance`, a mock USDT, and `InferencePayments` on the
/// provided Anvil instance. The faucet wallet is funded with the mock USDT.
pub async fn deploy_contracts(anvil: &ethers::utils::AnvilInstance) -> Result<DeployedContracts> {
    let endpoint = anvil.endpoint();
    let provider = Provider::<Http>::try_from(endpoint.as_str()).context("invalid anvil endpoint")?;
    let faucet = faucet_wallet(anvil)?;
    let faucet_address = faucet.address();

    let client = Arc::new(SignerMiddleware::new_with_provider_chain(provider.clone(), faucet.clone()).await?);

    // Deploy ModelGovernance
    let (gov_abi, gov_bin) = load_artifact("ModelGovernance")?;
    let gov_factory = ContractFactory::new(gov_abi, gov_bin, client.clone());
    let gov_contract = gov_factory.deploy((faucet_address,))?.send().await.context("failed to deploy ModelGovernance")?;
    let governance = gov_contract.address();
    info!("ModelGovernance deployed at {}", governance);

    // Deploy MockUSDT (6 decimals) with a large initial supply to the faucet.
    let (usdt_abi, usdt_bin) = load_artifact("MockUSDT")?;
    let usdt_factory = ContractFactory::new(usdt_abi, usdt_bin, client.clone());
    let initial_supply = U256::from(10_000_000_000u64) * U256::exp10(6);
    let usdt_contract = usdt_factory.deploy((initial_supply,))?.send().await.context("failed to deploy MockUSDT")?;
    let usdt = usdt_contract.address();
    info!("MockUSDT deployed at {}", usdt);

    // Deploy InferencePayments
    let (payments_abi, payments_bin) = load_artifact("InferencePayments")?;
    let payments_factory = ContractFactory::new(payments_abi, payments_bin, client.clone());
    let payments_contract = payments_factory
        .deploy((usdt, faucet_address, faucet_address, faucet_address))?
        .send()
        .await
        .context("failed to deploy InferencePayments")?;
    let payments = payments_contract.address();
    info!("InferencePayments deployed at {}", payments);

    Ok(DeployedContracts { endpoint, provider, faucet, governance, payments, usdt })
}
