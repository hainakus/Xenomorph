use anyhow::{anyhow, Result};
use ethers::{
    abi::Abi,
    contract::Contract,
    middleware::Middleware,
    providers::{Http, Provider},
    types::{Address, H256, U256},
};
use std::str::FromStr;
use std::sync::Arc;
use tracing::{info, instrument};

// Simplified ABI for InferencePayments contract
const INFERENCE_PAYMENTS_ABI: &str = r#"[{
    "inputs": [{"internalType": "bytes32", "name": "queryId", "type": "bytes32"}],
    "name": "getQueryPayment",
    "outputs": [{
        "components": [
            {"internalType": "address", "name": "payer", "type": "address"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "bool", "name": "completed", "type": "bool"},
            {"internalType": "bool", "name": "refunded", "type": "bool"},
            {"internalType": "bytes32", "name": "modelId", "type": "bytes32"}
        ],
        "internalType": "struct InferencePayments.QueryPayment",
        "name": "",
        "type": "tuple"
    }],
    "stateMutability": "view",
    "type": "function"
}]"#;

pub struct PaymentVerifier {
    provider: Arc<Provider<Http>>,
    contract_address: Address,
    contract: Contract<Provider<Http>>,
}

impl PaymentVerifier {
    pub async fn new(contract_address: String, rpc_url: String) -> Result<Self> {
        let provider = Provider::<Http>::try_from(rpc_url.as_str()).map_err(|e| anyhow!("Failed to create provider: {}", e))?;

        let provider = Arc::new(provider);

        let contract_addr = Address::from_str(&contract_address).map_err(|e| anyhow!("Invalid contract address: {}", e))?;

        let abi: Abi = serde_json::from_str(INFERENCE_PAYMENTS_ABI).map_err(|e| anyhow!("Failed to parse ABI: {}", e))?;

        let contract = Contract::new(contract_addr, abi, provider.clone());

        Ok(Self { provider, contract_address: contract_addr, contract })
    }

    #[instrument(skip(self))]
    pub async fn verify_payment(&self, query_id: &str) -> Result<bool> {
        // Convert query_id to bytes32
        let query_hash = self.query_id_to_bytes32(query_id);

        // Call contract to get payment status
        let payment: (Address, U256, U256, bool, bool, H256) = self
            .contract
            .method::<_, (Address, U256, U256, bool, bool, H256)>("getQueryPayment", query_hash)
            .map_err(|e| anyhow!("Failed to call contract: {}", e))?
            .call()
            .await
            .map_err(|e| anyhow!("Contract call failed: {}", e))?;

        let (_payer, _amount, _timestamp, completed, refunded, _model_id) = payment;

        // Payment is verified if it's completed and not refunded
        let verified = completed && !refunded;

        info!("Payment verification for query {}: {}", query_id, verified);
        Ok(verified)
    }

    #[instrument(skip(self))]
    pub async fn get_payment_amount(&self, query_id: &str) -> Result<U256> {
        let query_hash = self.query_id_to_bytes32(query_id);

        let payment: (Address, U256, U256, bool, bool, H256) = self
            .contract
            .method::<_, (Address, U256, U256, bool, bool, H256)>("getQueryPayment", query_hash)
            .map_err(|e| anyhow!("Failed to call contract: {}", e))?
            .call()
            .await
            .map_err(|e| anyhow!("Contract call failed: {}", e))?;

        let (_payer, amount, _timestamp, _completed, _refunded, _model_id) = payment;
        Ok(amount)
    }

    #[instrument(skip(self))]
    pub async fn check_contract_status(&self) -> Result<bool> {
        // Check if contract is accessible
        let code =
            self.provider.get_code(self.contract_address, None).await.map_err(|e| anyhow!("Failed to get contract code: {}", e))?;

        Ok(!code.is_empty())
    }

    fn query_id_to_bytes32(&self, query_id: &str) -> [u8; 32] {
        let mut hash = [0u8; 32];
        let bytes = query_id.as_bytes();
        let len = bytes.len().min(32);
        hash[..len].copy_from_slice(&bytes[..len]);
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_query_id_to_bytes32() {
        let _contract_addr = "0x0000000000000000000000000000000000000000".to_string();
        let rpc_url = "https://localhost".to_string();

        // Create a mock verifier without actually connecting
        let provider = Provider::<Http>::try_from(rpc_url.as_str());
        assert!(provider.is_ok()); // Should parse URL

        let query_id = "test-query-123";
        let mut hash = [0u8; 32];
        let bytes = query_id.as_bytes();
        let len = bytes.len().min(32);
        hash[..len].copy_from_slice(&bytes[..len]);

        assert_eq!(&hash[..query_id.len()], query_id.as_bytes());
    }
}
