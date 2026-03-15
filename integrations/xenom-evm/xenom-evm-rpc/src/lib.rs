use std::sync::Arc;

use jsonrpsee::{
    core::{async_trait, server::SubscriptionMessage, RpcResult, SubscriptionResult},
    proc_macros::rpc,
    server::{PendingSubscriptionSink, Server},
    types::ErrorObject,
};
use xenom_evm_core::{Address, BlockRecord, Bytes, B256, U256};
use serde::{Deserialize, Serialize};
use xenom_evm_core::EvmChain;

// ── RPC trait ─────────────────────────────────────────────────────────────────

#[rpc(server)]
pub trait EthRpc {
    #[method(name = "eth_chainId")]
    async fn chain_id(&self) -> RpcResult<String>;

    #[method(name = "net_version")]
    async fn net_version(&self) -> RpcResult<String>;

    #[method(name = "eth_blockNumber")]
    async fn block_number(&self) -> RpcResult<String>;

    #[method(name = "eth_gasPrice")]
    async fn gas_price(&self) -> RpcResult<String>;

    #[method(name = "eth_getBalance")]
    async fn get_balance(&self, addr: String, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_getTransactionCount")]
    async fn get_transaction_count(&self, addr: String, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_getCode")]
    async fn get_code(&self, addr: String, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_sendRawTransaction")]
    async fn send_raw_transaction(&self, raw: String) -> RpcResult<String>;

    #[method(name = "eth_call")]
    async fn eth_call(&self, req: CallRequest, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_estimateGas")]
    async fn estimate_gas(&self, req: CallRequest, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_getTransactionReceipt")]
    async fn get_transaction_receipt(&self, hash: String) -> RpcResult<Option<serde_json::Value>>;

    #[method(name = "eth_faucet")]
    async fn faucet(&self, addr: String, amount_eth: String) -> RpcResult<String>;

    #[method(name = "xenom_latestStateRoot")]
    async fn latest_state_root(&self) -> RpcResult<String>;

    #[method(name = "xenom_pendingCount")]
    async fn pending_count(&self) -> RpcResult<String>;

    #[method(name = "eth_getBlockByNumber")]
    async fn get_block_by_number(&self, tag: String, full_tx: bool) -> RpcResult<serde_json::Value>;

    #[method(name = "eth_getBlockByHash")]
    async fn get_block_by_hash(&self, hash: String, full_tx: bool) -> RpcResult<serde_json::Value>;

    #[method(name = "eth_getTransactionByHash")]
    async fn get_transaction_by_hash(&self, hash: String) -> RpcResult<Option<serde_json::Value>>;

    #[method(name = "eth_syncing")]
    async fn syncing(&self) -> RpcResult<bool>;

    #[method(name = "web3_clientVersion")]
    async fn client_version(&self) -> RpcResult<String>;

    #[method(name = "eth_getLogs")]
    async fn get_logs(&self, _filter: serde_json::Value) -> RpcResult<Vec<serde_json::Value>>;

    #[method(name = "eth_getFilterChanges")]
    async fn get_filter_changes(&self, _filter_id: String) -> RpcResult<Vec<serde_json::Value>>;

    #[method(name = "eth_newFilter")]
    async fn new_filter(&self, _filter: serde_json::Value) -> RpcResult<String>;

    #[method(name = "eth_uninstallFilter")]
    async fn uninstall_filter(&self, _filter_id: String) -> RpcResult<bool>;

    #[method(name = "eth_getStorageAt")]
    async fn get_storage_at(&self, addr: String, slot: String, _tag: Option<String>) -> RpcResult<String>;

    #[method(name = "eth_maxPriorityFeePerGas")]
    async fn max_priority_fee_per_gas(&self) -> RpcResult<String>;

    #[method(name = "eth_feeHistory")]
    async fn fee_history(&self, block_count: serde_json::Value, newest_block: String, _reward_percentiles: Option<Vec<f64>>) -> RpcResult<serde_json::Value>;

    #[subscription(name = "eth_subscribe", item = serde_json::Value, unsubscribe = "eth_unsubscribe")]
    async fn subscribe(&self, kind: String, params: Option<serde_json::Value>) -> SubscriptionResult;

    /// Devnet anchor: store arbitrary hex payload, returns keccak256 anchor ID.
    #[method(name = "xenom_anchor")]
    async fn xenom_anchor(&self, payload_hex: String) -> RpcResult<String>;

    /// Retrieve a previously stored anchor by its ID (hex).
    #[method(name = "xenom_getAnchor")]
    async fn xenom_get_anchor(&self, anchor_id: String) -> RpcResult<Option<serde_json::Value>>;
}

// ── CallRequest ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRequest {
    pub from: Option<String>,
    pub to: Option<String>,
    pub data: Option<String>,
    pub value: Option<String>,
    pub gas: Option<String>,
}

// ── Server impl ───────────────────────────────────────────────────────────────

pub struct EthServer {
    chain: Arc<EvmChain>,
}

impl EthServer {
    pub fn new(chain: Arc<EvmChain>) -> Self {
        Self { chain }
    }
}

fn rpc_err(msg: impl ToString) -> ErrorObject<'static> {
    ErrorObject::owned(-32000, msg.to_string(), None::<()>)
}

fn parse_addr(s: &str) -> Result<Address, ErrorObject<'static>> {
    let s = s.strip_prefix("0x").unwrap_or(s);
    let bytes = hex::decode(s).map_err(|e| rpc_err(format!("invalid address: {e}")))?;
    if bytes.len() != 20 {
        return Err(rpc_err("address must be 20 bytes"));
    }
    Ok(Address::from_slice(&bytes))
}

fn parse_hex_u256(s: &str) -> U256 {
    let s = s.strip_prefix("0x").unwrap_or(s);
    U256::from_str_radix(s, 16).unwrap_or(U256::ZERO)
}

fn parse_hex_bytes(s: &str) -> Bytes {
    let s = s.strip_prefix("0x").unwrap_or(s);
    hex::decode(s).unwrap_or_default().into()
}

#[async_trait]
impl EthRpcServer for EthServer {
    async fn chain_id(&self) -> RpcResult<String> {
        Ok(format!("0x{:x}", self.chain.chain_id))
    }

    async fn net_version(&self) -> RpcResult<String> {
        Ok(self.chain.chain_id.to_string())
    }

    async fn block_number(&self) -> RpcResult<String> {
        Ok(format!("0x{:x}", self.chain.block_number()))
    }

    async fn gas_price(&self) -> RpcResult<String> {
        Ok("0x3b9aca00".to_string()) // 1 gwei
    }

    async fn get_balance(&self, addr: String, _tag: Option<String>) -> RpcResult<String> {
        let a = parse_addr(&addr)?;
        Ok(format!("0x{:x}", self.chain.balance(a)))
    }

    async fn get_transaction_count(&self, addr: String, _tag: Option<String>) -> RpcResult<String> {
        let a = parse_addr(&addr)?;
        Ok(format!("0x{:x}", self.chain.nonce(a)))
    }

    async fn get_code(&self, addr: String, _tag: Option<String>) -> RpcResult<String> {
        let a = parse_addr(&addr)?;
        let code = self.chain.code_at(a);
        Ok(format!("0x{}", hex::encode(code)))
    }

    async fn send_raw_transaction(&self, raw: String) -> RpcResult<String> {
        let raw_hex = raw.strip_prefix("0x").unwrap_or(&raw);
        let raw_bytes = hex::decode(raw_hex).map_err(|e| rpc_err(format!("hex decode: {e}")))?;
        let hash = self.chain.send_raw_transaction(&raw_bytes).map_err(|e| rpc_err(e))?;
        Ok(format!("0x{}", hex::encode(hash)))
    }

    async fn eth_call(&self, req: CallRequest, _tag: Option<String>) -> RpcResult<String> {
        let to = req.to.as_deref().map(parse_addr).transpose()?.unwrap_or(Address::ZERO);
        let from = req.from.as_deref().map(parse_addr).transpose()?;
        let data = req.data.as_deref().map(parse_hex_bytes).unwrap_or_default();
        let value = req.value.as_deref().map(parse_hex_u256).unwrap_or(U256::ZERO);
        let gas = req.gas.as_deref()
            .and_then(|s| u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok())
            .unwrap_or(30_000_000);
        let out = self.chain.call(from, to, data, value, gas).map_err(|e| rpc_err(e))?;
        Ok(format!("0x{}", hex::encode(out)))
    }

    async fn estimate_gas(&self, req: CallRequest, _tag: Option<String>) -> RpcResult<String> {
        let from = req.from.as_deref().and_then(|s| parse_addr(s).ok());
        let value = req.value.as_deref()
            .and_then(|s| u128::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .map(U256::from).unwrap_or(U256::ZERO);
        let data: Bytes = req.data.as_deref()
            .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
            .unwrap_or_default().into();
        let gas_limit = req.gas.as_deref()
            .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
            .unwrap_or(30_000_000);
        // If no `to`, treat as contract creation estimate
        if req.to.is_none() {
            return Ok(format!("0x{:x}", 500_000u64));
        }
        let to = parse_addr(req.to.as_deref().unwrap_or("")).map_err(|e| rpc_err(e))?;
        match self.chain.call(from, to, data, value, gas_limit) {
            Ok(_) => {
                // Simulate succeeded — return a reasonable estimate with 20% buffer
                let base = 21_000u64;
                let data_cost = req.data.as_deref()
                    .and_then(|s| hex::decode(s.trim_start_matches("0x")).ok())
                    .map(|b| b.iter().map(|&x| if x == 0 { 4u64 } else { 16 }).sum::<u64>())
                    .unwrap_or(0);
                Ok(format!("0x{:x}", (base + data_cost) * 12 / 10))
            }
            Err(_) => Ok("0x5208".to_string()), // fallback 21000
        }
    }

    async fn get_transaction_receipt(&self, hash: String) -> RpcResult<Option<serde_json::Value>> {
        let h = hash.strip_prefix("0x").unwrap_or(&hash);
        let bytes = hex::decode(h).map_err(|e| rpc_err(e))?;
        if bytes.len() != 32 {
            return Ok(None);
        }
        let b256 = B256::from_slice(&bytes);
        match self.chain.receipt(b256) {
            Some(r) => Ok(Some(serde_json::to_value(r).unwrap_or(serde_json::Value::Null))),
            None => Ok(None),
        }
    }

    async fn faucet(&self, addr: String, amount_eth: String) -> RpcResult<String> {
        let a = parse_addr(&addr)?;
        let eth: u64 = amount_eth.parse().unwrap_or(1);
        let wei = U256::from(eth) * U256::from(1_000_000_000_000_000_000u128);
        self.chain.fund(a, wei);
        Ok(format!("funded {} with {eth} ETH", addr))
    }

    async fn latest_state_root(&self) -> RpcResult<String> {
        let root = self.chain.latest_state_root();
        Ok(format!("0x{}", hex::encode(root)))
    }

    async fn pending_count(&self) -> RpcResult<String> {
        Ok(format!("0x{:x}", self.chain.pending_count()))
    }

    async fn syncing(&self) -> RpcResult<bool> {
        Ok(false) // always fully synced (in-memory chain)
    }

    async fn client_version(&self) -> RpcResult<String> {
        Ok(format!("xenom-evm/0.15.2/chain={}", self.chain.chain_id))
    }

    async fn get_logs(&self, filter: serde_json::Value) -> RpcResult<Vec<serde_json::Value>> {
        let current = self.chain.block_number();

        let parse_block_tag = |v: Option<&serde_json::Value>| -> u64 {
            match v {
                None => current,
                Some(serde_json::Value::String(s)) => match s.as_str() {
                    "latest" | "pending" | "safe" | "finalized" => current,
                    "earliest" => 1,
                    hex => u64::from_str_radix(hex.trim_start_matches("0x"), 16).unwrap_or(current),
                },
                Some(serde_json::Value::Number(n)) => n.as_u64().unwrap_or(current),
                _ => current,
            }
        };

        let from = parse_block_tag(filter.get("fromBlock"));
        let to   = parse_block_tag(filter.get("toBlock"));

        // address filter: string or array-of-one
        let addr_filter: Option<String> = filter.get("address").and_then(|v| {
            v.as_str().map(|s| s.to_lowercase())
                .or_else(|| v.as_array()?.first()?.as_str().map(|s| s.to_lowercase()))
        });

        // topics[0] filter
        let topic0_filter: Option<String> = filter.get("topics")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str().map(|s| s.to_string()));

        let logs = self.chain.get_logs_for_blocks(
            from, to,
            addr_filter.as_deref(),
            topic0_filter.as_deref(),
        );

        Ok(logs.into_iter()
            .map(|l| serde_json::to_value(l).unwrap_or(serde_json::Value::Null))
            .collect())
    }

    async fn get_filter_changes(&self, _filter_id: String) -> RpcResult<Vec<serde_json::Value>> {
        Ok(vec![])
    }

    async fn new_filter(&self, _filter: serde_json::Value) -> RpcResult<String> {
        Ok("0x1".to_string()) // stub filter id
    }

    async fn uninstall_filter(&self, _filter_id: String) -> RpcResult<bool> {
        Ok(true)
    }

    async fn get_storage_at(&self, addr: String, slot: String, _tag: Option<String>) -> RpcResult<String> {
        let a = parse_addr(&addr).map_err(|e| rpc_err(e))?;
        let raw = slot.trim_start_matches("0x");
        let padded = if raw.len() % 2 == 1 { format!("0{raw}") } else { raw.to_string() };
        let slot_bytes = hex::decode(&padded).map_err(|e| rpc_err(e))?;
        let mut slot_arr = [0u8; 32];
        let start = 32usize.saturating_sub(slot_bytes.len());
        slot_arr[start..].copy_from_slice(&slot_bytes[..slot_bytes.len().min(32)]);
        let slot_b256 = B256::from(slot_arr);
        let val = self.chain.storage_at(a, slot_b256);
        Ok(format!("0x{}", hex::encode(val)))
    }

    async fn max_priority_fee_per_gas(&self) -> RpcResult<String> {
        Ok("0x3b9aca00".to_string()) // 1 gwei tip
    }

    async fn fee_history(&self, block_count: serde_json::Value, _newest_block: String, _reward_percentiles: Option<Vec<f64>>) -> RpcResult<serde_json::Value> {
        let count = block_count.as_u64().or_else(|| {
            block_count.as_str().and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok())
        }).unwrap_or(1).min(1024);
        let base_fee = "0x3b9aca00";
        Ok(serde_json::json!({
            "oldestBlock": format!("0x{:x}", self.chain.block_number().saturating_sub(count)),
            "baseFeePerGas": std::iter::repeat(base_fee).take(count as usize + 1).collect::<Vec<_>>(),
            "gasUsedRatio": std::iter::repeat(0.0f64).take(count as usize).collect::<Vec<_>>(),
            "reward": serde_json::json!([])
        }))
    }

    async fn get_block_by_number(&self, tag: String, _full_tx: bool) -> RpcResult<serde_json::Value> {
        let block_num = match tag.as_str() {
            "latest" | "pending" | "safe" | "finalized" => self.chain.block_number(),
            "earliest" => 1,
            hex => u64::from_str_radix(hex.strip_prefix("0x").unwrap_or(hex), 16)
                .unwrap_or(self.chain.block_number()),
        };
        Ok(build_block_object(block_num, &self.chain))
    }

    async fn get_block_by_hash(&self, hash: String, _full_tx: bool) -> RpcResult<serde_json::Value> {
        let h = hash.strip_prefix("0x").unwrap_or(&hash);
        let bytes = hex::decode(h).unwrap_or_default();
        if bytes.len() != 32 { return Ok(serde_json::Value::Null); }
        let b256 = B256::from_slice(&bytes);
        match self.chain.get_block_by_hash(b256) {
            Some(rec) => Ok(block_record_to_json(&rec)),
            None => Ok(serde_json::Value::Null),
        }
    }

    async fn get_transaction_by_hash(&self, hash: String) -> RpcResult<Option<serde_json::Value>> {
        let h = hash.strip_prefix("0x").unwrap_or(&hash);
        let bytes = hex::decode(h).unwrap_or_default();
        if bytes.len() != 32 { return Ok(None); }
        let b256 = B256::from_slice(&bytes);
        match self.chain.receipt(b256) {
            Some(r) => {
                let block_num = self.chain.get_tx_block(b256).unwrap_or(0);
                let obj = serde_json::json!({
                    "hash": r.transaction_hash,
                    "from": r.from,
                    "to": r.to,
                    "blockNumber": format!("0x{:x}", block_num),
                    "blockHash": format!("0x{}", hex::encode(
                        self.chain.get_block(block_num).map(|b| *b.hash.as_ref()).unwrap_or([0u8; 32])
                    )),
                    "transactionIndex": r.transaction_index,
                    "gas": r.gas_used,
                    "gasPrice": "0x3b9aca00",
                    "value": "0x0",
                    "input": "0x",
                    "nonce": "0x0",
                    "type": r.tx_type,
                    "v": "0x1",
                    "r": "0x0",
                    "s": "0x0"
                });
                Ok(Some(obj))
            }
            None => Ok(None),
        }
    }

    async fn xenom_anchor(&self, payload_hex: String) -> RpcResult<String> {
        let raw = payload_hex.strip_prefix("0x").unwrap_or(&payload_hex);
        let data = hex::decode(raw).map_err(|e| rpc_err(format!("invalid hex: {e}")))?;
        let id = self.chain.anchor_data(data);
        Ok(format!("0x{}", hex::encode(id)))
    }

    async fn xenom_get_anchor(&self, anchor_id: String) -> RpcResult<Option<serde_json::Value>> {
        let raw = anchor_id.strip_prefix("0x").unwrap_or(&anchor_id);
        let bytes = hex::decode(raw).map_err(|e| rpc_err(format!("invalid id: {e}")))?;
        if bytes.len() != 32 { return Ok(None); }
        let id = B256::from_slice(&bytes);
        match self.chain.get_anchor(id) {
            Some((block_num, data)) => Ok(Some(serde_json::json!({
                "anchorId":   anchor_id,
                "blockNumber": format!("0x{block_num:x}"),
                "payloadHex": format!("0x{}", hex::encode(&data)),
                "payloadLen": data.len(),
            }))),
            None => Ok(None),
        }
    }

    async fn subscribe(
        &self,
        pending: PendingSubscriptionSink,
        kind: String,
        params: Option<serde_json::Value>,
    ) -> SubscriptionResult {
        match kind.as_str() {
            "newHeads" => {
                let sink = pending.accept().await?;
                let mut rx = self.chain.subscribe_blocks();
                tokio::spawn(async move {
                    loop {
                        match rx.recv().await {
                            Ok(rec) => {
                                let val = block_record_to_json(&rec);
                                if let Ok(msg) = SubscriptionMessage::from_json(&val) {
                                    if sink.send(msg).await.is_err() { break; }
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        }
                    }
                });
            }
            "logs" => {
                let sink = pending.accept().await?;
                let mut rx = self.chain.subscribe_blocks();
                let addr_filter = params.as_ref()
                    .and_then(|p| p.get("address"))
                    .and_then(|v| v.as_str().map(|s| s.to_lowercase()));
                let topic0_filter = params.as_ref()
                    .and_then(|p| p.get("topics"))
                    .and_then(|v| v.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|v| v.as_str().map(|s| s.to_string()));
                let chain = Arc::clone(&self.chain);
                tokio::spawn(async move {
                    loop {
                        match rx.recv().await {
                            Ok(rec) => {
                                let logs = chain.get_logs_for_blocks(
                                    rec.number, rec.number,
                                    addr_filter.as_deref(),
                                    topic0_filter.as_deref(),
                                );
                                for log in logs {
                                    let val = serde_json::to_value(log)
                                        .unwrap_or(serde_json::Value::Null);
                                    if let Ok(msg) = SubscriptionMessage::from_json(&val) {
                                        if sink.send(msg).await.is_err() { break; }
                                    }
                                }
                            }
                            Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                            Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        }
                    }
                });
            }
            other => {
                pending.reject(ErrorObject::owned(
                    -32602,
                    format!("unsupported eth_subscribe kind: {other}"),
                    None::<()>,
                )).await;
            }
        }

        Ok(())
    }
}

// ── Block helpers ─────────────────────────────────────────────────────────────

fn block_record_to_json(rec: &BlockRecord) -> serde_json::Value {
    serde_json::json!({
        "number":     format!("0x{:x}", rec.number),
        "hash":       format!("0x{}", hex::encode(rec.hash)),
        "parentHash": format!("0x{}", hex::encode(rec.parent_hash)),
        "stateRoot":  format!("0x{}", hex::encode(rec.state_root)),
        "timestamp":  format!("0x{:x}", rec.timestamp),
        "gasUsed":    format!("0x{:x}", rec.gas_used),
        "gasLimit":   "0x1c9c380", // 30M
        "baseFeePerGas": "0x3b9aca00", // 1 gwei
        "miner":      "0x0000000000000000000000000000000000000000",
        "difficulty": "0x0",
        "totalDifficulty": "0x0",
        "nonce":      "0x0000000000000000",
        "extraData":  "0x",
        "logsBloom":  "0x".to_string() + &"0".repeat(512),
        "sha3Uncles": "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
        "transactionsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
        "receiptsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
        "mixHash":    "0x0000000000000000000000000000000000000000000000000000000000000000",
        "size":       "0x220",
        "uncles":     serde_json::json!([]),
        "transactions": rec.tx_hashes.iter()
            .map(|h| format!("0x{}", hex::encode(h)))
            .collect::<Vec<_>>(),
    })
}

fn build_block_object(block_num: u64, chain: &EvmChain) -> serde_json::Value {
    match chain.get_block(block_num) {
        Some(rec) => block_record_to_json(&rec),
        None => {
            // Block not yet recorded (e.g. genesis or future): return synthetic stub
            let state_root = chain.latest_state_root();
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
            serde_json::json!({
                "number":     format!("0x{:x}", block_num),
                "hash":       format!("0x{:064x}", block_num),
                "parentHash": format!("0x{:064x}", block_num.saturating_sub(1)),
                "stateRoot":  format!("0x{}", hex::encode(state_root)),
                "timestamp":  format!("0x{:x}", now),
                "gasUsed":    "0x0",
                "gasLimit":   "0x1c9c380",
                "baseFeePerGas": "0x3b9aca00",
                "miner":      "0x0000000000000000000000000000000000000000",
                "difficulty": "0x0",
                "totalDifficulty": "0x0",
                "nonce":      "0x0000000000000000",
                "extraData":  "0x",
                "logsBloom":  "0x".to_string() + &"0".repeat(512),
                "sha3Uncles": "0x1dcc4de8dec75d7aab85b567b6ccd41ad312451b948a7413f0a142fd40d49347",
                "transactionsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
                "receiptsRoot": "0x56e81f171bcc55a6ff8345e692c0f86e5b48e01b996cadc001622fb5e363b421",
                "mixHash":    "0x0000000000000000000000000000000000000000000000000000000000000000",
                "size":       "0x220",
                "uncles":     serde_json::json!([]),
                "transactions": serde_json::json!([]),
            })
        }
    }
}

// ── Start server ──────────────────────────────────────────────────────────────

pub async fn start_rpc_server(chain: Arc<EvmChain>, addr: &str) -> anyhow::Result<jsonrpsee::server::ServerHandle> {
    use tower_http::cors::{Any, CorsLayer};

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let server = Server::builder()
        .set_http_middleware(tower::ServiceBuilder::new().layer(cors))
        .build(addr)
        .await?;

    let module = EthServer::new(chain).into_rpc();
    let handle = server.start(module);
    log::info!("Xenom EVM RPC listening on http://{addr}");
    Ok(handle)
}
