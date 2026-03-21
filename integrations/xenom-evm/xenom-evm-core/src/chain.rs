use std::{
    collections::{HashMap, VecDeque},
    sync::atomic::{AtomicU64, Ordering},
};
use tokio::sync::broadcast;

use parking_lot::Mutex;
use revm::{
    db::InMemoryDB,
    primitives::{AccountInfo, Address, Bytes, ExecutionResult, Output, TransactTo, TxEnv, B256, U256},
    DatabaseRef, Evm,
};
use thiserror::Error;

use crate::backend::{InMemoryBackend, RocksDbBackend, StateBackend};
use crate::checkpoint::{anchor_root, genetics_settlement_root, receipts_root, tx_root, L1CheckpointV1, CHECKPOINT_VERSION_V1};
use crate::types::{decode_raw_tx, DecodedTx, EvmLog, EvmReceipt};

// ── Block record ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct BlockRecord {
    pub number: u64,
    pub hash: B256,
    pub parent_hash: B256,
    pub timestamp: u64,
    pub gas_used: u64,
    pub tx_hashes: Vec<B256>,
    pub state_root: B256,
}

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ChainError {
    #[error("tx decode: {0}")]
    Decode(#[from] crate::types::TxError),
    #[error("evm execution: {0}")]
    Evm(String),
    #[error("nonce mismatch: expected {expected}, got {got}")]
    Nonce { expected: u64, got: u64 },
}

// ── EvmChain ──────────────────────────────────────────────────────────────────

pub struct EvmChain {
    pub chain_id: u64,
    db: Mutex<InMemoryDB>,
    block_number: AtomicU64,
    tx_index: AtomicU64,
    receipts: Mutex<HashMap<B256, EvmReceipt>>,
    mempool: Mutex<VecDeque<(B256, Vec<u8>)>>, // (hash, raw_rlp)
    latest_state_root: Mutex<B256>,
    blocks: Mutex<HashMap<u64, BlockRecord>>,
    tx_to_block: Mutex<HashMap<B256, u64>>,       // tx_hash → block_number
    block_logs: Mutex<HashMap<u64, Vec<EvmLog>>>, // block_number → all logs in that block
    backend: Box<dyn StateBackend>,
    block_tx: broadcast::Sender<BlockRecord>,
    anchors: Mutex<HashMap<B256, (u64, Vec<u8>)>>, // keccak(data) → (block_num, data)
    latest_checkpoint: Mutex<Option<L1CheckpointV1>>,
}

impl EvmChain {
    pub fn new(chain_id: u64) -> Self {
        Self::new_inner(chain_id, InMemoryDB::default(), 1, 0, B256::ZERO, Box::new(InMemoryBackend::new()))
    }

    /// Backward-compatible constructor: persisted mode now uses RocksDB.
    pub fn new_with_persistence(chain_id: u64, state_dir: &std::path::Path) -> Self {
        Self::new_with_rocksdb(chain_id, state_dir)
    }

    /// Create an EvmChain backed by RocksDB.
    ///
    /// Stores snapshots and anchors in RocksDB CFs and restores state from
    /// `meta/state_snapshot_v1` when present.
    pub fn new_with_rocksdb(chain_id: u64, state_dir: &std::path::Path) -> Self {
        std::fs::create_dir_all(state_dir).ok();
        match RocksDbBackend::open(state_dir, chain_id) {
            Ok(backend) => match backend.load_snapshot() {
                Ok(Some(loaded)) => {
                    log::info!(
                        "rocksdb snapshot loaded: block={} tx_index={} root={}",
                        loaded.block_number,
                        loaded.tx_index,
                        loaded.state_root
                    );
                    Self::new_inner(chain_id, loaded.db, loaded.block_number, loaded.tx_index, loaded.state_root, Box::new(backend))
                }
                Ok(None) => {
                    log::info!("rocksdb state is empty — starting fresh");
                    Self::new_inner(chain_id, InMemoryDB::default(), 1, 0, B256::ZERO, Box::new(backend))
                }
                Err(e) => {
                    log::warn!("rocksdb snapshot load failed: {e}. Falling back to in-memory backend.");
                    Self::new(chain_id)
                }
            },
            Err(e) => {
                log::warn!("rocksdb backend init failed: {e}. Falling back to in-memory backend.");
                Self::new(chain_id)
            }
        }
    }

    fn new_inner(
        chain_id: u64,
        db: InMemoryDB,
        block_number: u64,
        tx_index: u64,
        state_root: B256,
        backend: Box<dyn StateBackend>,
    ) -> Self {
        Self {
            chain_id,
            db: Mutex::new(db),
            block_number: AtomicU64::new(block_number),
            tx_index: AtomicU64::new(tx_index),
            receipts: Mutex::new(HashMap::new()),
            mempool: Mutex::new(VecDeque::new()),
            latest_state_root: Mutex::new(state_root),
            blocks: Mutex::new(HashMap::new()),
            tx_to_block: Mutex::new(HashMap::new()),
            block_logs: Mutex::new(HashMap::new()),
            backend,
            block_tx: broadcast::channel(64).0,
            anchors: Mutex::new(HashMap::new()),
            latest_checkpoint: Mutex::new(None),
        }
    }

    /// Store an arbitrary payload anchored at the current block.
    /// Returns keccak256(data) as the anchor ID.
    pub fn anchor_data(&self, data: Vec<u8>) -> B256 {
        let id = B256::from_slice(&crate::types::keccak256(&data));
        let block = self.block_number();
        self.anchors.lock().insert(id, (block, data.clone()));
        if let Err(e) = self.backend.put_anchor(id, block, &data) {
            log::warn!("anchor persist failed: {e}");
        }
        id
    }

    /// List all stored anchors sorted by block number.
    /// Returns (anchor_id_hex, block_number, created_at_ms, payload).
    pub fn list_anchors(&self) -> Vec<(String, u64, u64, Vec<u8>)> {
        match self.backend.list_anchors() {
            Ok(v) => v,
            Err(e) => {
                log::warn!("list_anchors failed: {e}");
                vec![]
            }
        }
    }

    /// Retrieve a previously anchored payload by its ID.
    pub fn get_anchor(&self, id: B256) -> Option<(u64, Vec<u8>)> {
        if let Some(v) = self.anchors.lock().get(&id).cloned() {
            return Some(v);
        }
        match self.backend.get_anchor(id) {
            Ok(Some(v)) => {
                self.anchors.lock().insert(id, v.clone());
                Some(v)
            }
            Ok(None) => None,
            Err(e) => {
                log::warn!("anchor lookup failed: {e}");
                None
            }
        }
    }

    /// Subscribe to new mined blocks (newHeads).
    pub fn subscribe_blocks(&self) -> broadcast::Receiver<BlockRecord> {
        self.block_tx.subscribe()
    }

    // ── Faucet (devnet only) ──────────────────────────────────────────────────

    pub fn fund(&self, addr: Address, wei: U256) {
        let mut db = self.db.lock();
        let mut info = db.basic_ref(addr).unwrap().unwrap_or_default();
        info.balance += wei;
        db.insert_account_info(addr, info);
    }

    // ── Read-only queries ─────────────────────────────────────────────────────

    pub fn balance(&self, addr: Address) -> U256 {
        self.db.lock().basic_ref(addr).unwrap().unwrap_or_default().balance
    }

    pub fn nonce(&self, addr: Address) -> u64 {
        self.db.lock().basic_ref(addr).unwrap().unwrap_or_default().nonce
    }

    pub fn code_at(&self, addr: Address) -> Bytes {
        let db = self.db.lock();
        let info: AccountInfo = db.basic_ref(addr).unwrap().unwrap_or_default();
        info.code.map(|c| c.original_bytes()).unwrap_or_default()
    }

    pub fn receipt(&self, hash: B256) -> Option<EvmReceipt> {
        self.receipts.lock().get(&hash).cloned()
    }

    // ── eth_call (no state change) ────────────────────────────────────────────

    pub fn call(&self, from: Option<Address>, to: Address, data: Bytes, value: U256, gas_limit: u64) -> Result<Bytes, ChainError> {
        let mut db = self.db.lock();
        let block_num = self.block_number.load(Ordering::Relaxed);
        let caller = from.unwrap_or(Address::ZERO);

        // Use a snapshot so eth_call doesn't modify state
        let db_snapshot = db.clone();

        let tx = TxEnv {
            caller,
            transact_to: TransactTo::Call(to),
            data,
            value,
            gas_limit,
            gas_price: U256::from(1_000_000_000u64),
            ..Default::default()
        };

        let result = build_and_run(&mut db, block_num, self.chain_id, tx)?;

        // Restore snapshot (call must not mutate state)
        *db = db_snapshot;

        Ok(result.output)
    }

    // ── Mempool / block production ──────────────────────────────────────────────

    /// Validate + enqueue a raw transaction. Returns tx hash immediately.
    pub fn send_raw_transaction(&self, raw: &[u8]) -> Result<B256, ChainError> {
        let _decoded = decode_raw_tx(raw)?;
        let hash_bytes = crate::types::keccak256(raw);
        let hash = B256::from_slice(&hash_bytes);
        self.mempool.lock().push_back((hash, raw.to_vec()));
        log::debug!("mempool: queued tx {hash}");
        Ok(hash)
    }

    /// Drain the mempool, execute all pending txs, advance block number.
    /// Returns (block_number, state_root_hint) where state_root_hint is a
    /// simple hash of all executed tx hashes (used for L1 anchoring).
    pub fn mine_block(&self) -> (u64, B256) {
        let pending: Vec<(B256, Vec<u8>)> = {
            let mut mp = self.mempool.lock();
            mp.drain(..).collect()
        };

        let block = self.block_number.fetch_add(1, Ordering::SeqCst) + 1;
        let mut included_hashes = Vec::new();

        for (hash, raw) in pending {
            match decode_raw_tx(&raw) {
                Ok(decoded) => match self.execute_decoded(decoded, &raw) {
                    Ok(txhash) => included_hashes.push(txhash),
                    Err(e) => log::warn!("block {block}: tx {hash} failed: {e}"),
                },
                Err(e) => log::warn!("block {block}: tx {hash} decode failed: {e}"),
            }
        }

        // Compute a trivial state root: keccak256 of all included tx hashes
        let mut hasher_input = Vec::with_capacity(included_hashes.len() * 32);
        for h in &included_hashes {
            hasher_input.extend_from_slice(h.as_slice());
        }
        let state_root = B256::from_slice(&crate::types::keccak256(&hasher_input));

        if !included_hashes.is_empty() {
            log::info!("block {block}: {} txs, state_root={state_root}", included_hashes.len());
        }

        *self.latest_state_root.lock() = state_root;

        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let prev_hash =
            if block > 1 { self.blocks.lock().get(&(block - 1)).map(|b| b.hash).unwrap_or(B256::ZERO) } else { B256::ZERO };
        // Block hash = keccak256(block_number || state_root)
        let mut bhash_input = Vec::with_capacity(40);
        bhash_input.extend_from_slice(&block.to_be_bytes());
        bhash_input.extend_from_slice(state_root.as_slice());
        let block_hash = B256::from_slice(&crate::types::keccak256(&bhash_input));
        let gas_used: u64 = included_hashes.len() as u64 * 21_000;

        let new_block = BlockRecord {
            number: block,
            hash: block_hash,
            parent_hash: prev_hash,
            timestamp: now,
            gas_used,
            tx_hashes: included_hashes.clone(),
            state_root,
        };
        self.blocks.lock().insert(block, new_block.clone());
        let _ = self.block_tx.send(new_block); // ok if no subscribers
        {
            let mut t2b = self.tx_to_block.lock();
            for h in &included_hashes {
                t2b.insert(*h, block);
            }
        }

        // Patch block_hash + cumulativeGasUsed into receipts; collect block logs
        {
            let block_hash_hex = format!("0x{}", hex::encode(block_hash));
            let mut all_block_logs: Vec<EvmLog> = Vec::new();
            let mut receipts = self.receipts.lock();
            let mut global_log_idx: usize = 0;
            let mut cumulative: u64 = 0;
            for h in &included_hashes {
                if let Some(receipt) = receipts.get_mut(h) {
                    receipt.block_hash = block_hash_hex.clone();
                    // parse gas_used back to u64 to accumulate cumulative
                    let used = u64::from_str_radix(receipt.gas_used.trim_start_matches("0x"), 16).unwrap_or(0);
                    cumulative += used;
                    receipt.cumulative_gas_used = format!("0x{:x}", cumulative);
                    for log in &mut receipt.logs {
                        log.block_hash = block_hash_hex.clone();
                        log.log_index = format!("0x{:x}", global_log_idx);
                        global_log_idx += 1;
                    }
                    all_block_logs.extend(receipt.logs.clone());
                }
            }
            self.block_logs.lock().insert(block, all_block_logs);
        }

        // Persist state snapshot via selected backend
        let tx_idx = self.tx_index.load(Ordering::Relaxed);
        if let Err(e) = self.backend.persist_snapshot(&self.db.lock(), self.chain_id, block, tx_idx, state_root) {
            log::warn!("state persistence failed ({}): {e}", self.backend.name());
        }

        let included_receipts: Vec<EvmReceipt> = {
            let receipts = self.receipts.lock();
            included_hashes.iter().filter_map(|h| receipts.get(h).cloned()).collect()
        };

        let anchors_snapshot: Vec<(B256, u64, Vec<u8>)> =
            { self.anchors.lock().iter().map(|(id, (blk, payload))| (*id, *blk, payload.clone())).collect() };

        let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);

        let checkpoint = L1CheckpointV1 {
            version: CHECKPOINT_VERSION_V1,
            chain_id: self.chain_id,
            block_number: block,
            timestamp_ms: now_ms,
            state_root,
            receipts_root: receipts_root(&included_receipts),
            tx_root: tx_root(&included_hashes),
            anchor_root: anchor_root(&anchors_snapshot),
            genetics_settlement_root: genetics_settlement_root(&anchors_snapshot),
            reserved: 0,
        };

        if let Err(e) = self.backend.persist_checkpoint(&checkpoint) {
            log::warn!("checkpoint persistence failed ({}): {e}", self.backend.name());
        }
        *self.latest_checkpoint.lock() = Some(checkpoint);

        (block, state_root)
    }

    pub fn pending_count(&self) -> usize {
        self.mempool.lock().len()
    }

    pub fn block_number(&self) -> u64 {
        self.block_number.load(Ordering::Relaxed)
    }

    pub fn get_block(&self, number: u64) -> Option<BlockRecord> {
        self.blocks.lock().get(&number).cloned()
    }

    pub fn get_block_by_hash(&self, hash: B256) -> Option<BlockRecord> {
        self.blocks.lock().values().find(|b| b.hash == hash).cloned()
    }

    pub fn get_tx_block(&self, tx_hash: B256) -> Option<u64> {
        self.tx_to_block.lock().get(&tx_hash).copied()
    }

    /// Return all logs for blocks in `from..=to`, optionally filtered by
    /// contract address and/or topics (first topic = event signature).
    pub fn get_logs_for_blocks(
        &self,
        from_block: u64,
        to_block: u64,
        address_filter: Option<&str>,
        topic0_filter: Option<&str>,
    ) -> Vec<EvmLog> {
        let store = self.block_logs.lock();
        let mut out = Vec::new();
        for blk in from_block..=to_block {
            if let Some(logs) = store.get(&blk) {
                for log in logs {
                    if let Some(addr) = address_filter {
                        if !log.address.eq_ignore_ascii_case(addr) {
                            continue;
                        }
                    }
                    if let Some(t0) = topic0_filter {
                        if log.topics.first().map(|t| t.as_str()) != Some(t0) {
                            continue;
                        }
                    }
                    out.push(log.clone());
                }
            }
        }
        out
    }

    /// Latest EVM state root from the most recent mined block (for L1 anchoring).
    pub fn latest_state_root(&self) -> B256 {
        *self.latest_state_root.lock()
    }

    pub fn latest_checkpoint(&self) -> Option<L1CheckpointV1> {
        self.latest_checkpoint.lock().as_ref().copied()
    }

    pub fn get_checkpoint(&self, block_number: u64) -> Option<L1CheckpointV1> {
        if let Some(cp) = self.latest_checkpoint.lock().as_ref().copied() {
            if cp.block_number == block_number {
                return Some(cp);
            }
        }
        match self.backend.get_checkpoint(block_number) {
            Ok(v) => v,
            Err(e) => {
                log::warn!("checkpoint lookup failed: {e}");
                None
            }
        }
    }

    pub fn storage_at(&self, addr: Address, slot: B256) -> B256 {
        use revm::primitives::U256 as RU256;
        let db = self.db.lock();
        let slot_u256 = RU256::from_be_bytes(*slot.as_ref());
        let val = db.storage_ref(addr, slot_u256).unwrap_or(RU256::ZERO);
        B256::from(val.to_be_bytes::<32>())
    }

    fn execute_decoded(&self, tx: DecodedTx, _raw: &[u8]) -> Result<B256, ChainError> {
        let mut db = self.db.lock();
        let block_num = self.block_number.load(Ordering::Relaxed);

        // Check nonce
        let expected_nonce = db.basic_ref(tx.from).unwrap().unwrap_or_default().nonce;
        if tx.nonce != expected_nonce {
            return Err(ChainError::Nonce { expected: expected_nonce, got: tx.nonce });
        }

        let revm_tx = TxEnv {
            caller: tx.from,
            transact_to: match tx.to {
                Some(addr) => TransactTo::Call(addr),
                None => TransactTo::Create,
            },
            data: tx.data.clone(),
            value: tx.value,
            gas_limit: tx.gas_limit,
            gas_price: tx.gas_price,
            nonce: Some(tx.nonce),
            chain_id: None, // already validated during sig recovery; skip EVM recheck
            ..Default::default()
        };

        let res = build_and_run(&mut db, block_num, self.chain_id, revm_tx)?;

        let tx_idx = self.tx_index.fetch_add(1, Ordering::Relaxed);

        // Convert revm primitives::Log → EvmLog (block_hash filled as ZERO;
        // mine_block will patch it once the real block hash is known)
        let logs: Vec<EvmLog> = res
            .logs
            .iter()
            .enumerate()
            .map(|(i, log)| EvmLog {
                address: format!("0x{}", hex::encode(log.address)),
                topics: log.data.topics().iter().map(|t| format!("0x{}", hex::encode(t))).collect(),
                data: format!("0x{}", hex::encode(&log.data.data)),
                block_number: format!("0x{:x}", block_num),
                transaction_hash: format!("0x{}", hex::encode(tx.hash)),
                transaction_index: format!("0x{:x}", tx_idx),
                block_hash: "0x".to_string() + &"0".repeat(64), // patched in mine_block
                log_index: format!("0x{:x}", i),
                removed: false,
            })
            .collect();

        let gas_used_hex = format!("0x{:x}", res.gas_used);
        let receipt = EvmReceipt {
            transaction_hash: format!("0x{}", hex::encode(tx.hash)),
            block_hash: "0x".to_string() + &"0".repeat(64), // patched in mine_block
            block_number: format!("0x{:x}", block_num),
            from: format!("0x{}", hex::encode(tx.from)),
            to: tx.to.map(|a| format!("0x{}", hex::encode(a))),
            contract_address: res.contract_address.map(|a| format!("0x{}", hex::encode(a))),
            transaction_index: format!("0x{:x}", tx_idx),
            gas_used: gas_used_hex.clone(),
            cumulative_gas_used: gas_used_hex, // patched to running sum in mine_block
            effective_gas_price: format!("0x{:x}", tx.gas_price),
            status: if res.success { "0x1".into() } else { "0x0".into() },
            logs,
            logs_bloom: "0x".to_string() + &"0".repeat(512),
            tx_type: "0x0".into(),
        };

        self.receipts.lock().insert(tx.hash, receipt);
        log::info!("EVM tx {} from {} gas_used={} ok={}", tx.hash, tx.from, res.gas_used, res.success);

        Ok(tx.hash)
    }
}

// ── Execution result ──────────────────────────────────────────────────────────

struct ExecResult {
    output: Bytes,
    gas_used: u64,
    success: bool,
    contract_address: Option<Address>,
    logs: Vec<revm::primitives::Log>,
}

fn build_and_run(db: &mut InMemoryDB, block_number: u64, chain_id: u64, tx: TxEnv) -> Result<ExecResult, ChainError> {
    let _ = chain_id; // chain_id is embedded in TxEnv already
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);

    let result = Evm::builder()
        .with_db(db)
        .modify_block_env(|b| {
            b.number = U256::from(block_number);
            b.timestamp = U256::from(now);
            b.gas_limit = U256::from(30_000_000u64);
            b.basefee = U256::ZERO;
            b.coinbase = Address::ZERO;
        })
        .with_tx_env(tx)
        .build()
        .transact_commit()
        .map_err(|e| ChainError::Evm(format!("{e:?}")))?;

    Ok(match result {
        ExecutionResult::Success { gas_used, output, logs, .. } => {
            let (bytes, contract) = match output {
                Output::Call(b) => (b, None),
                Output::Create(b, addr) => (b, addr),
            };
            ExecResult { output: bytes, gas_used, success: true, contract_address: contract, logs }
        }
        ExecutionResult::Revert { gas_used, output } => {
            ExecResult { output, gas_used, success: false, contract_address: None, logs: vec![] }
        }
        ExecutionResult::Halt { gas_used, .. } => {
            ExecResult { output: Bytes::new(), gas_used, success: false, contract_address: None, logs: vec![] }
        }
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::keccak256;

    fn sign_legacy_tx(
        nonce: u64,
        gas_price: u64,
        gas_limit: u64,
        to: &[u8; 20],
        value_wei: u128,
        data: &[u8],
        chain_id: u64,
        privkey: &[u8; 32],
    ) -> Vec<u8> {
        use rlp::RlpStream;
        use secp256k1::{Message, Secp256k1, SecretKey};

        let strip = |v: &[u8]| -> Vec<u8> {
            let s = v.iter().position(|&b| b != 0).unwrap_or(v.len());
            v[s..].to_vec()
        };

        let gp_bytes = strip(&gas_price.to_be_bytes());
        let val_bytes = strip(&value_wei.to_be_bytes());

        let to_slice: &[u8] = to.as_slice();
        let data_slice: &[u8] = data;

        // Build signing input: RLP([nonce, gas_price, gas_limit, to, value, data, chain_id, 0, 0])
        let mut s = RlpStream::new_list(9);
        s.append(&nonce);
        s.append(&gp_bytes.as_slice()); // &&[u8] → E=&[u8] Sized ✓
        s.append(&gas_limit);
        s.append(&to_slice); // &&[u8] ✓
        s.append(&val_bytes.as_slice()); // &&[u8] ✓
        s.append(&data_slice); // &&[u8] ✓
        s.append(&chain_id);
        s.append(&0u64);
        s.append(&0u64);
        let signing_hash = keccak256(&s.out());

        let secp = Secp256k1::new();
        let sk = SecretKey::from_slice(privkey).unwrap();
        let msg = Message::from_digest_slice(&signing_hash).unwrap();
        let rec_sig = secp.sign_ecdsa_recoverable(&msg, &sk);
        let (rec_id, compact) = rec_sig.serialize_compact();

        let v = chain_id * 2 + 35 + rec_id.to_i32() as u64;
        let r: &[u8] = &compact[..32];
        let sig_s: &[u8] = &compact[32..];

        let mut signed = RlpStream::new_list(9);
        signed.append(&nonce);
        signed.append(&gp_bytes.as_slice());
        signed.append(&gas_limit);
        signed.append(&to_slice);
        signed.append(&val_bytes.as_slice());
        signed.append(&data_slice);
        signed.append(&v);
        signed.append(&r); // &&[u8] ✓
        signed.append(&sig_s); // &&[u8] ✓
        signed.out().to_vec()
    }

    #[test]
    fn test_eth_transfer_eip155() {
        let chain = EvmChain::new(1337);

        // Hardhat account #0: 0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266
        let sender_bytes: [u8; 20] = hex::decode("f39Fd6e51aad88F6F4ce6aB8827279cffFb92266").unwrap().try_into().unwrap();
        let sender = Address::from_slice(&sender_bytes);
        chain.fund(sender, U256::from(10_u128) * U256::from(1_000_000_000_000_000_000u128));

        // Hardhat account #1: 0x70997970C51812dc3A010C7d01b50e0d17dc79C8
        let recipient: [u8; 20] = hex::decode("70997970C51812dc3A010C7d01b50e0d17dc79C8").unwrap().try_into().unwrap();

        let privkey: [u8; 32] =
            hex::decode("ac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80").unwrap().try_into().unwrap();

        let one_eth = 1_000_000_000_000_000_000u128;
        let raw = sign_legacy_tx(0, 1_000_000_000, 21_000, &recipient, one_eth, &[], 1337, &privkey);

        let hash = chain.send_raw_transaction(&raw).expect("tx must enqueue");
        println!("queued txhash: 0x{}", hex::encode(hash));
        assert_eq!(chain.pending_count(), 1, "one tx pending");

        // Mine block — executes all pending txs
        let (block, state_root) = chain.mine_block();
        println!("mined block {block}, state_root={state_root}");
        assert_eq!(chain.pending_count(), 0, "mempool drained");

        let recipient_addr = Address::from_slice(&recipient);
        let balance = chain.balance(recipient_addr);
        assert_eq!(balance, U256::from(one_eth), "recipient should have 1 ETH");
        println!("recipient balance: {balance} wei ✓");

        let receipt = chain.receipt(hash).expect("receipt must exist");
        assert_eq!(receipt.status, "0x1", "tx must succeed");
        println!("receipt status: {} ✓", receipt.status);

        // Verify receipt fields patched by mine_block
        assert_ne!(receipt.block_hash, "0x".to_string() + &"0".repeat(64), "block_hash must be patched from placeholder");
        assert!(receipt.block_hash.starts_with("0x") && receipt.block_hash.len() == 66, "block_hash must be 32-byte hex");

        let cumulative =
            u64::from_str_radix(receipt.cumulative_gas_used.trim_start_matches("0x"), 16).expect("cumulativeGasUsed must parse");
        let used = u64::from_str_radix(receipt.gas_used.trim_start_matches("0x"), 16).expect("gasUsed must parse");
        assert!(cumulative >= used, "cumulativeGasUsed >= gasUsed");
        assert_eq!(cumulative, 21_000, "simple transfer uses exactly 21000 gas");

        assert!(receipt.effective_gas_price.starts_with("0x"), "effectiveGasPrice must be hex");
        let price =
            u64::from_str_radix(receipt.effective_gas_price.trim_start_matches("0x"), 16).expect("effectiveGasPrice must parse");
        assert_eq!(price, 1_000_000_000, "effectiveGasPrice = gas_price = 1 gwei");

        let checkpoint = chain.latest_checkpoint().expect("checkpoint must exist after mine");
        assert_eq!(checkpoint.chain_id, 1337);
        assert_eq!(checkpoint.block_number, block);
        assert_eq!(checkpoint.state_root, state_root);
        assert_ne!(checkpoint.tx_root, B256::ZERO, "tx root should include the transfer tx");

        println!("blockHash:         {} ✓", &receipt.block_hash[..10]);
        println!("cumulativeGasUsed: {} ✓", receipt.cumulative_gas_used);
        println!("effectiveGasPrice: {} ✓", receipt.effective_gas_price);
    }
}
