use std::{path::Path, sync::Arc, time::{SystemTime, UNIX_EPOCH}};

use revm::{db::InMemoryDB, primitives::B256};
use rocksdb::{ColumnFamilyDescriptor, DB, Options};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::checkpoint::L1CheckpointV1;
use crate::persist::{decode_snapshot_bin, encode_snapshot_bin, LoadedSnapshot};

pub const CF_META: &str = "meta";
pub const CF_ACCOUNTS: &str = "accounts";
pub const CF_ACCOUNT_CODE: &str = "account_code";
pub const CF_ACCOUNT_STORAGE: &str = "account_storage";
pub const CF_BLOCKS: &str = "blocks";
pub const CF_BLOCK_BY_NUMBER: &str = "block_by_number";
pub const CF_TRANSACTIONS: &str = "transactions";
pub const CF_RECEIPTS: &str = "receipts";
pub const CF_LOGS: &str = "logs";
pub const CF_MEMPOOL: &str = "mempool";
pub const CF_ANCHORS: &str = "anchors";
pub const CF_GENETICS_JOBS: &str = "genetics_jobs";
pub const CF_GENETICS_RESULTS: &str = "genetics_results";
pub const CF_GENETICS_SETTLEMENTS: &str = "genetics_settlements";
pub const CF_CHECKPOINTS: &str = "checkpoints";
pub const CF_L1_ANCHORS: &str = "l1_anchors";
pub const CF_INDEXES: &str = "indexes";

pub const ALL_CFS: &[&str] = &[
    CF_META,
    CF_ACCOUNTS,
    CF_ACCOUNT_CODE,
    CF_ACCOUNT_STORAGE,
    CF_BLOCKS,
    CF_BLOCK_BY_NUMBER,
    CF_TRANSACTIONS,
    CF_RECEIPTS,
    CF_LOGS,
    CF_MEMPOOL,
    CF_ANCHORS,
    CF_GENETICS_JOBS,
    CF_GENETICS_RESULTS,
    CF_GENETICS_SETTLEMENTS,
    CF_CHECKPOINTS,
    CF_L1_ANCHORS,
    CF_INDEXES,
];

const KEY_STATE_SNAPSHOT_V1: &[u8] = b"state_snapshot_v1";
const KEY_CHAIN_ID: &[u8] = b"chain_id";
const KEY_HEAD_BLOCK_NUMBER: &[u8] = b"head_block_number";
const KEY_LATEST_STATE_ROOT: &[u8] = b"latest_state_root";

#[derive(Debug, Error)]
pub enum BackendError {
    #[error("rocksdb open failed: {0}")]
    Open(String),
    #[error("rocksdb operation failed: {0}")]
    Db(String),
    #[error("missing column family: {0}")]
    MissingCf(&'static str),
    #[error("snapshot encode/decode failed: {0}")]
    Snapshot(String),
}

pub trait StateBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn load_snapshot(&self) -> Result<Option<LoadedSnapshot>, BackendError>;
    fn persist_snapshot(
        &self,
        db: &InMemoryDB,
        chain_id: u64,
        block_number: u64,
        tx_index: u64,
        state_root: B256,
    ) -> Result<(), BackendError>;
    fn put_anchor(&self, anchor_id: B256, block_number: u64, payload: &[u8]) -> Result<(), BackendError>;
    fn get_anchor(&self, anchor_id: B256) -> Result<Option<(u64, Vec<u8>)>, BackendError>;
    fn persist_checkpoint(&self, checkpoint: &L1CheckpointV1) -> Result<(), BackendError>;
    fn get_checkpoint(&self, block_number: u64) -> Result<Option<L1CheckpointV1>, BackendError>;
}

#[derive(Default)]
pub struct InMemoryBackend;

impl InMemoryBackend {
    pub fn new() -> Self {
        Self
    }
}

impl StateBackend for InMemoryBackend {
    fn name(&self) -> &'static str {
        "inmemory"
    }

    fn load_snapshot(&self) -> Result<Option<LoadedSnapshot>, BackendError> {
        Ok(None)
    }

    fn persist_snapshot(
        &self,
        _db: &InMemoryDB,
        _chain_id: u64,
        _block_number: u64,
        _tx_index: u64,
        _state_root: B256,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    fn put_anchor(&self, _anchor_id: B256, _block_number: u64, _payload: &[u8]) -> Result<(), BackendError> {
        Ok(())
    }

    fn get_anchor(&self, _anchor_id: B256) -> Result<Option<(u64, Vec<u8>)>, BackendError> {
        Ok(None)
    }

    fn persist_checkpoint(&self, _checkpoint: &L1CheckpointV1) -> Result<(), BackendError> {
        Ok(())
    }

    fn get_checkpoint(&self, _block_number: u64) -> Result<Option<L1CheckpointV1>, BackendError> {
        Ok(None)
    }
}

pub struct RocksDbBackend {
    db: Arc<DB>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnchorRecord {
    block_number: u64,
    created_at_ms: u64,
    payload: Vec<u8>,
}

impl RocksDbBackend {
    pub fn open(state_dir: &Path, chain_id: u64) -> Result<Self, BackendError> {
        std::fs::create_dir_all(state_dir).map_err(|e| BackendError::Open(e.to_string()))?;
        let db_path = state_dir.join(format!("evm-state-{chain_id}.rocksdb"));

        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        let cf_descriptors: Vec<ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, Options::default()))
            .collect();

        let db = DB::open_cf_descriptors(&opts, db_path, cf_descriptors)
            .map_err(|e| BackendError::Open(e.to_string()))?;

        Ok(Self { db: Arc::new(db) })
    }

    fn cf(&self, name: &'static str) -> Result<&rocksdb::ColumnFamily, BackendError> {
        self.db.cf_handle(name).ok_or(BackendError::MissingCf(name))
    }

    fn anchor_key(anchor_id: B256) -> Vec<u8> {
        format!("anchor:{}", hex::encode(anchor_id)).into_bytes()
    }

    fn checkpoint_key(block_number: u64) -> [u8; 8] {
        block_number.to_be_bytes()
    }

    fn l1_anchor_key(checkpoint_id: B256) -> Vec<u8> {
        format!("checkpoint:{}", hex::encode(checkpoint_id)).into_bytes()
    }
}

impl StateBackend for RocksDbBackend {
    fn name(&self) -> &'static str {
        "rocksdb"
    }

    fn load_snapshot(&self) -> Result<Option<LoadedSnapshot>, BackendError> {
        let meta = self.cf(CF_META)?;
        let raw = self
            .db
            .get_cf(meta, KEY_STATE_SNAPSHOT_V1)
            .map_err(|e| BackendError::Db(e.to_string()))?;

        match raw {
            Some(bytes) => decode_snapshot_bin(&bytes)
                .map(Some)
                .map_err(BackendError::Snapshot),
            None => Ok(None),
        }
    }

    fn persist_snapshot(
        &self,
        db: &InMemoryDB,
        chain_id: u64,
        block_number: u64,
        tx_index: u64,
        state_root: B256,
    ) -> Result<(), BackendError> {
        let meta = self.cf(CF_META)?;
        let snapshot = encode_snapshot_bin(db, chain_id, block_number, tx_index, state_root)
            .map_err(BackendError::Snapshot)?;

        self.db
            .put_cf(meta, KEY_STATE_SNAPSHOT_V1, snapshot)
            .map_err(|e| BackendError::Db(e.to_string()))?;
        self.db
            .put_cf(meta, KEY_CHAIN_ID, chain_id.to_be_bytes())
            .map_err(|e| BackendError::Db(e.to_string()))?;
        self.db
            .put_cf(meta, KEY_HEAD_BLOCK_NUMBER, block_number.to_be_bytes())
            .map_err(|e| BackendError::Db(e.to_string()))?;
        self.db
            .put_cf(meta, KEY_LATEST_STATE_ROOT, state_root.as_slice())
            .map_err(|e| BackendError::Db(e.to_string()))?;

        Ok(())
    }

    fn put_anchor(&self, anchor_id: B256, block_number: u64, payload: &[u8]) -> Result<(), BackendError> {
        let anchors = self.cf(CF_ANCHORS)?;
        let rec = AnchorRecord {
            block_number,
            created_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            payload: payload.to_vec(),
        };
        let encoded = bincode::serialize(&rec).map_err(|e| BackendError::Db(e.to_string()))?;

        self.db
            .put_cf(anchors, Self::anchor_key(anchor_id), encoded)
            .map_err(|e| BackendError::Db(e.to_string()))
    }

    fn get_anchor(&self, anchor_id: B256) -> Result<Option<(u64, Vec<u8>)>, BackendError> {
        let anchors = self.cf(CF_ANCHORS)?;
        let raw = self
            .db
            .get_cf(anchors, Self::anchor_key(anchor_id))
            .map_err(|e| BackendError::Db(e.to_string()))?;

        match raw {
            Some(bytes) => {
                let rec: AnchorRecord = bincode::deserialize(&bytes)
                    .map_err(|e| BackendError::Db(e.to_string()))?;
                Ok(Some((rec.block_number, rec.payload)))
            }
            None => Ok(None),
        }
    }

    fn persist_checkpoint(&self, checkpoint: &L1CheckpointV1) -> Result<(), BackendError> {
        let checkpoints = self.cf(CF_CHECKPOINTS)?;
        let l1_anchors = self.cf(CF_L1_ANCHORS)?;

        let bytes = checkpoint.to_bytes();
        let checkpoint_id = checkpoint.checkpoint_id();

        self.db
            .put_cf(checkpoints, Self::checkpoint_key(checkpoint.block_number), bytes)
            .map_err(|e| BackendError::Db(e.to_string()))?;

        self.db
            .put_cf(l1_anchors, Self::l1_anchor_key(checkpoint_id), bytes)
            .map_err(|e| BackendError::Db(e.to_string()))?;

        Ok(())
    }

    fn get_checkpoint(&self, block_number: u64) -> Result<Option<L1CheckpointV1>, BackendError> {
        let checkpoints = self.cf(CF_CHECKPOINTS)?;
        let raw = self
            .db
            .get_cf(checkpoints, Self::checkpoint_key(block_number))
            .map_err(|e| BackendError::Db(e.to_string()))?;

        match raw {
            Some(bytes) => Ok(L1CheckpointV1::from_bytes(&bytes)),
            None => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    use revm::{
        primitives::{AccountInfo, Address, B256, U256},
        DatabaseRef,
    };

    use crate::checkpoint::{L1CheckpointV1, CHECKPOINT_VERSION_V1};

    fn temp_rocks_dir(tag: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        p.push(format!("xenom-evm-{tag}-{ts}"));
        p
    }

    #[test]
    fn phase1_snapshot_persist_and_load_roundtrip() {
        let dir = temp_rocks_dir("phase1");
        let backend = RocksDbBackend::open(&dir, 1337).expect("open rocksdb");

        let mut db = InMemoryDB::default();
        let addr = Address::from_slice(&[0x11u8; 20]);
        db.insert_account_info(
            addr,
            AccountInfo {
                balance: U256::from(123_456u64),
                nonce: 7,
                ..Default::default()
            },
        );

        let root = B256::from([0xabu8; 32]);
        backend
            .persist_snapshot(&db, 1337, 9, 17, root)
            .expect("persist snapshot");

        let loaded = backend
            .load_snapshot()
            .expect("load snapshot")
            .expect("snapshot exists");

        assert_eq!(loaded.block_number, 9);
        assert_eq!(loaded.tx_index, 17);
        assert_eq!(loaded.state_root, root);

        let acc = loaded
            .db
            .basic_ref(addr)
            .expect("db query")
            .expect("account exists");
        assert_eq!(acc.balance, U256::from(123_456u64));
        assert_eq!(acc.nonce, 7);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn phase2_checkpoint_persisted_in_checkpoints_and_l1_anchors() {
        let dir = temp_rocks_dir("phase2");
        let backend = RocksDbBackend::open(&dir, 1337).expect("open rocksdb");

        let cp = L1CheckpointV1 {
            version: CHECKPOINT_VERSION_V1,
            chain_id: 1337,
            block_number: 21,
            timestamp_ms: 1_750_000_000_000,
            state_root: B256::from([1u8; 32]),
            receipts_root: B256::from([2u8; 32]),
            tx_root: B256::from([3u8; 32]),
            anchor_root: B256::from([4u8; 32]),
            genetics_settlement_root: B256::from([5u8; 32]),
            reserved: 0,
        };

        backend.persist_checkpoint(&cp).expect("persist checkpoint");

        let loaded = backend
            .get_checkpoint(21)
            .expect("get checkpoint")
            .expect("checkpoint exists");
        assert_eq!(loaded, cp);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
