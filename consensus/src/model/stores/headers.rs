use std::sync::Arc;

use kaspa_consensus_core::{header::Header, BlockHasher, BlockLevel};
use kaspa_consensus_core::BlueWorkType;
use kaspa_database::prelude::{BatchDbWriter, CachedDbAccess, DbKey};
use kaspa_database::prelude::{CachePolicy, DB};
use kaspa_database::prelude::{StoreError, StoreResult};
use kaspa_database::registry::DatabaseStorePrefixes;
use kaspa_hashes::Hash;
use kaspa_muhash::Hash as Blake2Hash;
use kaspa_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

/// Header layout as stored by pre-epoch_seed binaries (V0).
/// Field order must exactly match the old `Header` struct for bincode to work.
#[derive(Deserialize)]
struct HeaderV0 {
    hash:                    Hash,
    version:                 u16,
    parents_by_level:        Vec<Vec<Hash>>,
    hash_merkle_root:        Hash,
    accepted_id_merkle_root: Hash,
    utxo_commitment:         Blake2Hash,
    timestamp:               u64,
    bits:                    u32,
    nonce:                   u64,
    daa_score:               u64,
    blue_work:               BlueWorkType,
    blue_score:              u64,
    pruning_point:           Hash,
}

#[derive(Deserialize)]
struct HeaderWithBlockLevelV0 {
    header:      HeaderV0,
    block_level: BlockLevel,
}

pub trait HeaderStoreReader {
    fn get_daa_score(&self, hash: Hash) -> Result<u64, StoreError>;
    fn get_blue_score(&self, hash: Hash) -> Result<u64, StoreError>;
    fn get_timestamp(&self, hash: Hash) -> Result<u64, StoreError>;
    fn get_bits(&self, hash: Hash) -> Result<u32, StoreError>;
    fn get_header(&self, hash: Hash) -> Result<Arc<Header>, StoreError>;
    fn get_header_with_block_level(&self, hash: Hash) -> Result<HeaderWithBlockLevel, StoreError>;
    fn get_compact_header_data(&self, hash: Hash) -> Result<CompactHeaderData, StoreError>;
}

#[derive(Clone, Serialize, Deserialize)]
pub struct HeaderWithBlockLevel {
    pub header: Arc<Header>,
    pub block_level: BlockLevel,
}

impl MemSizeEstimator for HeaderWithBlockLevel {
    fn estimate_mem_bytes(&self) -> usize {
        self.header.as_ref().estimate_mem_bytes() + size_of::<Self>()
    }
}

pub trait HeaderStore: HeaderStoreReader {
    // This is append only
    fn insert(&self, hash: Hash, header: Arc<Header>, block_level: BlockLevel) -> Result<(), StoreError>;
    fn delete(&self, hash: Hash) -> Result<(), StoreError>;
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct CompactHeaderData {
    pub daa_score: u64,
    pub timestamp: u64,
    pub bits: u32,
    pub blue_score: u64,
}

impl MemSizeEstimator for CompactHeaderData {}

impl From<&Header> for CompactHeaderData {
    fn from(header: &Header) -> Self {
        Self { daa_score: header.daa_score, timestamp: header.timestamp, bits: header.bits, blue_score: header.blue_score }
    }
}

/// A DB + cache implementation of `HeaderStore` trait, with concurrency support.
#[derive(Clone)]
pub struct DbHeadersStore {
    db: Arc<DB>,
    compact_headers_access: CachedDbAccess<Hash, CompactHeaderData, BlockHasher>,
    headers_access: CachedDbAccess<Hash, HeaderWithBlockLevel, BlockHasher>,
}

impl DbHeadersStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy, compact_cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            compact_headers_access: CachedDbAccess::new(
                Arc::clone(&db),
                compact_cache_policy,
                DatabaseStorePrefixes::HeadersCompact.into(),
            ),
            headers_access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::Headers.into()),
        }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy, compact_cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy, compact_cache_policy)
    }

    pub fn has(&self, hash: Hash) -> StoreResult<bool> {
        self.headers_access.has(hash)
    }

    pub fn insert_batch(
        &self,
        batch: &mut WriteBatch,
        hash: Hash,
        header: Arc<Header>,
        block_level: BlockLevel,
    ) -> Result<(), StoreError> {
        if self.headers_access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.headers_access.write(BatchDbWriter::new(batch), hash, HeaderWithBlockLevel { header: header.clone(), block_level })?;
        self.compact_headers_access.write(BatchDbWriter::new(batch), hash, header.as_ref().into())?;
        Ok(())
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> Result<(), StoreError> {
        self.compact_headers_access.delete(BatchDbWriter::new(batch), hash)?;
        self.headers_access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl HeaderStoreReader for DbHeadersStore {
    fn get_daa_score(&self, hash: Hash) -> Result<u64, StoreError> {
        if let Some(header_with_block_level) = self.headers_access.read_from_cache(hash) {
            return Ok(header_with_block_level.header.daa_score);
        }
        Ok(self.compact_headers_access.read(hash)?.daa_score)
    }

    fn get_blue_score(&self, hash: Hash) -> Result<u64, StoreError> {
        if let Some(header_with_block_level) = self.headers_access.read_from_cache(hash) {
            return Ok(header_with_block_level.header.blue_score);
        }
        Ok(self.compact_headers_access.read(hash)?.blue_score)
    }

    fn get_timestamp(&self, hash: Hash) -> Result<u64, StoreError> {
        if let Some(header_with_block_level) = self.headers_access.read_from_cache(hash) {
            return Ok(header_with_block_level.header.timestamp);
        }
        Ok(self.compact_headers_access.read(hash)?.timestamp)
    }

    fn get_bits(&self, hash: Hash) -> Result<u32, StoreError> {
        if let Some(header_with_block_level) = self.headers_access.read_from_cache(hash) {
            return Ok(header_with_block_level.header.bits);
        }
        Ok(self.compact_headers_access.read(hash)?.bits)
    }

    fn get_header(&self, hash: Hash) -> Result<Arc<Header>, StoreError> {
        match self.headers_access.read(hash) {
            Ok(hwl) => Ok(hwl.header),
            Err(StoreError::DeserializationError(_)) => self.get_header_v0_fallback(hash),
            Err(e) => Err(e),
        }
    }

    fn get_header_with_block_level(&self, hash: Hash) -> Result<HeaderWithBlockLevel, StoreError> {
        self.headers_access.read(hash)
    }

    fn get_compact_header_data(&self, hash: Hash) -> Result<CompactHeaderData, StoreError> {
        if let Some(header_with_block_level) = self.headers_access.read_from_cache(hash) {
            return Ok(header_with_block_level.header.as_ref().into());
        }
        self.compact_headers_access.read(hash)
    }
}

impl DbHeadersStore {
    /// Fallback for headers stored by the pre-epoch_seed binary (V0 format).
    /// Reads raw bytes, deserializes as HeaderWithBlockLevelV0, upgrades to V1
    /// (epoch_seed = Hash::default(), stored hash preserved), then lazily rewrites
    /// the DB entry so future reads use the fast V1 path.
    fn get_header_v0_fallback(&self, hash: Hash) -> Result<Arc<Header>, StoreError> {
        let db_key = DbKey::new(DatabaseStorePrefixes::Headers.as_ref(), hash);
        let Some(slice) = self.db.get_pinned(&db_key)? else {
            return Err(StoreError::KeyNotFound(db_key));
        };
        let old: HeaderWithBlockLevelV0 = bincode::deserialize(&slice)?;
        let v0 = old.header;
        // Build V1 Header: epoch_seed defaults to zeros; preserve the stored hash
        // (do NOT call finalize() — the hash was computed by the old binary without
        // epoch_seed and is still correct for pre-fork blocks).
        let header = Arc::new(Header {
            hash:                    v0.hash,
            version:                 v0.version,
            parents_by_level:        v0.parents_by_level,
            hash_merkle_root:        v0.hash_merkle_root,
            accepted_id_merkle_root: v0.accepted_id_merkle_root,
            utxo_commitment:         v0.utxo_commitment,
            timestamp:               v0.timestamp,
            bits:                    v0.bits,
            nonce:                   v0.nonce,
            daa_score:               v0.daa_score,
            blue_work:               v0.blue_work,
            blue_score:              v0.blue_score,
            epoch_seed:              Hash::default(),
            pruning_point:           v0.pruning_point,
        });
        // Lazy migration: rewrite in V1 format so this code path is hit only once.
        let mut batch = WriteBatch::default();
        self.headers_access.write(
            BatchDbWriter::new(&mut batch),
            hash,
            HeaderWithBlockLevel { header: header.clone(), block_level: old.block_level },
        )?;
        let _ = self.db.write(batch); // best-effort; non-fatal if it fails
        Ok(header)
    }
}

impl HeaderStore for DbHeadersStore {
    fn insert(&self, hash: Hash, header: Arc<Header>, block_level: u8) -> Result<(), StoreError> {
        if self.headers_access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        if self.compact_headers_access.has(hash)? {
            return Err(StoreError::DataInconsistency(format!("store has compact data for {} but is missing full data", hash)));
        }
        let mut batch = WriteBatch::default();
        self.compact_headers_access.write(BatchDbWriter::new(&mut batch), hash, header.as_ref().into())?;
        self.headers_access.write(BatchDbWriter::new(&mut batch), hash, HeaderWithBlockLevel { header, block_level })?;
        self.db.write(batch)?;
        Ok(())
    }

    fn delete(&self, hash: Hash) -> Result<(), StoreError> {
        let mut batch = WriteBatch::default();
        self.compact_headers_access.delete(BatchDbWriter::new(&mut batch), hash)?;
        self.headers_access.delete(BatchDbWriter::new(&mut batch), hash)?;
        self.db.write(batch)?;
        Ok(())
    }
}
