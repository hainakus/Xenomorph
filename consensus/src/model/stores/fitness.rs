use std::sync::Arc;

use kaspa_consensus_core::BlockHasher;
use kaspa_database::prelude::{BatchDbWriter, CachedDbAccess, CachePolicy, DirectDbWriter, StoreError, StoreResult, DB};
use kaspa_database::registry::DatabaseStorePrefixes;
use kaspa_hashes::Hash;
use kaspa_utils::mem_size::MemSizeEstimator;
use rocksdb::WriteBatch;
use serde::{Deserialize, Serialize};

/// Stored fitness value for a single block's coinbase.
#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct BlockFitnessData {
    /// Genome PoW fitness score as reported by the miner in the coinbase v2 payload.
    pub fitness: u32,
    /// DAA score of the block, stored alongside fitness to allow epoch-window queries.
    pub daa_score: u64,
}

impl MemSizeEstimator for BlockFitnessData {}

pub trait FitnessStoreReader {
    fn get_fitness(&self, hash: Hash) -> Result<BlockFitnessData, StoreError>;
}

pub trait FitnessStore: FitnessStoreReader {
    fn insert(&self, hash: Hash, data: BlockFitnessData) -> StoreResult<()>;
    fn delete(&self, hash: Hash) -> StoreResult<()>;
}

/// DB + cache implementation of `FitnessStore`.
#[derive(Clone)]
pub struct DbFitnessStore {
    db: Arc<DB>,
    access: CachedDbAccess<Hash, BlockFitnessData, BlockHasher>,
}

impl DbFitnessStore {
    pub fn new(db: Arc<DB>, cache_policy: CachePolicy) -> Self {
        Self {
            db: Arc::clone(&db),
            access: CachedDbAccess::new(db, cache_policy, DatabaseStorePrefixes::BlockFitness.into()),
        }
    }

    pub fn clone_with_new_cache(&self, cache_policy: CachePolicy) -> Self {
        Self::new(Arc::clone(&self.db), cache_policy)
    }

    pub fn insert_batch(&self, batch: &mut WriteBatch, hash: Hash, data: BlockFitnessData) -> StoreResult<()> {
        if self.access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.access.write(BatchDbWriter::new(batch), hash, data)?;
        Ok(())
    }

    pub fn delete_batch(&self, batch: &mut WriteBatch, hash: Hash) -> StoreResult<()> {
        self.access.delete(BatchDbWriter::new(batch), hash)
    }
}

impl FitnessStoreReader for DbFitnessStore {
    fn get_fitness(&self, hash: Hash) -> Result<BlockFitnessData, StoreError> {
        self.access.read(hash)
    }
}

impl FitnessStore for DbFitnessStore {
    fn insert(&self, hash: Hash, data: BlockFitnessData) -> StoreResult<()> {
        if self.access.has(hash)? {
            return Err(StoreError::HashAlreadyExists(hash));
        }
        self.access.write(DirectDbWriter::new(&self.db), hash, data)?;
        Ok(())
    }

    fn delete(&self, hash: Hash) -> StoreResult<()> {
        self.access.delete(DirectDbWriter::new(&self.db), hash)
    }
}
