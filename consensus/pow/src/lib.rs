// public for benchmarks
#[doc(hidden)]
pub mod genome_file;
#[doc(hidden)]
pub mod genome_pow;
#[doc(hidden)]
pub mod matrix;
#[cfg(feature = "wasm32-sdk")]
pub mod wasm;
#[doc(hidden)]
pub mod xoshiro;

use std::cmp::max;

use crate::matrix::Matrix;
use genome_pow::GenomePowState;
use kaspa_consensus_core::{hashing, header::Header, BlockLevel};
use kaspa_hashes::PowHash;
use kaspa_math::Uint256;

/// State is an intermediate data structure with pre-computed values to speed up mining.
pub struct State {
    pub(crate) matrix: Matrix,
    pub(crate) target: Uint256,
    // PRE_POW_HASH || TIME || 32 zero byte padding; without NONCE
    pub(crate) hasher: PowHash,
}

impl State {
    /// Build a `State` directly from stratum-provided raw parts.
    /// Used by stratum miners that receive pre_pow_hash + timestamp + bits separately.
    #[inline]
    pub fn from_parts(pre_pow_hash: kaspa_hashes::Hash, timestamp: u64, bits: u32) -> Self {
        let target = Uint256::from_compact_target_bits(bits);
        let hasher = PowHash::new(pre_pow_hash, timestamp);
        let matrix = Matrix::generate(pre_pow_hash);
        Self { matrix, target, hasher }
    }

    #[inline]
    pub fn new(header: &Header) -> Self {
        let target = Uint256::from_compact_target_bits(header.bits);
        // KHeavyHash NEVER includes epoch_seed — epoch_seed is exclusive to Genome PoW
        // and is handled explicitly by genome_pow_state / GenomePowState.
        // Including it here (activation=0) caused block-level mis-computation during IBD
        // for blocks whose headers carry a non-zero epoch_seed set by the virtual processor
        // but that were mined without it in the pre-pow hash.
        let pre_pow_hash = hashing::header::hash_override_nonce_time_with_activation(header, 0, 0, u64::MAX);
        // PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
        let hasher = PowHash::new(pre_pow_hash, header.timestamp);
        let matrix = Matrix::generate(pre_pow_hash);
        Self { matrix, target, hasher }
    }

    #[inline]
    #[must_use]
    /// PRE_POW_HASH || TIME || 32 zero byte padding || NONCE
    pub fn calculate_pow(&self, nonce: u64) -> Uint256 {
        // Hasher already contains PRE_POW_HASH || TIME || 32 zero byte padding; so only the NONCE is missing
        let hash = self.hasher.clone().finalize_with_nonce(nonce);
        let hash = self.matrix.heavy_hash(hash);
        Uint256::from_le_bytes(hash.as_bytes())
    }

    #[inline]
    #[must_use]
    pub fn check_pow(&self, nonce: u64) -> (bool, Uint256) {
        let pow = self.calculate_pow(nonce);
        // The pow hash must be less or equal than the claimed target.
        (pow <= self.target, pow)
    }
}

/// Builds a `GenomePowState` from a block header (used when genome PoW is active).
pub fn genome_pow_state(header: &Header, fragment_size_bytes: u32) -> GenomePowState {
    let target = Uint256::from_compact_target_bits(header.bits);
    // Genome PoW miners include epoch_seed in the pre-pow hash (activation=0 means
    // include when non-zero).  epoch_seed is ALSO fed into GenomePowState explicitly
    // as the fragment-selection seed — both usages are intentional for security.
    let pre_pow_hash = hashing::header::hash_override_nonce_time(header, 0, 0);
    GenomePowState::new(pre_pow_hash, target, header.epoch_seed, fragment_size_bytes)
}

pub fn calc_block_level(header: &Header, max_block_level: BlockLevel) -> BlockLevel {
    if header.parents_by_level.is_empty() {
        return max_block_level; // Genesis has the max block level
    }

    let state = State::new(header);
    let (_, pow) = state.check_pow(header.nonce);
    let signed_block_level = max_block_level as i64 - pow.bits() as i64;
    max(signed_block_level, 0) as BlockLevel
}

/// Calculates block level using genome PoW when a fragment is supplied.
///
/// Used by validators once the genome dataset is available.
/// Falls back to legacy KHeavyHash when `fragment` is `None`.
pub fn calc_block_level_genome(
    header: &Header,
    max_block_level: BlockLevel,
    fragment: Option<&[u8]>,
    fragment_size_bytes: u32,
) -> BlockLevel {
    if header.parents_by_level.is_empty() {
        return max_block_level;
    }
    let pow = match fragment {
        Some(frag) => {
            let state = genome_pow_state(header, fragment_size_bytes);
            let (_, pow, _) = state.check_pow_with_fragment(header.nonce, frag);
            pow
        }
        None => {
            let state = State::new(header);
            let (_, pow) = state.check_pow(header.nonce);
            pow
        }
    };
    let signed_block_level = max_block_level as i64 - pow.bits() as i64;
    max(signed_block_level, 0) as BlockLevel
}
