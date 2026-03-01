use super::*;
use crate::constants;
use crate::errors::{BlockProcessResult, RuleError};
use crate::model::services::reachability::ReachabilityService;
use crate::model::stores::statuses::StatusesStoreReader;
use kaspa_consensus_core::blockhash::BlockHashExtensions;
use kaspa_consensus_core::blockstatus::BlockStatus::StatusInvalid;
use kaspa_consensus_core::header::Header;
use kaspa_consensus_core::BlockLevel;
use kaspa_core::time::unix_now;
use kaspa_database::prelude::StoreResultExtensions;
use std::cmp::max;
use blake3;

impl HeaderProcessor {
    /// Validates the header in isolation including pow check against header declared bits.
    /// Returns the block level as computed from pow state or a rule error if such was encountered
    pub(super) fn validate_header_in_isolation(&self, header: &Header) -> BlockProcessResult<BlockLevel> {
        self.check_header_version(header)?;
        self.check_block_timestamp_in_isolation(header)?;
        self.check_parents_limit(header)?;
        Self::check_parents_not_origin(header)?;
        self.check_pow_and_calc_block_level(header)
    }

    pub(super) fn validate_parent_relations(&self, header: &Header) -> BlockProcessResult<()> {
        self.check_parents_exist(header)?;
        self.check_parents_incest(header)?;
        Ok(())
    }

    fn check_header_version(&self, header: &Header) -> BlockProcessResult<()> {
        if header.version != constants::BLOCK_VERSION {
            return Err(RuleError::WrongBlockVersion(header.version));
        }
        Ok(())
    }

    fn check_block_timestamp_in_isolation(&self, header: &Header) -> BlockProcessResult<()> {
        // Timestamp deviation tolerance is in seconds so we multiply by 1000 to get milliseconds (without BPS dependency)
        let max_block_time = unix_now() + self.timestamp_deviation_tolerance * 1000;
        if header.timestamp > max_block_time {
            return Err(RuleError::TimeTooFarIntoTheFuture(header.timestamp, max_block_time));
        }
        Ok(())
    }

    fn check_parents_limit(&self, header: &Header) -> BlockProcessResult<()> {
        if header.direct_parents().is_empty() {
            return Err(RuleError::NoParents);
        }

        if header.direct_parents().len() > self.max_block_parents as usize {
            return Err(RuleError::TooManyParents(header.direct_parents().len(), self.max_block_parents as usize));
        }

        Ok(())
    }

    fn check_parents_not_origin(header: &Header) -> BlockProcessResult<()> {
        if header.direct_parents().iter().any(|&parent| parent.is_origin()) {
            return Err(RuleError::OriginParent);
        }

        Ok(())
    }

    fn check_parents_exist(&self, header: &Header) -> BlockProcessResult<()> {
        let mut missing_parents = Vec::new();
        for parent in header.direct_parents() {
            match self.statuses_store.read().get(*parent).unwrap_option() {
                None => missing_parents.push(*parent),
                Some(StatusInvalid) => {
                    return Err(RuleError::InvalidParent(*parent));
                }
                Some(_) => {}
            }
        }
        if !missing_parents.is_empty() {
            return Err(RuleError::MissingParents(missing_parents));
        }
        Ok(())
    }

    fn check_parents_incest(&self, header: &Header) -> BlockProcessResult<()> {
        let parents = header.direct_parents();
        for parent_a in parents.iter() {
            for parent_b in parents.iter() {
                if parent_a == parent_b {
                    continue;
                }

                if self.reachability_service.is_dag_ancestor_of(*parent_a, *parent_b) {
                    return Err(RuleError::InvalidParentsRelation(*parent_a, *parent_b));
                }
            }
        }

        Ok(())
    }

    fn check_pow_and_calc_block_level(&self, header: &Header) -> BlockProcessResult<BlockLevel> {
        if self.skip_proof_of_work {
            return Ok(0);
        }
        let pow = if header.daa_score >= self.genome_pow_activation_daa_score {
            self.check_genome_pow(header)?
        } else {
            let state = kaspa_pow::State::new(header);
            let (passed, pow) = state.check_pow(header.nonce);
            if !passed {
                return Err(RuleError::InvalidPoW);
            }
            pow
        };
        let signed_block_level = self.max_block_level as i64 - pow.bits() as i64;
        Ok(max(signed_block_level, 0) as BlockLevel)
    }

    /// Validates genome PoW for `header`.
    ///
    /// Derives the genome fragment deterministically from `(epoch_seed, nonce, fragment_index)`.
    /// When the real on-disk GRCh38 dataset loader is integrated, replace the
    /// `synthesize_fragment` call with an actual disk read keyed on `fragment_idx`.
    fn check_genome_pow(&self, header: &Header) -> BlockProcessResult<kaspa_math::Uint256> {
        use kaspa_pow::genome_pow::{fragment_index, GenomePowState};
        let state = GenomePowState::new(
            kaspa_consensus_core::hashing::header::hash_override_nonce_time(header, 0, 0),
            kaspa_math::Uint256::from_compact_target_bits(header.bits),
            header.epoch_seed,
            self.genome_fragment_size_bytes,
        );
        let fragment_idx = fragment_index(&header.epoch_seed, header.nonce, self.genome_fragment_size_bytes);
        let fragment = match &self.genome_dataset_loader {
            Some(loader) => match loader.load_fragment(fragment_idx) {
                Some(f) => f,
                None => return Err(RuleError::InvalidPoW), // loader present but fragment unavailable
            },
            None => self.synthesize_fragment(fragment_idx, &header.epoch_seed),
        };
        let (passed, pow, _fitness) = state.check_pow_with_fragment(header.nonce, &fragment);
        if passed {
            Ok(pow)
        } else {
            Err(RuleError::InvalidPoW)
        }
    }

    /// Produces a deterministic pseudo-fragment of `genome_fragment_size_bytes` bytes.
    ///
    /// This is a placeholder until the production genome dataset (GRCh38) is loaded
    /// from disk.  Every 32-byte chunk is `blake3(fragment_idx_le ‖ epoch_seed ‖ chunk_idx_le)`.
    fn synthesize_fragment(&self, fragment_idx: u64, epoch_seed: &kaspa_hashes::Hash) -> Vec<u8> {
        let size = self.genome_fragment_size_bytes as usize;
        let mut out = Vec::with_capacity(size);
        let mut chunk = 0u64;
        while out.len() < size {
            let mut h = blake3::Hasher::new();
            h.update(&fragment_idx.to_le_bytes());
            h.update(epoch_seed.as_ref());
            h.update(&chunk.to_le_bytes());
            out.extend_from_slice(h.finalize().as_bytes());
            chunk += 1;
        }
        out.truncate(size);
        out
    }
}
