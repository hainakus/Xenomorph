use super::HasherExtensions;
use crate::header::Header;
use kaspa_hashes::{Hash, HasherBase};

/// DAA score at which Genome PoW activates on mainnet.
pub const EPOCH_SEED_HASH_ACTIVATION_MAINNET: u64 = 21_370_801;

/// Returns the header hash using the provided nonce+timestamp instead of those in the header.
///
/// `epoch_seed` is included in the hash **only when it is non-zero**.  A zero (default) epoch_seed
/// means either the block predates Genome PoW activation or was received from a legacy node that
/// did not set the field; in both cases we produce the same hash as those legacy nodes.
/// Once Genome PoW is active the node sets a real (non-zero) epoch_seed in new block templates,
/// which then becomes part of the canonical hash — forming a clean hard-fork boundary.
#[inline]
pub fn hash_override_nonce_time(header: &Header, nonce: u64, timestamp: u64) -> Hash {
    hash_override_nonce_time_with_activation(header, nonce, timestamp, 0)
}

/// Like [`hash_override_nonce_time`] but with an explicit genome-PoW activation DAA score.
/// `epoch_seed` is included in the hash only when BOTH conditions hold:
///   1. `header.daa_score >= genome_pow_activation_daa_score`
///   2. `header.epoch_seed != Hash::default()` (non-zero sentinel)
/// Legacy nodes never set `epochSeed`; the P2P layer defaults it to `Hash::default()`,
/// so condition 2 ensures we produce the same hash as legacy nodes for their blocks.
#[inline]
pub fn hash_override_nonce_time_with_activation(
    header: &Header,
    nonce: u64,
    timestamp: u64,
    genome_pow_activation_daa_score: u64,
) -> Hash {
    let mut hasher = kaspa_hashes::BlockHash::new();
    hasher.update(header.version.to_le_bytes()).write_len(header.parents_by_level.len());

    header.parents_by_level.iter().for_each(|level| {
        hasher.write_var_array(level);
    });

    hasher
        .update(header.hash_merkle_root)
        .update(header.accepted_id_merkle_root)
        .update(header.utxo_commitment)
        .update(timestamp.to_le_bytes())
        .update(header.bits.to_le_bytes())
        .update(nonce.to_le_bytes())
        .update(header.daa_score.to_le_bytes())
        .update(header.blue_score.to_le_bytes())
        .write_blue_work(header.blue_work);

    if header.daa_score >= genome_pow_activation_daa_score && header.epoch_seed != Hash::default() {
        hasher.update(header.epoch_seed);
    }

    hasher.update(header.pruning_point);
    hasher.finalize()
}

/// Returns the header hash.
pub fn hash(header: &Header) -> Hash {
    hash_override_nonce_time(header, header.nonce, header.timestamp)
}

#[cfg(test)]
mod tests {
    use core::str::FromStr;
    use std::hash::Hash;
    use super::*;
    use crate::{blockhash, BlueWorkType};
    use crate::block::Block;
    use crate::config::params::GENESIS;

    #[test]
    fn test_header_hashing() {
        let header = Header::from_precomputed_hash("f7812a38f08473ff60e5b314a069e034919ad8b4b1b6812902d2d3f541c6e174".parse().unwrap(), vec![]);
        println!("HEADER {:?}", header);
       let h = Block::from_header(header.clone());

        assert_eq!(h.hash(), header.hash);
    }

    #[test]
    fn test_hash_blue_work() {
        let tests: Vec<(BlueWorkType, Vec<u8>)> =
            vec![(0.into(), vec![0, 0, 0, 0, 0, 0, 0, 0]), (123456.into(), vec![3, 0, 0, 0, 0, 0, 0, 0, 1, 226, 64])];

        for test in tests {
            let mut hasher = kaspa_hashes::BlockHash::new();
            hasher.write_blue_work(test.0);

            let mut hasher2 = kaspa_hashes::BlockHash::new();
            hasher2.update(test.1);
            assert_eq!(hasher.finalize(), hasher2.finalize())
        }
    }
}
