use std::sync::Arc;

use borsh_miner::{BorshDeserialize, BorshSerialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use super::archive::{packed_bytes_for, GenomeArchive};

/// A slice of the genome selected for MLM training.
/// `chunk_idx` is the fragment index inside the `.xenom` archive.
#[derive(Debug, Clone, PartialEq, BorshSerialize, BorshDeserialize)]
pub struct GenomeSlice {
    pub chunk_idx: u64,
    pub start_base: u32,
    pub length: u32,
}

/// A training batch composed of genome slices ready to be served to a miner.
#[derive(Debug, Clone, PartialEq, BorshSerialize, BorshDeserialize)]
pub struct GenomeTrainingBatch {
    pub batch_id: u64,
    pub model_id: String,
    pub genome_merkle_root: [u8; 32],
    pub data_indices: Vec<GenomeSlice>,
    pub mask_ratio: f32,
    pub seq_length: usize,
}

/// Deterministic generator for `GenomeTrainingBatch` from a `.xenom` genome archive.
pub struct GenomeBatchGenerator {
    archive: Arc<GenomeArchive>,
    rng: StdRng,
}

impl GenomeBatchGenerator {
    /// Create a new generator seeded with a 32-byte block hash.
    pub fn new(archive: Arc<GenomeArchive>, seed: [u8; 32]) -> Self {
        let rng = StdRng::from_seed(seed);
        Self { archive, rng }
    }

    /// Generate a batch of `batch_size` genome slices, each up to `seq_len` bases long.
    ///
    /// The returned batch also includes a reproducible `batch_id` derived from the archive
    /// merkle root and the generator's RNG state.
    pub fn generate_batch(&mut self, batch_size: usize, seq_len: usize) -> GenomeTrainingBatch {
        let mut slices = Vec::with_capacity(batch_size);
        let fragment_count = self.archive.num_fragments();
        let max_seq_len = seq_len.min(self.archive.fragment_size as usize);

        if fragment_count == 0 || max_seq_len == 0 {
            return GenomeTrainingBatch {
                batch_id: 0,
                model_id: String::new(),
                genome_merkle_root: self.archive.header.merkle_root,
                data_indices: slices,
                mask_ratio: 0.15,
                seq_length: seq_len,
            };
        }

        for _ in 0..batch_size {
            let fragment_idx = self.rng.gen_range(0..fragment_count);
            let fragment_bases = self.archive.fragment_base_count(fragment_idx).unwrap_or(0) as usize;
            let length = max_seq_len.min(fragment_bases);
            if length == 0 {
                continue;
            }
            let start_base = if fragment_bases == length {
                0
            } else {
                self.rng.gen_range(0..=(fragment_bases - length))
            };

            slices.push(GenomeSlice {
                chunk_idx: fragment_idx,
                start_base: start_base as u32,
                length: length as u32,
            });
        }

        let batch_id = self.rng.gen::<u64>();

        GenomeTrainingBatch {
            batch_id,
            model_id: String::new(),
            genome_merkle_root: self.archive.header.merkle_root,
            data_indices: slices,
            mask_ratio: 0.15,
            seq_length: seq_len,
        }
    }

    /// Extract the actual DNA sequence for a single genome slice.
    pub fn extract_for_miner(&self, slice: &GenomeSlice) -> anyhow::Result<String> {
        self.archive.extract_sequence(slice.chunk_idx, slice.start_base, slice.length)
    }

    /// Convenience helper to extract all sequences for a batch.
    pub fn extract_sequences(&self, batch: &GenomeTrainingBatch) -> Vec<String> {
        batch
            .data_indices
            .iter()
            .filter_map(|slice| self.extract_for_miner(slice).ok())
            .collect()
    }
}

/// Encode a base character to its 2-bit XENOGEN1 representation.
/// A/a → 0, C/c → 1, G/g → 2, T/t → 3, N/other → 0 (A, for determinism).
fn encode_base(c: char) -> u8 {
    match c {
        'A' | 'a' => 0b00,
        'C' | 'c' => 0b01,
        'G' | 'g' => 0b10,
        'T' | 't' => 0b11,
        _ => 0b00,
    }
}

/// Encode a DNA sequence string into 2-bit packed bytes using the XENOGEN1 scheme.
pub fn pack_sequence(seq: &str) -> Vec<u8> {
    let bases: Vec<u8> = seq.chars().map(encode_base).collect();
    let mut packed = vec![0u8; packed_bytes_for(bases.len() as u32) as usize];
    for (i, bits) in bases.iter().enumerate() {
        let byte_idx = i / 4;
        let shift = 6 - 2 * (i % 4);
        packed[byte_idx] |= (*bits & 0b11) << shift;
    }
    packed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::base_from_bits;
    use super::super::archive::{GenomeArchive, GENOME_FILE_HEADER_SIZE, GENOME_FILE_MAGIC};

    fn tiny_archive(seq: &str, fragment_size: u32) -> GenomeArchive {
        let packed = pack_sequence(seq);
        let total_bases = seq.len() as u64;
        let total_packed = packed.len() as u64;

        // Compute merkle root for a single fragment.
        let leaf = {
            let mut h = blake3::Hasher::new();
            h.update(&0u64.to_le_bytes());
            let unpacked: Vec<u8> = seq.chars().map(|c| c as u8).collect();
            h.update(&unpacked);
            *h.finalize().as_bytes()
        };

        let mut header_bytes = Vec::with_capacity(GENOME_FILE_HEADER_SIZE);
        header_bytes.extend_from_slice(GENOME_FILE_MAGIC);
        header_bytes.extend_from_slice(&1u32.to_le_bytes());
        header_bytes.extend_from_slice(&0u32.to_le_bytes());
        header_bytes.extend_from_slice(&total_bases.to_le_bytes());
        header_bytes.extend_from_slice(&total_packed.to_le_bytes());
        header_bytes.extend_from_slice(&leaf);

        let mut bytes = header_bytes;
        bytes.extend_from_slice(&packed);

        GenomeArchive::from_bytes(&bytes, fragment_size).unwrap()
    }

    #[test]
    fn test_pack_sequence_mapping() {
        // A=00, C=01, G=10, T=11, MSB-first
        assert_eq!(pack_sequence("ACGT"), vec![0b00011011]);
        assert_eq!(pack_sequence("AAAA"), vec![0b00000000]);
        assert_eq!(pack_sequence("TTTT"), vec![0b11111111]);
        assert_eq!(base_from_bits(0b00), 'A');
        assert_eq!(base_from_bits(0b01), 'C');
        assert_eq!(base_from_bits(0b10), 'G');
        assert_eq!(base_from_bits(0b11), 'T');
    }

    #[test]
    fn test_deterministic_batch_generation() {
        let archive = tiny_archive("ACGTACGTACGTACGTACGTACGTACGTACGT", 32);
        let seed = [7u8; 32];

        let mut gen1 = GenomeBatchGenerator::new(Arc::new(archive.clone()), seed);
        let mut gen2 = GenomeBatchGenerator::new(Arc::new(archive), seed);

        let batch1 = gen1.generate_batch(4, 8);
        let batch2 = gen2.generate_batch(4, 8);

        assert_eq!(batch1, batch2);
        assert_eq!(batch1.data_indices.len(), 4);
    }

    #[test]
    fn test_extract_for_miner() {
        let archive = tiny_archive("ACGTACGTACGTACGT", 16);
        let mut gen = GenomeBatchGenerator::new(Arc::new(archive), [1u8; 32]);

        let batch = gen.generate_batch(1, 4);
        let slice = &batch.data_indices[0];
        let seq = gen.extract_for_miner(slice).unwrap();

        assert_eq!(seq.len(), slice.length as usize);
        assert!(seq.chars().all(|c| "ACGT".contains(c)));
    }
}
