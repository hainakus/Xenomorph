use borsh_miner::{BorshDeserialize, BorshSerialize};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use super::archive::{packed_bytes_for, GenomeArchive};

/// A slice of the genome selected for MLM training.
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

/// Deterministic generator for `GenomeTrainingBatch` from a packed genome archive.
pub struct GenomeBatchGenerator {
    archive: GenomeArchive,
    rng: StdRng,
}

impl GenomeBatchGenerator {
    /// Create a new generator seeded with a 32-byte block hash.
    pub fn new(archive: GenomeArchive, seed: [u8; 32]) -> Self {
        let rng = StdRng::from_seed(seed);
        Self { archive, rng }
    }

    /// Generate a batch of `batch_size` genome slices, each up to `seq_len` bases long.
    ///
    /// The returned batch also includes a reproducible `batch_id` derived from the archive
    /// merkle root and the generator's RNG state.
    pub fn generate_batch(&mut self, batch_size: usize, seq_len: usize) -> GenomeTrainingBatch {
        let mut slices = Vec::with_capacity(batch_size);
        let chunk_count = self.archive.index.len() as u64;

        for _ in 0..batch_size {
            if chunk_count == 0 {
                break;
            }
            let chunk_idx = self.rng.gen_range(0..chunk_count);
            let chunk = &self.archive.index[chunk_idx as usize];

            let chunk_bases = chunk.length as usize;
            let length = seq_len.min(chunk_bases);
            if length == 0 {
                continue;
            }
            let start_base = if chunk_bases == length {
                0
            } else {
                self.rng.gen_range(0..=(chunk_bases - length))
            };

            slices.push(GenomeSlice {
                chunk_idx,
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
        self.archive.extract_sequence(slice.chunk_idx as usize, slice.start_base, slice.length)
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

/// Encode a DNA sequence string into 2-bit packed bytes using the Xenom packing scheme.
pub fn pack_sequence(seq: &str) -> Vec<u8> {
    let bases: Vec<u8> = seq
        .chars()
        .map(|c| match c {
            'A' | 'a' => 0b00,
            'T' | 't' => 0b01,
            'C' | 'c' => 0b10,
            'G' | 'g' => 0b11,
            _ => 0b00,
        })
        .collect();

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
    use super::super::archive::{ChunkIndex, GenomeArchive, XenomHeader};

    fn tiny_archive(seq: &str) -> GenomeArchive {
        let packed = pack_sequence(seq);
        let leaf = *blake3::hash(&packed).as_bytes();
        let header = XenomHeader {
            magic: *b"XENOM\0",
            version: 1,
            merkle_root: leaf,
            chunks: 1,
        };
        let index = vec![ChunkIndex {
            offset: 0,
            length: seq.len() as u32,
            chromosome: 1,
            position: 0,
        }];
        GenomeArchive {
            header,
            index,
            data: packed,
        }
    }

    #[test]
    fn test_deterministic_batch_generation() {
        let archive = tiny_archive("ATCGATCGATCGATCGATCGATCGATCG");
        let seed = [7u8; 32];

        let mut gen1 = GenomeBatchGenerator::new(archive.clone(), seed);
        let mut gen2 = GenomeBatchGenerator::new(archive, seed);

        let batch1 = gen1.generate_batch(4, 8);
        let batch2 = gen2.generate_batch(4, 8);

        assert_eq!(batch1, batch2);
        assert_eq!(batch1.data_indices.len(), 4);
    }

    #[test]
    fn test_extract_for_miner() {
        let archive = tiny_archive("ATCGATCGATCG");
        let mut gen = GenomeBatchGenerator::new(archive, [1u8; 32]);

        let batch = gen.generate_batch(1, 4);
        let slice = &batch.data_indices[0];
        let seq = gen.extract_for_miner(slice).unwrap();

        assert_eq!(seq.len(), slice.length as usize);
        assert!(seq.chars().all(|c| "ATCG".contains(c)));
    }
}
