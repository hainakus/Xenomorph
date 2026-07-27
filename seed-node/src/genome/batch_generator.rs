use std::sync::Arc;

use borsh::{BorshDeserialize, BorshSerialize};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

/// Maximum number of times the generator will resample a slice before giving up
/// and accepting a low-complexity region.  Keeps batches from stalling if the
/// archive is dominated by N/poly-X stretches.
const MAX_HOMOPOLYMER_RETRIES: u32 = 20;

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

    /// Return true if the extracted sequence for `slice` consists of a single
    /// repeated base (e.g. AAAAA... or TTTTT...).  These regions provide no
    /// information for MLM training and can bias the model toward a single base.
    fn is_homopolymer(&self, slice: &GenomeSlice) -> bool {
        if slice.length < 2 {
            return false;
        }
        let Ok(seq) = self.extract_for_miner(slice) else { return false };
        let first = seq.as_bytes().first().copied().unwrap_or(b'A');
        seq.as_bytes().iter().all(|&b| b == first)
    }

    /// Pick a fragment and contiguous start position that is not a homopolymer.
    fn pick_non_homopolymer_contiguous(&mut self, batch_size: usize, max_seq_len: usize) -> Option<(u64, usize)> {
        let fragment_count = self.archive.num_fragments();
        let total_span = if batch_size == 1 { max_seq_len } else { max_seq_len + (batch_size - 1) * max_seq_len };

        for _ in 0..MAX_HOMOPOLYMER_RETRIES {
            let fragment_idx = self.rng.gen_range(0..fragment_count);
            let fragment_bases = self.archive.fragment_base_count(fragment_idx).unwrap_or(0) as usize;
            if fragment_bases < total_span {
                continue;
            }
            let max_start = fragment_bases - total_span;
            let start_base = if max_start == 0 { 0 } else { self.rng.gen_range(0..=max_start) };
            let probe = GenomeSlice { chunk_idx: fragment_idx, start_base: start_base as u32, length: max_seq_len as u32 };
            if !self.is_homopolymer(&probe) {
                return Some((fragment_idx, start_base));
            }
        }
        None
    }

    /// Pick a single non-homopolymer slice, falling back to the last draw if
    /// the archive is dominated by low-complexity regions.
    fn pick_non_homopolymer_slice(&mut self, max_seq_len: usize) -> Option<GenomeSlice> {
        let fragment_count = self.archive.num_fragments();
        let mut last_slice = None;

        for _ in 0..MAX_HOMOPOLYMER_RETRIES {
            let fragment_idx = self.rng.gen_range(0..fragment_count);
            let fragment_bases = self.archive.fragment_base_count(fragment_idx).unwrap_or(0) as usize;
            let length = max_seq_len.min(fragment_bases);
            if length == 0 {
                continue;
            }
            let start_base = if fragment_bases == length { 0 } else { self.rng.gen_range(0..=(fragment_bases - length)) };
            let slice = GenomeSlice { chunk_idx: fragment_idx, start_base: start_base as u32, length: length as u32 };
            last_slice = Some(slice.clone());
            if !self.is_homopolymer(&slice) {
                return Some(slice);
            }
        }

        last_slice
    }

    /// Generate a batch of `batch_size` genome slices, each up to `seq_len` bases long.
    ///
    /// When the genome fragment is large enough, the slices are adjacent windows taken
    /// from a single fragment, so a training batch covers a contiguous genomic region
    /// instead of scattered random positions.  This gives the DNABERT-2 attention layers
    /// meaningful local context.  If the fragment is too small, the generator falls back
    /// to the original random-window behaviour.
    ///
    /// Regions that are all one base (e.g. poly-A tails or runs of unknown bases packed
    /// as A) are skipped, because they provide no training signal and can collapse the
    /// MLM head to a single class.
    ///
    /// The returned batch also includes a reproducible `batch_id` derived from the archive
    /// merkle root and the generator's RNG state.
    pub fn generate_batch(&mut self, batch_size: usize, seq_len: usize) -> GenomeTrainingBatch {
        let mut slices = Vec::with_capacity(batch_size);
        let fragment_count = self.archive.num_fragments();
        let max_seq_len = seq_len.min(self.archive.fragment_size as usize);

        if fragment_count == 0 || max_seq_len == 0 || batch_size == 0 {
            return GenomeTrainingBatch {
                batch_id: 0,
                model_id: String::new(),
                genome_merkle_root: self.archive.header.merkle_root,
                data_indices: slices,
                mask_ratio: 0.15,
                seq_length: seq_len,
            };
        }

        // Try to produce adjacent windows from a single fragment.  The step size is
        // the window length so consecutive slices are contiguous.
        if let Some((fragment_idx, start_base)) = self.pick_non_homopolymer_contiguous(batch_size, max_seq_len) {
            let fragment_bases = self.archive.fragment_base_count(fragment_idx).unwrap_or(0) as usize;
            let step = max_seq_len;
            for i in 0..batch_size {
                let s = start_base + i * step;
                let length = max_seq_len.min(fragment_bases - s);
                if length == 0 {
                    continue;
                }
                slices.push(GenomeSlice { chunk_idx: fragment_idx, start_base: s as u32, length: length as u32 });
            }
        } else {
            // Fragment too small for contiguous windows, or all contiguous probes were
            // homopolymers: fall back to random slices.
            for _ in 0..batch_size {
                if let Some(slice) = self.pick_non_homopolymer_slice(max_seq_len) {
                    slices.push(slice);
                }
            }
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
        batch.data_indices.iter().filter_map(|slice| self.extract_for_miner(slice).ok()).collect()
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
    use super::super::archive::{GenomeArchive, GENOME_FILE_HEADER_SIZE, GENOME_FILE_MAGIC};
    use super::*;
    use crate::genome::base_from_bits;

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
