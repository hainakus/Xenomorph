use std::collections::HashSet;

use anyhow::Result;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::rpc::messages::TrainingBatch;
use crate::tokenizer::DnaTokenizer;

/// A single masked-language-model batch ready to be converted into `candle` tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct MlmBatch {
    pub input_ids: Vec<u32>,
    pub token_type_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub labels: Vec<u32>,
    pub mask: Vec<u8>,
    pub seq_len: usize,
    pub batch_size: usize,
    /// Indices that map each batch row back to its source training example.
    /// When reverse-complement augmentation is enabled this vector is doubled
    /// (forward + RC) while keeping the same source index for both rows.
    pub batch_indices: Vec<u64>,
}

/// Generates synthetic DNA training batches with random MLM masking.
#[derive(Clone)]
pub struct MlmBatchGenerator {
    tokenizer: DnaTokenizer,
    seq_len: usize,
    mask_prob: f64,
    dna_bases: Vec<char>,
    /// Mask contiguous spans of this many tokens (a "k-mer" span) instead of
    /// individual tokens.  Default is `min(seq_len, 6)` to approximate 6-mer
    /// masking for DNABERT-2 while gracefully handling short test sequences.
    span_len: usize,
    /// When enabled, each input sequence is augmented with its reverse
    /// complement, exposing Watson-Crick base-pair relationships to the model.
    reverse_complement: bool,
}

impl MlmBatchGenerator {
    pub const DEFAULT_SEQ_LEN: usize = 128;
    pub const DEFAULT_MASK_PROB: f64 = 0.15;
    /// Default span length used for k-mer style span masking.
    pub const DEFAULT_SPAN_LEN: usize = 6;

    /// Create a new generator with the given tokenizer and sequence length.
    pub fn new(tokenizer: DnaTokenizer, seq_len: usize) -> Self {
        Self {
            tokenizer,
            seq_len,
            mask_prob: Self::DEFAULT_MASK_PROB,
            dna_bases: vec!['A', 'T', 'C', 'G'],
            span_len: seq_len.clamp(1, Self::DEFAULT_SPAN_LEN),
            reverse_complement: false,
        }
    }

    /// Override the default 15% MLM mask probability.
    pub fn with_mask_prob(mut self, mask_prob: f64) -> Self {
        self.mask_prob = mask_prob.clamp(0.0, 1.0);
        self
    }

    /// Set the contiguous span length used for k-mer style masking.
    pub fn with_span_len(mut self, span_len: usize) -> Self {
        self.span_len = span_len.max(1);
        self
    }

    /// Enable or disable reverse-complement augmentation.
    pub fn with_reverse_complement(mut self, enabled: bool) -> Self {
        self.reverse_complement = enabled;
        self
    }

    /// Generate an MLM batch from a list of DNA sequences.
    ///
    /// Each sequence is tokenized, truncated to the model's maximum `seq_len`, and
    /// masked using an RNG seeded from `seed` and `batch_id`.  The returned batch's
    /// `seq_len` is the actual maximum encoded length in the batch, so short sequences
    /// are not padded up to the model's full token budget.
    pub fn generate_from_sequences(&self, sequences: &[String], seed: &[u8; 32], batch_id: u64) -> Result<MlmBatch> {
        self.generate_from_sequences_with_indices(sequences, seed, batch_id, None)
    }

    /// Generate an MLM batch from DNA sequences while preserving a source index
    /// for each row.  This is the version used by genome batches so the miner can
    /// report which genome slices contributed to the gradient.
    pub fn generate_from_sequences_with_indices(
        &self,
        sequences: &[String],
        seed: &[u8; 32],
        batch_id: u64,
        batch_indices: Option<&[u64]>,
    ) -> Result<MlmBatch> {
        if sequences.is_empty() {
            return Ok(MlmBatch {
                input_ids: Vec::new(),
                token_type_ids: Vec::new(),
                attention_mask: Vec::new(),
                labels: Vec::new(),
                mask: Vec::new(),
                seq_len: 0,
                batch_size: 0,
                batch_indices: Vec::new(),
            });
        }

        let base_seed = u64::from_le_bytes(seed[..8].try_into().unwrap_or([0u8; 8])) ^ batch_id;

        // Optionally augment every sequence with its reverse complement.  This lets
        // the model learn that A pairs with T and C pairs with G by seeing both
        // strands of the same genomic window.
        let factor = if self.reverse_complement { 2 } else { 1 };
        let mut all_sequences = Vec::with_capacity(sequences.len() * factor);
        let mut all_indices = Vec::with_capacity(sequences.len() * factor);
        for (b, seq) in sequences.iter().enumerate() {
            let idx = batch_indices.map(|v| v[b]).unwrap_or(b as u64);
            all_sequences.push(seq.clone());
            all_indices.push(idx);
            if self.reverse_complement {
                all_sequences.push(reverse_complement(seq));
                all_indices.push(idx);
            }
        }

        let mut encoded_seqs = Vec::with_capacity(all_sequences.len());
        let mut per_seq_seeds = Vec::with_capacity(all_sequences.len());
        let mut output_indices = Vec::with_capacity(all_sequences.len());
        for (b, (seq, &idx)) in all_sequences.iter().zip(all_indices.iter()).enumerate() {
            let mut encoded = self.tokenizer.encode(seq, false)?;
            encoded.truncate(self.seq_len);
            encoded_seqs.push(encoded);

            let seed = base_seed.wrapping_add(idx);
            // Use a slightly different seed for the reverse-complement copy so the
            // model does not see the exact same mask pattern on both strands.
            let seed = if self.reverse_complement && b % 2 == 1 { seed.wrapping_add(1) } else { seed };
            per_seq_seeds.push(seed);
            output_indices.push(idx);
        }

        self.prepare_mlm_batch(encoded_seqs, &per_seq_seeds, &output_indices)
    }

    /// Generate an MLM batch from a non-genome `TrainingBatch`.
    ///
    /// This is a synthetic fallback used for devnet/dry-run when no real genome
    /// archive is configured.  It builds random DNA strings from `data_indices`
    /// and reuses the same masking/padding path as `generate_from_sequences`.
    pub fn generate(&self, batch: &TrainingBatch) -> Result<MlmBatch> {
        if batch.data_indices.is_empty() {
            return Ok(MlmBatch {
                input_ids: Vec::new(),
                token_type_ids: Vec::new(),
                attention_mask: Vec::new(),
                labels: Vec::new(),
                mask: Vec::new(),
                seq_len: 0,
                batch_size: 0,
                batch_indices: Vec::new(),
            });
        }

        let base_seed = u64::from_le_bytes(batch.base_checkpoint[..8].try_into().unwrap_or([0u8; 8]));
        let raw_len = (self.seq_len * 4).max(16);

        let factor = if self.reverse_complement { 2 } else { 1 };
        let mut sequences = Vec::with_capacity(batch.data_indices.len() * factor);
        let mut per_seq_seeds = Vec::with_capacity(batch.data_indices.len() * factor);
        let mut batch_indices = Vec::with_capacity(batch.data_indices.len() * factor);
        for &index in &batch.data_indices {
            let mut rng = Self::seeded_rng(&batch.base_checkpoint, index);

            // Generate a random DNA string long enough to tokenize into at least seq_len ids.
            let sequence: String = (0..raw_len).map(|_| self.dna_bases[rng.gen_range(0..4)]).collect();

            sequences.push(sequence.clone());
            per_seq_seeds.push(base_seed ^ index);
            batch_indices.push(index);

            if self.reverse_complement {
                sequences.push(reverse_complement(&sequence));
                per_seq_seeds.push((base_seed ^ index).wrapping_add(1));
                batch_indices.push(index);
            }
        }

        let mut encoded_seqs = Vec::with_capacity(sequences.len());
        for sequence in &sequences {
            let mut encoded = self.tokenizer.encode(sequence, false)?;
            encoded.truncate(self.seq_len);
            encoded_seqs.push(encoded);
        }

        self.prepare_mlm_batch(encoded_seqs, &per_seq_seeds, &batch_indices)
    }

    /// Build an MLM batch from already-encoded sequences, per-sequence RNG seeds,
    /// and source indices.
    ///
    /// The output `seq_len` is the actual maximum encoded length in the batch, so
    /// short sequences are not padded up to the model's full token budget.  Masking
    /// is performed on contiguous spans of length `span_len` whenever the sequence
    /// is long enough, which mimics masking whole k-mers instead of single bases.
    fn prepare_mlm_batch(&self, encoded_seqs: Vec<Vec<u32>>, per_seq_seeds: &[u64], batch_indices: &[u64]) -> Result<MlmBatch> {
        let batch_size = encoded_seqs.len();
        let max_encoded_len = encoded_seqs.iter().map(|v| v.len()).max().unwrap_or(0).min(self.seq_len).max(1);
        let total_len = batch_size * max_encoded_len;

        let mut input_ids = vec![self.tokenizer.pad_token_id; total_len];
        let token_type_ids = vec![0u32; total_len];
        let mut attention_mask = vec![0u32; total_len];
        let mut labels = vec![u32::MAX; total_len];
        let mut mask = vec![0u8; total_len];

        for (b, encoded) in encoded_seqs.iter().enumerate() {
            let masked = self.mask_positions(encoded, per_seq_seeds[b]);
            let offset = b * max_encoded_len;
            for (i, &original_id) in encoded.iter().enumerate() {
                let pos = offset + i;
                attention_mask[pos] = 1;
                input_ids[pos] = original_id;

                if masked.contains(&i) {
                    mask[pos] = 1;
                    labels[pos] = original_id;
                    input_ids[pos] = self.tokenizer.mask_token_id;
                }
            }
        }

        Ok(MlmBatch {
            input_ids,
            token_type_ids,
            attention_mask,
            labels,
            mask,
            seq_len: max_encoded_len,
            batch_size,
            batch_indices: batch_indices.to_vec(),
        })
    }

    /// Select a set of token positions to mask using span-based k-mer masking.
    ///
    /// Instead of flipping a coin per token, this masks contiguous spans of length
    /// `span_len` (clamped to the sequence length and the remaining target), which
    /// forces the model to predict whole k-mer groups and preserves local context.
    fn mask_positions(&self, encoded: &[u32], seed: u64) -> HashSet<usize> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let len = encoded.len();
        let target = ((len as f64) * self.mask_prob).ceil().max(1.0) as usize;
        let target = target.min(len);

        let mut masked = HashSet::with_capacity(target);
        let mut attempts = 0;
        let max_attempts = (len.saturating_mul(20)).max(100);

        while masked.len() < target && attempts < max_attempts {
            let remaining = target - masked.len();
            let span = self.span_len.min(len).min(remaining).max(1);
            if len - masked.len() < span {
                break;
            }
            let max_start = len - span + 1;
            let start = rng.gen_range(0..max_start);

            if (start..start + span).all(|p| !masked.contains(&p)) {
                for p in start..start + span {
                    masked.insert(p);
                }
            }
            attempts += 1;
        }

        // Fallback: fill any still-unmasked positions one by one.
        let mut available: Vec<usize> = (0..len).filter(|p| !masked.contains(p)).collect();
        while masked.len() < target && !available.is_empty() {
            let idx = rng.gen_range(0..available.len());
            masked.insert(available.swap_remove(idx));
        }

        masked
    }

    fn seeded_rng(base_checkpoint: &[u8; 32], index: u64) -> ChaCha8Rng {
        let seed_bytes: [u8; 8] = base_checkpoint[..8].try_into().expect("base_checkpoint has 32 bytes");
        let seed = u64::from_le_bytes(seed_bytes) ^ index;
        ChaCha8Rng::seed_from_u64(seed)
    }
}

/// Return the reverse complement of a DNA string.
///
/// Non-ACGT characters are passed through unchanged and only uppercase mappings
/// are handled; the genome archive already returns uppercase sequences.
fn reverse_complement(seq: &str) -> String {
    seq.chars().rev().map(complement_base).collect()
}

fn complement_base(c: char) -> char {
    match c.to_ascii_uppercase() {
        'A' => 'T',
        'T' => 'A',
        'C' => 'G',
        'G' => 'C',
        _ => c,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::tokenizer::AddedToken;
    use tokenizers::Tokenizer;

    fn build_test_tokenizer() -> DnaTokenizer {
        let mut vocab: Vocab = Vocab::new();
        vocab.insert("A".to_string(), 0);
        vocab.insert("T".to_string(), 1);
        vocab.insert("C".to_string(), 2);
        vocab.insert("G".to_string(), 3);
        vocab.insert("<mask>".to_string(), 4);
        vocab.insert("<pad>".to_string(), 5);

        let bpe = BPE::new(vocab, vec![]);
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);

        let bytes = serde_json::to_vec(&tokenizer).unwrap();
        DnaTokenizer::from_bytes(&bytes).unwrap()
    }

    #[test]
    fn test_batch_shape_and_masking() {
        let tokenizer = build_test_tokenizer();
        let generator = MlmBatchGenerator::new(tokenizer, 16);

        let batch = TrainingBatch {
            batch_id: 1,
            model_id: "dnabert2".to_string(),
            base_checkpoint: [42u8; 32],
            data_indices: vec![0, 1, 2],
            target_improvement: 0.01,
            learning_rate: 0.01,
        };

        let mlm = generator.generate(&batch).unwrap();
        assert_eq!(mlm.batch_size, 3);
        assert_eq!(mlm.seq_len, 16);
        assert_eq!(mlm.input_ids.len(), 48);
        assert_eq!(mlm.labels.len(), 48);
        assert_eq!(mlm.mask.len(), 48);

        // All non-pad positions should have attention_mask == 1.
        for i in 0..48 {
            if mlm.input_ids[i] != 5 {
                assert_eq!(mlm.attention_mask[i], 1);
            }
        }

        // Some tokens should be masked (statistically very likely with 15% over 48 positions).
        let masked_count = mlm.mask.iter().filter(|&&m| m == 1).count();
        assert!(masked_count > 0, "expected at least one masked token");

        // Labels should only be set for masked positions.
        for i in 0..48 {
            if mlm.mask[i] == 1 {
                assert_ne!(mlm.labels[i], u32::MAX, "label should be original token id at masked position");
                assert_eq!(mlm.input_ids[i], 4, "masked input should be <mask> token id");
            } else {
                assert_eq!(mlm.labels[i], u32::MAX);
            }
        }
    }

    #[test]
    fn test_generate_from_sequences() {
        let tokenizer = build_test_tokenizer();
        let generator = MlmBatchGenerator::new(tokenizer, 8);

        let sequences = vec!["ATCGATCG".to_string(), "GCTAGCTA".to_string()];
        let seed = [42u8; 32];
        let mlm = generator.generate_from_sequences(&sequences, &seed, 1).unwrap();

        assert_eq!(mlm.batch_size, 2);
        assert_eq!(mlm.seq_len, 8);
        assert_eq!(mlm.input_ids.len(), 16);
        assert_eq!(mlm.labels.len(), 16);
        assert_eq!(mlm.mask.len(), 16);

        // Determinism: same seed should produce the same batch.
        let mlm2 = generator.generate_from_sequences(&sequences, &seed, 1).unwrap();
        assert_eq!(mlm, mlm2);
    }

    #[test]
    fn test_reverse_complement() {
        assert_eq!(reverse_complement("ATCG"), "CGAT");
        assert_eq!(reverse_complement("GCTA"), "TAGC");
        assert_eq!(reverse_complement("AAA"), "TTT");
    }
}
