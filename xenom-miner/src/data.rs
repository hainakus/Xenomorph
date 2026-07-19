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
}

/// Generates synthetic DNA training batches with random MLM masking.
#[derive(Clone)]
pub struct MlmBatchGenerator {
    tokenizer: DnaTokenizer,
    seq_len: usize,
    mask_prob: f64,
    dna_bases: Vec<char>,
}

impl MlmBatchGenerator {
    pub const DEFAULT_SEQ_LEN: usize = 128;
    pub const DEFAULT_MASK_PROB: f64 = 0.15;

    /// Create a new generator with the given tokenizer and sequence length.
    pub fn new(tokenizer: DnaTokenizer, seq_len: usize) -> Self {
        Self { tokenizer, seq_len, mask_prob: Self::DEFAULT_MASK_PROB, dna_bases: vec!['A', 'T', 'C', 'G'] }
    }

    /// Override the default 15% MLM mask probability.
    pub fn with_mask_prob(mut self, mask_prob: f64) -> Self {
        self.mask_prob = mask_prob.clamp(0.0, 1.0);
        self
    }

    /// Generate an MLM batch from a list of DNA sequences.
    ///
    /// Each sequence is tokenized, truncated or padded to `seq_len`, and masked
    /// using an RNG seeded from `seed` and `batch_id`.
    pub fn generate_from_sequences(&self, sequences: &[String], seed: &[u8; 32], batch_id: u64) -> Result<MlmBatch> {
        let batch_size = sequences.len();
        let total_len = batch_size * self.seq_len;

        let mut input_ids = vec![self.tokenizer.pad_token_id; total_len];
        let token_type_ids = vec![0u32; total_len];
        let mut attention_mask = vec![0u32; total_len];
        let mut labels = vec![u32::MAX; total_len];
        let mut mask = vec![0u8; total_len];

        let base_seed = u64::from_le_bytes(seed[..8].try_into().unwrap_or([0u8; 8])) ^ batch_id;

        for (b, sequence) in sequences.iter().enumerate() {
            let mut rng = ChaCha8Rng::seed_from_u64(base_seed.wrapping_add(b as u64));

            let mut encoded = self.tokenizer.encode(sequence, false)?;
            encoded.truncate(self.seq_len);

            let offset = b * self.seq_len;
            for (i, &original_id) in encoded.iter().enumerate() {
                let pos = offset + i;
                attention_mask[pos] = 1;

                if rng.gen::<f64>() < self.mask_prob {
                    mask[pos] = 1;
                    labels[pos] = original_id;
                    input_ids[pos] = self.tokenizer.mask_token_id;
                } else {
                    input_ids[pos] = original_id;
                    labels[pos] = u32::MAX;
                }
            }
        }

        Ok(MlmBatch {
            input_ids,
            token_type_ids,
            attention_mask,
            labels,
            mask,
            seq_len: self.seq_len,
            batch_size,
        })
    }

    /// Generate an MLM batch from a `TrainingBatch`.
    pub fn generate(&self, batch: &TrainingBatch) -> Result<MlmBatch> {
        let batch_size = batch.data_indices.len();
        let total_len = batch_size * self.seq_len;

        let mut input_ids = vec![self.tokenizer.pad_token_id; total_len];
        let token_type_ids = vec![0u32; total_len];
        let mut attention_mask = vec![0u32; total_len];
        let mut labels = vec![u32::MAX; total_len];
        let mut mask = vec![0u8; total_len];

        for (b, &index) in batch.data_indices.iter().enumerate() {
            let mut rng = Self::seeded_rng(&batch.base_checkpoint, index);

            // Generate a random DNA string long enough to tokenize into at least seq_len ids.
            let raw_len = (self.seq_len * 4).max(16);
            let sequence: String = (0..raw_len).map(|_| self.dna_bases[rng.gen_range(0..4)]).collect();

            let mut encoded = self.tokenizer.encode(&sequence, false)?;
            encoded.truncate(self.seq_len);

            let offset = b * self.seq_len;
            for (i, &original_id) in encoded.iter().enumerate() {
                let pos = offset + i;
                attention_mask[pos] = 1;

                if rng.gen::<f64>() < self.mask_prob {
                    mask[pos] = 1;
                    labels[pos] = original_id;
                    input_ids[pos] = self.tokenizer.mask_token_id;
                } else {
                    input_ids[pos] = original_id;
                    labels[pos] = u32::MAX;
                }
            }
        }

        Ok(MlmBatch {
            input_ids,
            token_type_ids,
            attention_mask,
            labels,
            mask,
            seq_len: self.seq_len,
            batch_size,
        })
    }

    fn seeded_rng(base_checkpoint: &[u8; 32], index: u64) -> ChaCha8Rng {
        let seed_bytes: [u8; 8] = base_checkpoint[..8].try_into().expect("base_checkpoint has 32 bytes");
        let seed = u64::from_le_bytes(seed_bytes) ^ index;
        ChaCha8Rng::seed_from_u64(seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{BPE, Vocab};
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
        tokenizer.add_special_tokens(&[
            AddedToken::from("<mask>", true),
            AddedToken::from("<pad>", true),
        ]);

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
}
