use anyhow::{anyhow, Result};
use tokenizers::tokenizer::Tokenizer;

/// Wrapper around the Hugging Face `tokenizers` crate for DNA sequence tokenization.
/// This loads the tokenizer from the raw `tokenizer.json` bytes returned by the seed-node.
#[derive(Clone)]
pub struct DnaTokenizer {
    inner: Tokenizer,
    pub mask_token: String,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
    pub vocab_size: usize,
}

impl DnaTokenizer {
    /// Load a tokenizer from the raw `tokenizer.json` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = Tokenizer::from_bytes(bytes).map_err(|e| anyhow!("Failed to parse tokenizer.json from bytes: {e}"))?;
        let vocab_size = inner.get_vocab_size(true);
        let (mask_token, mask_token_id) =
            Self::find_special_token(&inner, &["<mask>", "[MASK]", "<MASK>"]).unwrap_or_else(|| ("<mask>".to_string(), 0));
        let pad_token_id = Self::find_special_token(&inner, &["<pad>", "", "<|endoftext|>", "<PAD>"]).map(|(_, id)| id).unwrap_or(0);
        Ok(Self { inner, mask_token, mask_token_id, pad_token_id, vocab_size })
    }

    /// Return the mask token string (e.g. `"<mask>"`).
    pub fn mask_token(&self) -> &str {
        &self.mask_token
    }

    /// Return the mask token id.
    pub fn mask_token_id(&self) -> u32 {
        self.mask_token_id
    }

    /// Return the full token vocabulary as id -> token string.
    pub fn token_strings(&self) -> Vec<String> {
        let mut id_to_token = vec![String::new(); self.vocab_size];
        for (token, id) in self.inner.get_vocab(true) {
            let idx = id as usize;
            if idx < self.vocab_size {
                id_to_token[idx] = token;
            }
        }
        id_to_token
    }

    /// Return the token id -> token string mapping for tokens whose string
    /// consists only of DNA bases (A, T, C, G).  This is useful for generating
    /// synthetic BPE/k-mer sequences that the tokenizer will recognise.
    pub fn dna_token_ids(&self) -> Vec<u32> {
        self.inner
            .get_vocab(true)
            .iter()
            .filter(|(token, _)| !token.is_empty() && token.chars().all(|c| matches!(c.to_ascii_uppercase(), 'A' | 'T' | 'C' | 'G')))
            .map(|(_, id)| *id)
            .collect()
    }

    /// Encode a DNA sequence (or any text) into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special_tokens).map_err(|e| anyhow!("Failed to encode sequence: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids back into a string using the Tokenizer's default decoder.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow!("Failed to decode token ids: {e}"))
    }

    /// Decode token ids by concatenating token strings directly, skipping any special tokens.
    ///
    /// This is required for DNABERT-2 style BPE tokenizers whose default decoder joins tokens
    /// with spaces. Concatenating the raw token strings preserves the continuous DNA sequence.
    pub fn decode_to_sequence(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        let mut result = String::new();
        for id in ids {
            if let Some(token) = self.inner.id_to_token(*id) {
                if skip_special_tokens && self.inner.get_added_vocabulary().is_special_token(&token) {
                    continue;
                }
                result.push_str(&token);
            }
        }
        Ok(result)
    }

    fn find_special_token(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<(String, u32)> {
        candidates.iter().find_map(|&t| tokenizer.token_to_id(t).map(|id| (t.to_string(), id)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{Vocab, BPE};
    use tokenizers::tokenizer::AddedToken;

    fn build_test_tokenizer_bytes() -> Vec<u8> {
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

        serde_json::to_vec(&tokenizer).expect("failed to serialize test tokenizer")
    }

    #[test]
    fn test_from_bytes_and_encode() {
        let bytes = build_test_tokenizer_bytes();
        let tokenizer = DnaTokenizer::from_bytes(&bytes).unwrap();

        let ids = tokenizer.encode("ATCG", false).unwrap();
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_special_token_ids() {
        let bytes = build_test_tokenizer_bytes();
        let tokenizer = DnaTokenizer::from_bytes(&bytes).unwrap();

        assert_eq!(tokenizer.mask_token_id, 4);
        assert_eq!(tokenizer.pad_token_id, 5);
    }

    #[test]
    fn test_decode_to_sequence_concatenates_no_spaces() {
        let bytes = build_test_tokenizer_bytes();
        let tokenizer = DnaTokenizer::from_bytes(&bytes).unwrap();

        let ids = vec![0, 1, 2, 3];
        assert_eq!(tokenizer.decode_to_sequence(&ids, false).unwrap(), "ATCG");

        // Skipping special tokens removes <mask>/<pad> without inserting spaces.
        let ids_with_mask = vec![0, tokenizer.mask_token_id, 2, tokenizer.pad_token_id, 3];
        assert_eq!(tokenizer.decode_to_sequence(&ids_with_mask, true).unwrap(), "ACG");
    }

    fn build_bpe_test_tokenizer_bytes() -> Vec<u8> {
        let mut vocab: Vocab = Vocab::new();
        vocab.insert("A".to_string(), 0);
        vocab.insert("T".to_string(), 1);
        vocab.insert("C".to_string(), 2);
        vocab.insert("G".to_string(), 3);
        vocab.insert("<mask>".to_string(), 4);
        vocab.insert("<pad>".to_string(), 5);

        // Add 2-mers so the test tokenizer exercises BPE/k-mer tokenization.
        let bases = ['A', 'T', 'C', 'G'];
        let mut id = 6u32;
        for a in bases {
            for b in bases {
                let mut kmer = String::with_capacity(2);
                kmer.push(a);
                kmer.push(b);
                vocab.insert(kmer, id);
                id += 1;
            }
        }

        let bpe = BPE::new(vocab, vec![]);
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer.add_special_tokens(&[AddedToken::from("<mask>", true), AddedToken::from("<pad>", true)]);

        serde_json::to_vec(&tokenizer).expect("failed to serialize test tokenizer")
    }

    #[test]
    fn test_bpe_tokenizer_ids_and_strings() {
        let bytes = build_bpe_test_tokenizer_bytes();
        let tokenizer = DnaTokenizer::from_bytes(&bytes).unwrap();

        let dna_ids = tokenizer.dna_token_ids();
        // 4 single bases + 16 2-mers.
        assert_eq!(dna_ids.len(), 20, "expected 20 DNA-only tokens");

        let token_strings = tokenizer.token_strings();
        assert_eq!(token_strings.len(), tokenizer.vocab_size);
        assert_eq!(token_strings[0], "A");
        assert_eq!(token_strings[4], "<mask>");
    }

    #[test]
    fn test_bpe_encode_decode() {
        let bytes = build_bpe_test_tokenizer_bytes();
        let tokenizer = DnaTokenizer::from_bytes(&bytes).unwrap();

        // "ATCG" should decode back to itself regardless of whether the BPE
        // tokenizer preferred 2-mers or single bases.
        let ids = tokenizer.encode("ATCG", false).unwrap();
        assert!(!ids.is_empty(), "tokenizer produced no tokens for ATCG");
        assert_eq!(tokenizer.decode_to_sequence(&ids, false).unwrap(), "ATCG");
    }
}
