use anyhow::{anyhow, Result};
use tokenizers::tokenizer::Tokenizer;

/// Wrapper around the Hugging Face `tokenizers` crate for DNA sequence tokenization.
/// This loads the tokenizer from the raw `tokenizer.json` bytes returned by the seed-node.
#[derive(Clone)]
pub struct DnaTokenizer {
    inner: Tokenizer,
    pub mask_token_id: u32,
    pub pad_token_id: u32,
    pub vocab_size: usize,
}

impl DnaTokenizer {
    /// Load a tokenizer from the raw `tokenizer.json` bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = Tokenizer::from_bytes(bytes).map_err(|e| anyhow!("Failed to parse tokenizer.json from bytes: {e}"))?;
        let vocab_size = inner.get_vocab_size(true);
        let mask_token_id = Self::find_special_token(&inner, &["<mask>", "[MASK]", "<MASK>"]).unwrap_or(0);
        let pad_token_id = Self::find_special_token(&inner, &["<pad>", "", "<|endoftext|>", "<PAD>"]).unwrap_or(0);
        Ok(Self { inner, mask_token_id, pad_token_id, vocab_size })
    }

    /// Encode a DNA sequence (or any text) into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self.inner.encode(text, add_special_tokens).map_err(|e| anyhow!("Failed to encode sequence: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids back into a string.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner.decode(ids, skip_special_tokens).map_err(|e| anyhow!("Failed to decode token ids: {e}"))
    }

    fn find_special_token(tokenizer: &Tokenizer, candidates: &[&str]) -> Option<u32> {
        candidates.iter().find_map(|&t| tokenizer.token_to_id(t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::bpe::{BPE, Vocab};
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
        tokenizer.add_special_tokens(&[
            AddedToken::from("<mask>", true),
            AddedToken::from("<pad>", true),
        ]);

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
}
